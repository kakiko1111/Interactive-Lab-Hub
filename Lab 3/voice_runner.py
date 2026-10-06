#!/usr/bin/env python3
"""Voice-driven runner for the Mini PiTFT.

Make any sound and the character runs right. Make a sharp sound -- a clap, a
"pa!", a "tk!" -- and it jumps.

Two separate signals are pulled out of the same 20ms audio frame:

  loudness      RMS in dBFS, compared against a noise floor measured at startup.
                Above the gate, the character runs. This is deliberately dumb:
                it does not care what you say, only that you are making sound.

  onset         Normalized spectral flux -- the positive frame-to-frame change
                in the magnitude spectrum, divided by the frame's own total
                magnitude. A clap puts energy into every bin at once, so nearly
                all of the frame's magnitude is new and the ratio approaches 1.
                A vowel you are already holding scores near 0 however loud it
                is, because its spectrum is barely changing frame to frame.

                Dividing by the frame's magnitude is what makes this work with
                one fixed threshold: it measures the *proportion* of the sound
                that is new, so a clap reads the same whether you are close to
                the mic or across the room, and a loud steady sound never
                creeps over the line the way a raw-flux threshold lets it.

                So the jump fires on the *attack* of a sound and then stays
                quiet while you hold it: one "aaaah" is one jump, however long
                and loud you drag it out.

The screen shows the input level and the gate line at all times, so the person
talking to it can see what it is hearing and why it did or didn't react.

    (.venv) $ python voice_runner.py
    (.venv) $ python voice_runner.py --no-display   # tune audio without the Pi
"""

import argparse
import collections
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

# --- audio ---------------------------------------------------------------

SAMPLE_RATE = 16000
FRAME = 320          # 20ms, matches the hop the VAD in listen.py uses
CALIBRATION_SEC = 1.0
# One clap spans several frames; without a refractory its decay fires a second
# jump. 9 frames = 180ms, comfortably longer than a clap and shorter than the
# gap between two deliberate claps.
REFRACTORY_FRAMES = 9

# The run gate is a Schmitt trigger, not a bare threshold: it takes GATE_ON to
# start running and drops out only GATE_HYSTERESIS_DB lower. With one threshold
# a voice sitting near the line toggles the character on and off frame by frame,
# which reads as a stutter rather than as running.
GATE_HYSTERESIS_DB = 5.0
# Level feeding the gate is an envelope follower: instant attack so the
# character starts the moment you do, slow release so the gaps between
# syllables do not stop it dead.
ENVELOPE_RELEASE = 0.12
# The noise floor tracks the room asymmetrically: it falls fast and creeps up
# slowly. Gating this on "only while idle" does not work -- once a louder room
# trips the gate, an idle-only tracker freezes exactly when it is needed. Since
# speech is intermittent, the fast fall wins during the gaps and your own voice
# cannot drag the floor up over itself; a genuinely louder room, sustained for
# tens of seconds, does move it.
FLOOR_FALL = 0.05     # ~0.4s to follow the room down
FLOOR_RISE = 0.0008   # ~25s to concede the room got louder
FLOOR_MIN_DB, FLOOR_MAX_DB = -75.0, -20.0

# --- screen (Adafruit Mini PiTFT 1.14", landscape) -----------------------

WIDTH, HEIGHT = 240, 135
ROTATION = 90
GROUND_Y = 118
FPS = 30

# --- character physics (pixels, seconds) ---------------------------------

RUN_SPEED = 95.0
GRAVITY = 900.0
JUMP_VELOCITY = -330.0   # apex ~60px, hang time ~0.73s
CHAR_SIZE = 16
CHAR_X = 64

# --- palette -------------------------------------------------------------

BG = (18, 18, 24)
GROUND = (70, 72, 90)
CHAR = (242, 201, 76)
CHAR_AIR = (247, 224, 140)
OBSTACLE = (219, 88, 96)
METER_BG = (38, 38, 48)
METER_FG = (108, 196, 142)
GATE_MARK = (232, 232, 240)
INK = (170, 172, 190)


class AudioFeatures:
    """Reads the mic and exposes the two numbers the game needs.

    Runs sounddevice's callback on its own thread; everything the render loop
    touches is behind `lock`. Onsets are delivered through a queue rather than a
    flag so a clap can never be dropped by landing between two frames.
    """

    def __init__(self, gate_margin_db, jump_threshold, device=None):
        self.gate_margin_db = gate_margin_db
        self.jump_threshold = jump_threshold
        self.device = device

        self.lock = threading.Lock()
        self.level_db = -90.0
        self.gate_db = -40.0
        self.flux = 0.0
        self.running = False

        self._env_db = -90.0
        self._floor_db = -60.0

        self.onsets = queue.Queue()
        # Raw audio for the speech thread. One InputStream feeds both paths:
        # opening a second one on the same ALSA capture device fails on the Pi.
        # Bounded, so if the speech thread stalls we drop old audio instead of
        # growing without limit.
        self.pcm = collections.deque(maxlen=250)   # ~5s at 20ms a frame

        self._window = np.hanning(FRAME).astype(np.float32)
        self._prev_mag = None
        # Refractory is counted in audio frames, not wall clock: the frames are
        # what actually measure elapsed sound, and a scheduling hiccup on the
        # audio thread must not widen or shrink the window.
        self._frames_seen = 0
        self._last_onset_frame = -REFRACTORY_FRAMES
        self._calibrating = True
        self._noise_samples = []
        self._stream = None

    # -- called on the audio thread ---------------------------------------

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"audio: {status}", file=sys.stderr)

        mono = indata[:, 0].astype(np.float32)
        self.pcm.append(mono.copy())

        rms = float(np.sqrt(np.mean(mono**2)) + 1e-10)
        level_db = 20.0 * np.log10(rms)

        mag = np.abs(np.fft.rfft(mono * self._window))
        if self._prev_mag is None:
            flux = 0.0
        else:
            rise = float(np.sum(np.maximum(0.0, mag - self._prev_mag)))
            flux = rise / (float(np.sum(mag)) + 1e-9)
        self._prev_mag = mag

        if self._calibrating:
            self._noise_samples.append(level_db)
            return

        # Envelope: jump straight up to a new peak, ease back down.
        if level_db > self._env_db:
            self._env_db = level_db
        else:
            self._env_db += (level_db - self._env_db) * ENVELOPE_RELEASE

        on_db = self._floor_db + self.gate_margin_db
        off_db = on_db - GATE_HYSTERESIS_DB
        running = self._env_db > (off_db if self.running else on_db)

        rate = FLOOR_FALL if level_db < self._floor_db else FLOOR_RISE
        self._floor_db += (level_db - self._floor_db) * rate
        self._floor_db = min(max(self._floor_db, FLOOR_MIN_DB), FLOOR_MAX_DB)

        with self.lock:
            self.level_db = level_db
            self.flux = flux
            self.gate_db = on_db
            self.running = running

        self._frames_seen += 1
        loud_enough = level_db > on_db + 6.0
        rested = self._frames_seen - self._last_onset_frame > REFRACTORY_FRAMES
        if flux > self.jump_threshold and loud_enough and rested:
            self._last_onset_frame = self._frames_seen
            self.onsets.put(self._frames_seen)

    # -- called on the main thread ----------------------------------------

    def start(self):
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

        print(f"calibrating noise floor, stay quiet for {CALIBRATION_SEC:.0f}s...")
        time.sleep(CALIBRATION_SEC)
        self._calibrating = False

        if self._noise_samples:
            floor = float(np.percentile(self._noise_samples, 90))
        else:
            floor = -60.0
        floor = min(max(floor, FLOOR_MIN_DB), FLOOR_MAX_DB)
        self._floor_db = floor
        self._env_db = floor
        with self.lock:
            self.gate_db = floor + self.gate_margin_db
        print(f"noise floor {floor:.1f} dBFS, gate {floor + self.gate_margin_db:.1f} dBFS")

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def read(self):
        with self.lock:
            return self.level_db, self.gate_db, self.flux, self.running

    def drain_pcm(self):
        out = []
        while True:
            try:
                out.append(self.pcm.popleft())
            except IndexError:
                break
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)

    def take_onset(self):
        got = False
        while True:
            try:
                self.onsets.get_nowait()
                got = True
            except queue.Empty:
                return got


# Whisper hears "Go.", "go!", "Jump, jump" and once in a while "jumped", so
# each command accepts a few variants. Matching is on whole words pulled out
# with a regex rather than a substring test: `"go" in text` would fire on
# "ago", "golf" and "good", which is the kind of bug you only find in a demo.
COMMAND_WORDS = {
    "go": ("go", "going", "run"),
    "stop": ("stop", "stopped", "halt", "freeze"),
    "jump": ("jump", "jumps", "jumped", "hop"),
}


class SpeechCommands(threading.Thread):
    """Slow path: Silero VAD endpoints the stream, whisper transcribes, and a
    recognised word is posted back to the render loop.

    This runs on its own thread because a transcription takes about a second on
    the Pi -- doing it inline would freeze the animation for thirty frames. It
    deliberately shares AudioFeatures' stream rather than opening its own.

    Nothing here can move the character quickly. Measured in Part C, the round
    trip is min-silence plus transcription, ~1.5s, where the acoustic path is
    ~30ms. That gap is the point: state changes ride the slow path, motion does
    not.
    """

    def __init__(self, audio, vad_path, model_name, min_silence):
        super().__init__(daemon=True)
        self.audio = audio
        self.vad_path = vad_path
        self.model_name = model_name
        self.min_silence = min_silence
        self.commands = queue.Queue()
        self.thinking = False
        self.ready = False
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        import sherpa_onnx
        from faster_whisper import WhisperModel

        recognizer = WhisperModel(self.model_name, device="cpu", compute_type="int8")

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(self.vad_path)
        config.silero_vad.min_silence_duration = self.min_silence
        config.silero_vad.min_speech_duration = 0.12   # command words are short
        config.sample_rate = SAMPLE_RATE
        vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
        window = config.silero_vad.window_size

        self.ready = True
        pending = np.empty(0, dtype=np.float32)

        while not self._stop.is_set():
            chunk = self.audio.drain_pcm()
            if chunk.size == 0:
                time.sleep(0.02)
                continue
            pending = np.concatenate([pending, chunk])
            while len(pending) > window:
                vad.accept_waveform(pending[:window])
                pending = pending[window:]

            while not vad.empty():
                utterance = np.array(vad.front.samples, dtype=np.float32)
                vad.pop()
                if utterance.size < SAMPLE_RATE * 0.1:
                    continue

                self.thinking = True
                t0 = time.perf_counter()
                segments, _ = recognizer.transcribe(utterance, beam_size=1)
                text = " ".join(seg.text for seg in segments)
                think = time.perf_counter() - t0
                self.thinking = False

                words = set(re.findall(r"[a-z]+", text.lower()))
                for name, variants in COMMAND_WORDS.items():
                    if words.intersection(variants):
                        # Total lag the person actually feels: the endpointing
                        # wait before we even knew they stopped, plus whisper.
                        self.commands.put((name, self.min_silence + think, text.strip()))
                        break

    def take(self):
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return None


class World:
    """Character, scrolling ground, and obstacles.

    The character holds station at CHAR_X and the world slides left underneath,
    so running is open-ended rather than ending at the right edge of a 240px
    screen.
    """

    def __init__(self, obstacles=True):
        self.y = float(GROUND_Y - CHAR_SIZE)
        self.vy = 0.0
        self.grounded = True
        self.distance = 0.0
        self.scroll = 0.0
        self.obstacles_on = obstacles
        self.obstacles = []
        self._next_spawn = 260.0
        self.hits = 0
        self._hit_flash = 0.0
        # Set by "go"/"stop". Authoritative for running, because loudness
        # cannot be: saying the word "stop" is itself a sound, so a
        # loudness-driven gate would restart the character on the very
        # utterance meant to halt it.
        self.latched = False

    @property
    def ground_y(self):
        return float(GROUND_Y - CHAR_SIZE)

    def jump(self):
        if self.grounded:
            self.vy = JUMP_VELOCITY
            self.grounded = False

    def update(self, dt, running):
        if running:
            step = RUN_SPEED * dt
            self.distance += step
            self.scroll += step

        self.vy += GRAVITY * dt
        self.y += self.vy * dt
        if self.y >= self.ground_y:
            self.y = self.ground_y
            self.vy = 0.0
            self.grounded = True

        if self._hit_flash > 0.0:
            self._hit_flash -= dt

        if not self.obstacles_on:
            return

        # Obstacles live in world coordinates and are culled once they pass the
        # character, so the list stays a handful of entries no matter how far
        # the player runs.
        if self.distance > self._next_spawn:
            self.obstacles.append(self.distance + WIDTH)
            self._next_spawn = self.distance + float(np.random.uniform(150, 320))

        char_top = self.y
        char_bottom = self.y + CHAR_SIZE
        for pos in list(self.obstacles):
            x = CHAR_X + (pos - self.distance)
            if x < -20:
                self.obstacles.remove(pos)
                continue
            overlaps_x = x < CHAR_X + CHAR_SIZE and x + 10 > CHAR_X
            clears_it = char_bottom <= GROUND_Y - 18
            if overlaps_x and not clears_it and char_top < GROUND_Y:
                if self._hit_flash <= 0.0:
                    self.hits += 1
                    self._hit_flash = 0.4
                self.obstacles.remove(pos)

    def hit_flash(self):
        return self._hit_flash > 0.0


def draw_frame(draw, world, level_db, gate_db, running, thinking=False, heard=None):
    draw.rectangle((0, 0, WIDTH, HEIGHT), fill=BG)

    # Ground, with tick marks that slide left so motion is visible even on a
    # blank stretch with no obstacles in view.
    draw.line((0, GROUND_Y, WIDTH, GROUND_Y), fill=GROUND, width=2)
    offset = int(world.scroll) % 24
    for i in range(-1, WIDTH // 24 + 2):
        x = i * 24 - offset
        draw.line((x, GROUND_Y + 3, x + 6, GROUND_Y + 3), fill=GROUND, width=1)

    if world.obstacles_on:
        for pos in world.obstacles:
            x = CHAR_X + (pos - world.distance)
            if -20 < x < WIDTH + 20:
                draw.rectangle((x, GROUND_Y - 18, x + 10, GROUND_Y), fill=OBSTACLE)

    body = CHAR if world.grounded else CHAR_AIR
    if world.hit_flash():
        body = OBSTACLE
    draw.rectangle(
        (CHAR_X, world.y, CHAR_X + CHAR_SIZE, world.y + CHAR_SIZE),
        fill=body,
    )
    # Eye faces the direction of travel, so the character reads as running
    # rather than sliding.
    draw.rectangle((CHAR_X + 10, world.y + 4, CHAR_X + 13, world.y + 7), fill=BG)

    # Input meter. This is the part that tells the person what the device is
    # hearing: bar is live level, the notch is the gate it has to clear.
    mx, my, mw, mh = 8, 8, 132, 6
    draw.rectangle((mx, my, mx + mw, my + mh), fill=METER_BG)

    lo, hi = -60.0, -5.0
    frac = (max(lo, min(hi, level_db)) - lo) / (hi - lo)
    if frac > 0.01:
        draw.rectangle((mx, my, mx + int(mw * frac), my + mh), fill=METER_FG)

    gate_frac = (max(lo, min(hi, gate_db)) - lo) / (hi - lo)
    gx = mx + int(mw * gate_frac)
    draw.line((gx, my - 3, gx, my + mh + 3), fill=GATE_MARK, width=1)

    if thinking:
        state = "thinking..."
    elif running:
        state = "RUNNING"
    else:
        state = "listening"
    draw.text((8, 20), state, fill=CHAR if running else INK)
    label = f"{int(world.distance)}m"
    if world.obstacles_on:
        label += f"   hits {world.hits}"
    draw.text((WIDTH - 8 - 6 * len(label), 20), label, fill=INK)

    # What the slow path last understood, and how long it took. Printing the
    # lag on screen is the honest thing to do: the person can see that the
    # device heard them a beat ago, rather than guessing it ignored them.
    if heard is not None:
        word, lag = heard
        draw.text((8, HEIGHT - 14), f'heard "{word}"  {lag:.1f}s ago', fill=INK)


def open_display():
    import board
    import digitalio
    import adafruit_rgb_display.st7789 as st7789

    cs_pin = digitalio.DigitalInOut(board.D5)
    dc_pin = digitalio.DigitalInOut(board.D25)
    disp = st7789.ST7789(
        board.SPI(),
        cs=cs_pin,
        dc=dc_pin,
        rst=None,
        baudrate=64000000,
        width=135,
        height=240,
        x_offset=53,
        y_offset=40,
    )
    backlight = digitalio.DigitalInOut(board.D22)
    backlight.switch_to_output()
    backlight.value = True
    return disp, backlight


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate-margin", type=float, default=8.0,
                    help="dB above the measured noise floor before the character runs")
    ap.add_argument("--jump-threshold", type=float, default=0.70,
                    help="fraction of the frame's spectrum that must be new to "
                         "count as a jump, 0-1. Lower = jumps more easily.")
    ap.add_argument("--no-speech", action="store_true",
                    help="skip whisper/VAD entirely -- acoustic control only")
    ap.add_argument("--acoustic-run", action="store_true",
                    help="any sound runs the character, as before. 'stop' cannot "
                         "work reliably in this mode (saying it re-triggers the gate)")
    ap.add_argument("--whisper-model", default="tiny.en",
                    help="smaller is faster; command words need little accuracy")
    ap.add_argument("--min-silence", type=float, default=0.35,
                    help="seconds of silence that end a spoken command")
    ap.add_argument("--no-obstacles", action="store_true",
                    help="just run and jump, nothing to clear")
    ap.add_argument("--no-display", action="store_true",
                    help="print to terminal instead of the PiTFT")
    ap.add_argument("--device", default=None, help="input device index or name")
    args = ap.parse_args()

    # With speech off, the latch can never be set, so loudness has to drive
    # the running or nothing moves the character at all.
    if args.no_speech and not args.acoustic_run:
        args.acoustic_run = True

    device = args.device
    if device is not None and device.isdigit():
        device = int(device)

    audio = AudioFeatures(args.gate_margin, args.jump_threshold, device)
    world = World(obstacles=not args.no_obstacles)

    speech = None
    if not args.no_speech:
        vad_path = Path(__file__).resolve().parent / "models" / "silero_vad.onnx"
        if not vad_path.is_file():
            sys.exit(f"VAD model not found at {vad_path}. Run speech-scripts/setup.sh, "
                     f"or pass --no-speech for acoustic control only.")
        speech = SpeechCommands(audio, vad_path, args.whisper_model, args.min_silence)

    disp = backlight = image = draw = None
    if not args.no_display:
        from PIL import Image, ImageDraw
        disp, backlight = open_display()
        image = Image.new("RGB", (WIDTH, HEIGHT))
        draw = ImageDraw.Draw(image)

    audio.start()

    if speech is not None:
        print(f"loading whisper ({args.whisper_model})...")
        speech.start()
        while not speech.ready:
            time.sleep(0.05)
        print('say "go" to start, "stop" to halt, "jump" or clap to jump.')
    else:
        print("make noise to run, clap to jump.")
    print("ctrl-c to stop.")

    frame_time = 1.0 / FPS
    last = time.monotonic()
    last_clap = -99.0
    heard = None
    heard_at = 0.0
    try:
        while True:
            now = time.monotonic()
            dt = min(now - last, 0.1)   # clamp so a stall can't teleport anyone
            last = now

            level_db, gate_db, flux, running = audio.read()

            if audio.take_onset():
                world.jump()
                last_clap = now

            if speech is not None:
                got = speech.take()
                if got is not None:
                    word, lag, raw = got
                    if word == "go":
                        world.latched = True
                    elif word == "stop":
                        world.latched = False
                    elif word == "jump":
                        # Saying "jump" is itself a sharp sound, so the fast
                        # path has usually already fired for this very
                        # utterance. Jumping again when the transcript lands
                        # would read as a stutter, not a second command.
                        if now - last_clap > lag + 0.5:
                            world.jump()
                    heard = (word, lag)
                    heard_at = now
                    print(f'  heard "{word}" after {lag:.2f}s  (transcript: {raw!r})')

            if heard is not None and now - heard_at > 4.0:
                heard = None

            if not args.acoustic_run:
                running = world.latched
            world.update(dt, running)

            if disp is not None:
                draw_frame(draw, world, level_db, gate_db, running,
                           thinking=speech is not None and speech.thinking,
                           heard=heard)
                disp.image(image, ROTATION)
            else:
                # flux is printed so you can watch what your own claps score
                # and pick --jump-threshold from real numbers, not guesswork.
                bar = "#" * max(0, int((level_db + 60) / 2))
                air = " ^" if not world.grounded else "  "
                print(f"\r{level_db:6.1f}dB |{bar:<28}|{air} flux {flux:4.2f} "
                      f"{int(world.distance):4d}m hits {world.hits}",
                      end="", flush=True)

            slack = frame_time - (time.monotonic() - now)
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        print(f"\nran {int(world.distance)}m, {world.hits} hits")
    finally:
        if speech is not None:
            speech.stop()
        audio.stop()
        # Leave the backlight on and post a final frame. Killing the backlight
        # here looks identical to a crashed Pi, and you stop and start this
        # constantly while tuning thresholds.
        if disp is not None:
            draw.rectangle((0, 0, WIDTH, HEIGHT), fill=BG)
            draw.text((8, HEIGHT // 2 - 8), f"stopped  --  {int(world.distance)}m", fill=INK)
            disp.image(image, ROTATION)


if __name__ == "__main__":
    main()
