#!/usr/bin/env python3
"""Local speech detection and transcription for the robot."""

import asyncio
import math
import os
import threading
import time
from collections import deque
from typing import Awaitable, Callable, Optional

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


class SpeechDetector:
    """
    Detect utterances from microphone volume and transcribe each completed
    utterance once. Faster-Whisper's Silero VAD removes non-speech audio before
    transcription, while explicit pause controls exclude the robot's own voice.
    """

    def __init__(
        self,
        model_size: str = "tiny.en",
        sample_rate: int = 16000,
        channels: int = 1,
        processing_window: float = 1.0,
        overlap: float = 0.2,
        on_partial_speech: Optional[Callable[[str], None]] = None,
        on_final_speech: Optional[Callable[[str], None]] = None,
    ):
        # Keep these arguments for compatibility with earlier assignment versions.
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.channels = channels
        self.processing_window = processing_window
        self.overlap = overlap
        self.on_partial_speech = on_partial_speech
        self.on_final_speech = on_final_speech

        self.stream = None
        self.whisper_model = None
        self.running = False
        self.vad_running = False
        self.is_recording = False
        self.recording_start_time = None

        self.blocksize = 1024
        self.pre_roll_chunks = deque(maxlen=5)
        max_chunks = math.ceil(30 * sample_rate / self.blocksize)
        self.recording_chunks = deque(maxlen=max_chunks)
        self.buffer_lock = threading.Lock()

        self.current_transcription = ""
        self.final_transcription = ""
        self.transcription_lock = threading.Lock()

        self._pause_evt = threading.Event()
        self._skip_silence_evt = threading.Event()
        self._latest_spl = None

        print(f"SpeechDetector initialized with Faster-Whisper {model_size}")

    def list_audio_devices(self):
        """List available audio input devices."""
        print("\nAvailable audio devices:")
        print(sd.query_devices())

    def _load_whisper_model(self):
        """Load a quantized CPU model suitable for student hardware."""
        if self.whisper_model is not None:
            return True

        print(f"Loading Faster-Whisper {self.model_size} model...")
        try:
            cpu_threads = max(1, min(4, os.cpu_count() or 1))
            self.whisper_model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=cpu_threads,
                num_workers=1,
            )
        except Exception as error:
            print(f"❌ Error loading Faster-Whisper: {error}")
            return False

        print("✅ Faster-Whisper model loaded")
        return True

    def start(self):
        """Load the model and start microphone capture."""
        if self.running:
            return True
        if not self._load_whisper_model():
            return False

        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            print(f"Using audio input device: {devices[default_input]['name']}")
        except Exception as error:
            print(f"Warning: Could not query audio devices: {error}")

        self._clear_audio()
        self._pause_evt.clear()
        self.running = True
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                callback=self._audio_callback,
                blocksize=self.blocksize,
            )
            self.stream.start()
        except Exception as error:
            print(f"❌ Error starting audio stream: {error}")
            self.running = False
            self.stream = None
            return False

        print("✅ Speech detector ready")
        return True

    def stop(self):
        """Stop microphone capture and voice activity detection."""
        self.running = False
        self.vad_running = False
        self._pause_evt.set()

        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None

        self._clear_audio()
        print("✅ Speech detector stopped")

    def _audio_callback(self, indata, frames, callback_time, status):
        """Capture a small pre-roll and the active utterance."""
        if status:
            print(f"Audio callback status: {status}")
        if not self.running or self._pause_evt.is_set():
            return

        try:
            if indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1, dtype=np.float32)
            else:
                audio_data = indata[:, 0]
            audio_chunk = np.asarray(audio_data, dtype=np.float32).copy()

            rms = float(np.sqrt(np.mean(audio_chunk * audio_chunk)))
            self._latest_spl = -np.inf if rms < 1e-10 else 20 * np.log10(rms)

            with self.buffer_lock:
                # Recheck under the lock so a simultaneous TTS pause cannot leak
                # robot audio into either buffer.
                if self._pause_evt.is_set():
                    return
                self.pre_roll_chunks.append(audio_chunk)
                if self.is_recording:
                    self.recording_chunks.append(audio_chunk)
        except Exception as error:
            print(f"Audio callback error: {error}")

    def _clear_audio(self):
        with self.buffer_lock:
            self.pre_roll_chunks.clear()
            self.recording_chunks.clear()
            self.is_recording = False
            self.recording_start_time = None
        self._latest_spl = None
        self._skip_silence_evt.clear()

    def start_recording(self):
        """Start an utterance with a short pre-speech prefix."""
        if not self.running or self._pause_evt.is_set() or self.is_recording:
            return

        with self.buffer_lock:
            if self._pause_evt.is_set():
                return
            self.recording_chunks.clear()
            self.recording_chunks.extend(self.pre_roll_chunks)
            self.is_recording = True
            self.recording_start_time = time.monotonic()

        with self.transcription_lock:
            self.current_transcription = ""
            self.final_transcription = ""
        print("🎤 Started recording")

    def stop_recording(self):
        """Finish the utterance and perform one transcription pass."""
        if not self.is_recording:
            return

        with self.buffer_lock:
            self.is_recording = False
            chunks = list(self.recording_chunks)
            self.recording_chunks.clear()
            self.pre_roll_chunks.clear()

        duration = (
            time.monotonic() - self.recording_start_time
            if self.recording_start_time is not None
            else 0.0
        )
        self.recording_start_time = None
        print(f"ℹ️  Stopped recording ({duration:.1f}s) - transcribing...")

        if not chunks:
            return
        audio = np.concatenate(chunks).astype(np.float32, copy=False)
        if len(audio) < int(0.4 * self.sample_rate):
            return

        final_text = self._transcribe(audio)
        with self.transcription_lock:
            self.current_transcription = final_text
            self.final_transcription = final_text

        if final_text and self.on_final_speech:
            self.on_final_speech(final_text)

    def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe speech using INT8 inference and Silero VAD filtering."""
        try:
            segments, _ = self.whisper_model.transcribe(
                audio,
                language="en",
                task="transcribe",
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
                without_timestamps=True,
                vad_filter=True,
                vad_parameters={
                    "min_speech_duration_ms": 150,
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200,
                },
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as error:
            print(f"Faster-Whisper transcription failed: {error}")
            return ""

        if self._is_hallucination(text):
            print(f"🚫 Ignoring likely Whisper hallucination: '{text[:100]}...'")
            return ""
        return text

    @staticmethod
    def _is_hallucination(text: str) -> bool:
        """Reject unusually long or strongly repetitive results."""
        if not text or len(text) < 50:
            return False
        if len(text) > 500:
            return True

        words = text.split()
        for index in range(max(0, len(words) - 6)):
            phrase = " ".join(words[index:index + 3])
            if " ".join(words[index + 3:]).count(phrase) >= 3:
                return True
        return False

    def get_current_transcription(self) -> str:
        with self.transcription_lock:
            return self.current_transcription

    def get_final_transcription(self) -> str:
        with self.transcription_lock:
            return self.final_transcription

    def is_currently_recording(self) -> bool:
        return self.is_recording

    def get_sound_pressure_level(self) -> Optional[float]:
        return self._latest_spl

    async def listen_with_voice_activity_detection(
        self,
        voice_threshold: float = -40,
        silence_timeout: float = 3.0,
        check_interval: float = 0.1,
        on_pause: Optional[Callable[[str], Awaitable[None]]] = None,
        on_end: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        """Detect utterance boundaries and transcribe until stopped."""
        if not self.running:
            raise RuntimeError("Speech handler must be started first")
        if self.vad_running:
            raise RuntimeError("Voice activity detection is already running")

        self.vad_running = True
        silence_started = None
        print("🎧 Listening for speech...")

        try:
            while self.vad_running and self.running:
                await self._wait_if_paused(check_interval)
                if self._pause_evt.is_set():
                    continue

                spl = self.get_sound_pressure_level()
                if spl is None:
                    await asyncio.sleep(check_interval)
                    continue

                voice_detected = spl > voice_threshold
                if voice_detected:
                    if not self.is_recording:
                        print(f"🎤 Voice detected ({spl:.1f} dB)")
                        self.start_recording()
                    silence_started = None
                    self._skip_silence_evt.clear()
                elif self.is_recording:
                    if silence_started is None:
                        silence_started = time.monotonic()
                        if on_pause:
                            await on_pause(self.get_current_transcription())

                    timed_out = time.monotonic() - silence_started >= silence_timeout
                    finalize_now = self._skip_silence_evt.is_set()
                    if timed_out or finalize_now:
                        self._skip_silence_evt.clear()
                        self.stop_recording()
                        silence_started = None
                        final_text = self.get_final_transcription()
                        if final_text and on_end:
                            await on_end(final_text)
                        print("🎧 Listening for speech...")

                await asyncio.sleep(check_interval)
        except Exception as error:
            print(f"Error in voice activity detection: {error}")
        finally:
            if self.is_recording:
                self.stop_recording()
                final_text = self.get_final_transcription()
                if final_text and on_end:
                    await on_end(final_text)
            self.vad_running = False

    async def _wait_if_paused(self, check_interval: float):
        while self._pause_evt.is_set() and self.vad_running and self.running:
            await asyncio.sleep(check_interval)

    def stop_voice_activity_detection(self):
        self.vad_running = False

    def is_voice_activity_detection_running(self) -> bool:
        return self.vad_running

    def pause_voice_detection(self, should_pause: bool):
        """Exclude all microphone audio while the robot is speaking."""
        if should_pause:
            self._pause_evt.set()
            self._clear_audio()
            print("🔇 Microphone transcription paused")
        else:
            # Clear before reopening the gate so buffered speaker tail cannot
            # trigger a new utterance.
            self._clear_audio()
            self._pause_evt.clear()
            print("🎤 Microphone transcription resumed")

    def voice_detection_is_paused(self) -> bool:
        return self._pause_evt.is_set()

    def get_immediate_final_transcription(self) -> None:
        if self.is_recording:
            self._skip_silence_evt.set()


if __name__ == "__main__":
    async def print_transcription(transcription: str):
        print(f"Transcription: {transcription}")

    async def run_example():
        detector = SpeechDetector()
        if detector.start():
            try:
                await detector.listen_with_voice_activity_detection(
                    on_end=print_transcription
                )
            finally:
                detector.stop()

    asyncio.run(run_example())
