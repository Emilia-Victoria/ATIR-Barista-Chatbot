import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice
import threading
import time
from typing import Optional, Callable
import queue


class RobotVoice:
    """
    Piper-based TTS with an internal text queue.
    Callbacks fire once per 'burst':
      - on_speech_start: when queue goes from idle -> active (first item enqueued).
      - on_speech_end: when playback drains and queue returns to idle (not interrupted).
      - on_speech_interrupted: when playback returns to idle due to interruption/flush.
    """

    def __init__(
        self,
        auto_start_stream: bool = True,
        on_speech_start: Optional[Callable[[str], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
        on_speech_interrupted: Optional[Callable[[], None]] = None,
    ):
        self.model_path = "en_US-amy-medium.onnx"
        self.voice = None
        self.stream = None

        # Playback/worker state
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._interrupt_flag = threading.Event()
        self._paused = threading.Event()
        self._paused.clear()
        self._speech_lock = threading.Lock()
        self._is_speaking = False
        self._current_text = ""

        # Burst/queue state
        self._queue_active = False           # True while we're in a burst (idle->active->idle)
        self._first_item_of_burst = None     # cache first text to pass to on_speech_start

        # Callbacks (now burst-level)
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_speech_interrupted = on_speech_interrupted

        self._load_voice_model()
        if auto_start_stream:
            self.start_stream()
        self._ensure_worker()

    # ---------- setup/teardown ----------

    def _load_voice_model(self):
        try:
            self.voice = PiperVoice.load(self.model_path)
            print(f"Loaded voice model: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load voice model {self.model_path}: {e}")

    def start_stream(self):
        if self.stream is not None:
            return
        try:
            self.stream = sd.OutputStream(
                samplerate=self.voice.config.sample_rate,
                channels=1,
                dtype="int16",
            )
            self.stream.start()
            print("Audio stream started")
        except Exception as e:
            raise RuntimeError(f"Failed to start audio stream: {e}")

    def stop_stream(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None
                print("Audio stream stopped")

    def _ensure_worker(self):
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    # ---------- public API ----------

    def enqueue(self, text: str):
        """
        Queue text for speaking. If we were idle, starts a new 'burst' and fires on_speech_start once.
        """
        if not text or not text.strip():
            return
        if self.stream is None:
            print("Audio stream not started. Call start_stream() first.")
            return

        # If we were idle, this begins a new burst.
        if not self._queue_active and self._queue.empty() and not self.is_speaking():
            self._queue_active = True
            self._first_item_of_burst = text
            if self.on_speech_start:
                # fire asynchronously
                self._schedule_after(0.0, self.on_speech_start, text)

        self._ensure_worker()
        self._queue.put(text)

    def speak(self, text: str, blocking: bool = False, interrupt_current: bool = True) -> bool:
        """
        Convenience around enqueue(). If blocking=True, waits for the queue to become idle again.
        """
        if interrupt_current and self.is_speaking():
            self.interrupt_current()
            time.sleep(0.05)

        self.enqueue(text)

        if not blocking:
            return True

        # Wait for queue to drain (burst ends)
        return self.wait_until_idle(timeout=3600.0)

    def wait_until_idle(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until no audio is playing and the queue is empty (i.e., burst finished).
        """
        start = time.time()
        while True:
            if self._queue.empty() and not self.is_speaking() and not self._queue_active:
                return True
            if timeout is not None and (time.time() - start) > timeout:
                return False
            time.sleep(0.05)

    def is_speaking(self) -> bool:
        with self._speech_lock:
            return self._is_speaking

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def interrupt_current(self):
        """
        Interrupt the current utterance (does not clear the queue).
        If this was the last thing and we return to idle, on_speech_interrupted fires.
        """
        if self.is_speaking():
            print("Interrupting current utterance...")
            self._interrupt_flag.set()

    def flush(self):
        """
        Stop the current utterance and clear the queue.
        This ends the burst due to interruption → on_speech_interrupted.
        """
        self._interrupt_flag.set()
        emptied_any = False
        try:
            while True:
                self._queue.get_nowait()
                self._queue.task_done()
                emptied_any = True
        except queue.Empty:
            pass
        # If we were mid-burst and there's nothing left (and current will stop), mark interrupted.
        self._maybe_finish_burst(interrupted=True, force=True)

    def shutdown(self):
        self.flush()
        self._stop_event.set()
        self._queue.put(None)
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
        self.stop_stream()

    # ---------- internals ----------

    def _schedule_after(self, delay_s: float, fn: Callable, *args):
        t = threading.Timer(max(0.0, float(delay_s)), fn, args=args)
        t.daemon = True
        t.start()

    def _hardware_tail_seconds(self) -> float:
        try:
            lat = getattr(self.stream, "latency", 0.0)
            if isinstance(lat, (tuple, list)):
                return float(lat[-1]) if len(lat) > 0 else 0.0
            return float(lat) if lat is not None else 0.0
        except Exception:
            return 0.0

    def _worker_loop(self):
        while not self._stop_event.is_set():
            item = self._queue.get()
            try:
                if item is None:
                    break
                self._speak_one(item)
            finally:
                self._queue.task_done()
                # After each item, see if we just became idle.
                self._maybe_finish_burst(interrupted=False)

    def _speak_one(self, text: str):
        if self.stream is None:
            return

        self._interrupt_flag.clear()
        with self._speech_lock:
            self._is_speaking = True
            self._current_text = text

        interrupted = False
        try:
            for chunk in self.voice.synthesize(text):
                # Pause handling
                while self._paused.is_set() and not self._interrupt_flag.is_set():
                    time.sleep(0.02)

                if self._interrupt_flag.is_set():
                    interrupted = True
                    break

                if self.stream is None:
                    interrupted = True
                    break

                try:
                    self.stream.write(chunk.audio_int16_array)
                except Exception as e:
                    print(f"Error writing to audio stream: {e}")
                    interrupted = True
                    break

        except Exception as e:
            print(f"Error during speech synthesis: {e}")
            interrupted = True
        finally:
            with self._speech_lock:
                self._is_speaking = False
                self._current_text = ""

            # If that was the last thing, finish burst appropriately.
            self._maybe_finish_burst(interrupted=interrupted)

    def _maybe_finish_burst(self, interrupted: bool, force: bool = False):
        """
        If we're in a burst and have no more work (queue empty & not speaking),
        mark the burst finished and fire the right callback exactly once.
        - interrupted=True  -> on_speech_interrupted
        - interrupted=False -> on_speech_end (after hardware tail)
        """
        # Small defer lets any just-enqueued text arrive.
        def _check_and_fire():
            if self._queue_active and self._queue.empty() and not self.is_speaking():
                self._queue_active = False
                # choose callback based on reason
                if interrupted:
                    if self.on_speech_interrupted:
                        self._schedule_after(0.0, self.on_speech_interrupted)
                else:
                    if self.on_speech_end:
                        tail = self._hardware_tail_seconds()
                        self._schedule_after(tail, self.on_speech_end)

        if force:
            _check_and_fire()
        else:
            # Defer slightly to avoid races with new enqueues right at utterance boundaries.
            self._schedule_after(0.01, _check_and_fire)

    # ---------- context / GC ----------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


# Example usage
if __name__ == "__main__":
    def started(t): print(f"[CB] burst start: first text={t!r}")
    def ended(): print("[CB] burst end (queue drained, hardware drained)")
    def interrupted(): print("[CB] burst interrupted")

    rv = RobotVoice(on_speech_start=started, on_speech_end=ended, on_speech_interrupted=interrupted)

    # Start a burst by streaming partial strings
    rv.enqueue("Hello! ")
    rv.enqueue("Streaming while more text arrives. ")
    time.sleep(0.2)
    rv.enqueue("I'll keep going until the queue is empty.")
    rv.wait_until_idle(timeout=10.0)  # triggers on_speech_end once

    # Demonstrate interruption ending a burst
    rv.enqueue("This will get cut… ")
    time.sleep(0.3)
    rv.interrupt_current()            # if nothing else queued, burst ends → on_speech_interrupted
    rv.wait_until_idle(timeout=5.0)

    # Demonstrate flush
    rv.enqueue("You won't hear me.")
    rv.flush()                        # burst ends → on_speech_interrupted
    rv.shutdown()
