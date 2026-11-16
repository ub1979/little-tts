"""Bounded microbatch queue: combine independent callers into real ONNX batches."""
from concurrent.futures import Future
import math
from queue import Queue, Empty, Full
import threading
import time


class BatchQueue:
    def __init__(self, synthesizer, max_wait_ms=2, max_pending=256):
        if max_wait_ms < 0 or max_pending < 1:
            raise ValueError("Queue settings must be nonnegative with positive capacity")
        self.synthesizer = synthesizer
        self.max_wait = max_wait_ms / 1000
        self.queue = Queue(maxsize=max_pending)
        self.closed = False
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self._work, name="little-tts-batcher", daemon=True)
        self.worker.start()

    def submit(self, text, voice="92", speed=1.0, seed=0):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Each request needs nonempty text")
        if voice not in self.synthesizer.voices:
            raise ValueError("Unknown voice")
        if not isinstance(speed, (float, int)) or not math.isfinite(speed) or not .5 <= speed <= 2:
            raise ValueError("Speed must be between .5 and 2")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        future = Future()
        with self.lock:
            if self.closed:
                raise RuntimeError("Batch queue is closed")
            try:
                self.queue.put_nowait((future, text, voice, speed, seed))
            except Full:
                raise RuntimeError("Batch queue is full; retry later") from None
        return future

    def _work(self):
        stopping = False
        while not stopping:
            first = self.queue.get()
            if first is None:
                break
            pending = [first]
            deadline = time.monotonic() + self.max_wait
            while len(pending) < self.synthesizer.max_batch:
                remaining = deadline - time.monotonic()
                try:
                    item = self.queue.get(timeout=max(0, remaining))
                except Empty:
                    break
                if item is None:
                    stopping = True
                    break
                pending.append(item)
            active = [r for r in pending if r[0].set_running_or_notify_cancel()]
            if not active:
                continue
            try:
                results = self.synthesizer.speak_batch([r[1] for r in active], voices=[r[2] for r in active],
                                                       speeds=[r[3] for r in active], seeds=[r[4] for r in active])
                if len(results) != len(active):
                    raise RuntimeError("Inference returned an unexpected result count")
            except Exception as error:
                for request in active:
                    request[0].set_exception(error)
            else:
                for request, result in zip(active, results):
                    request[0].set_result(result)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        self.queue.put(None)
        self.worker.join()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
