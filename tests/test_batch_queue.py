import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batch_queue import BatchQueue


class FakeSynth:
    voices = ["92", "6097"]
    max_batch = 4

    def __init__(self): self.batches = []

    def speak_batch(self, texts, **settings):
        self.batches.append(list(texts))
        if "bad" in texts: raise ValueError("bad request")
        return [(text, voice) for text, voice in zip(texts, settings["voices"])]


class QueueTests(unittest.TestCase):
    def test_independent_callers_form_batches_without_losing_order_or_settings(self):
        synth = FakeSynth()
        with BatchQueue(synth, max_wait_ms=20) as queue:
            futures = [queue.submit(str(i), voice=synth.voices[i % 2]) for i in range(11)]
            self.assertEqual([f.result(timeout=2) for f in futures], [(str(i), synth.voices[i % 2]) for i in range(11)])
        self.assertTrue(any(len(b) > 1 for b in synth.batches))
        self.assertEqual([t for batch in synth.batches for t in batch], list(map(str, range(11))))
        self.assertTrue(all(len(b) <= 4 for b in synth.batches))

    def test_close_flushes_requests_and_rejects_new_work(self):
        queue = BatchQueue(FakeSynth())
        future = queue.submit("hello")
        queue.close()
        self.assertEqual(future.result(), ("hello", "92"))
        with self.assertRaisesRegex(RuntimeError, "closed"): queue.submit("later")
        queue.close()

    def test_errors_reach_waiting_callers(self):
        with BatchQueue(FakeSynth()) as queue:
            future = queue.submit("bad")
            with self.assertRaisesRegex(ValueError, "bad request"): future.result(timeout=2)

    def test_invalid_settings_are_rejected_before_joining_other_requests(self):
        with BatchQueue(FakeSynth()) as queue:
            for speed in [0, float("nan"), 4]:
                with self.assertRaises(ValueError): queue.submit("hello", speed=speed)
            with self.assertRaises(ValueError): queue.submit("hello", voice="unknown")
            with self.assertRaises(ValueError): queue.submit("hello", seed=-1)


if __name__ == "__main__": unittest.main()
