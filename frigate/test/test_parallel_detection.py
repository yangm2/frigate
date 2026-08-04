import re
import threading
import time
import unittest
from unittest.mock import patch

from frigate.const import REGEX_CAMERA_NAME
from frigate.object_detection.parallel import (
    ParallelRemoteObjectDetector,
    lane_names,
)


class FakeEventsPerSecond:
    def __init__(self) -> None:
        self.count = 0

    def update(self) -> None:
        self.count += 1

    def eps(self) -> float:
        return float(self.count)


class FakeRemoteObjectDetector:
    """Stands in for the real one, which needs shm segments and a ZMQ socket."""

    def __init__(self, name, labels, detection_queue, model_config, stop_event):
        self.name = name
        self.fps = FakeEventsPerSecond()
        self.in_flight = False

    def detect(self, tensor_input, threshold=0.4):
        # A real lane is not reentrant: one input buffer, one output buffer, one
        # socket. Catch any dispatch that hands the same lane to two threads.
        assert not self.in_flight, f"lane {self.name} used concurrently"
        self.in_flight = True
        try:
            time.sleep(tensor_input / 100)
            self.fps.update()
            return [(self.name, tensor_input)]
        finally:
            self.in_flight = False


def build(name="front", lanes=1):
    with patch(
        "frigate.object_detection.parallel.RemoteObjectDetector",
        FakeRemoteObjectDetector,
    ):
        return ParallelRemoteObjectDetector(name, {}, None, None, None, lanes=lanes)


class TestLaneNames(unittest.TestCase):
    def test_lane_zero_is_the_bare_camera_name(self):
        """A camera at the default lanes:1 must reuse the shm that already
        exists for it rather than allocating a lane-suffixed one."""
        self.assertEqual(lane_names("front", 1), ["front"])
        self.assertEqual(lane_names("front", 3)[0], "front")

    def test_lane_names_cannot_collide_with_a_camera(self):
        """Lane names double as shm segment names and ZMQ topics, so a lane
        must never be spellable as a camera name."""
        for lane in lane_names("front", 4)[1:]:
            self.assertIsNone(re.match(REGEX_CAMERA_NAME, lane))

    def test_lane_names_are_not_prefixes_of_each_other(self):
        """Results route by ZMQ SUB prefix match on 'object_detector/<lane>/'.
        The trailing slash is what keeps #1 from swallowing #10 -- assert the
        property the routing depends on."""
        topics = [f"object_detector/{lane}/" for lane in lane_names("front", 12)]
        for topic in topics:
            others = [t for t in topics if t != topic]
            self.assertFalse(any(other.startswith(topic) for other in others))


class TestParallelDispatch(unittest.TestCase):
    def test_results_come_back_in_region_order(self):
        """reduce_detections() must see what the serial loop would have built,
        so slow regions may not overtake fast ones."""
        detector = build(lanes=4)
        # Descending durations: without ordering, results would come back
        # reversed.
        durations = [4, 3, 2, 1]
        results = detector.map(lambda d: detector.detect(d), durations)
        self.assertEqual([r[0][1] for r in results], durations)

    def test_regions_run_concurrently(self):
        detector = build(lanes=4)
        started = time.monotonic()
        detector.map(lambda d: detector.detect(d), [3, 3, 3, 3])
        elapsed = time.monotonic() - started
        # Serial would be ~0.12s; four lanes should land near 0.03s.
        self.assertLess(elapsed, 0.09)

    def test_more_regions_than_lanes_still_holds_the_invariant(self):
        """FakeRemoteObjectDetector asserts single use, so this fails loudly if
        a lane is handed out twice."""
        detector = build(lanes=2)
        results = detector.map(lambda d: detector.detect(d), [1] * 10)
        self.assertEqual(len(results), 10)
        self.assertEqual({r[0][0] for r in results}, {"front", "front#1"})

    def test_single_lane_is_serial_and_uses_no_pool(self):
        detector = build(lanes=1)
        self.assertIsNone(detector._pool)
        results = detector.map(lambda d: detector.detect(d), [1, 2])
        self.assertEqual([r[0][0] for r in results], ["front", "front"])

    def test_detect_outside_map_falls_through_to_lane_zero(self):
        detector = build(lanes=3)
        self.assertEqual(detector.detect(1)[0][0], "front")

    def test_thread_binding_is_cleared_after_map(self):
        detector = build(lanes=2)
        detector.map(lambda d: detector.detect(d), [1, 1])
        # Worker threads survive the pool; a stale binding would silently pin a
        # later frame's region to a lane already handed to someone else.
        bindings = []

        def check():
            bindings.append(getattr(detector._bound, "detector", None))

        threads = [threading.Thread(target=check) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(bindings, [None] * 4)


class TestAggregateFps(unittest.TestCase):
    def test_detection_fps_sums_every_lane(self):
        """camera_metrics.detection_fps reads this. Reporting one lane would
        under-report the camera by up to a factor of lanes, with no error."""
        detector = build(lanes=4)
        detector.map(lambda d: detector.detect(d), [1] * 8)
        self.assertEqual(detector.fps.eps(), 8.0)
        self.assertLess(detector.detectors[0].fps.eps(), 8.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
