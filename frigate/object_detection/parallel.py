"""Fan one camera's regions out across several detector lanes.

`RemoteObjectDetector` is single-lane by construction: one input shared memory
segment named after the camera, one `out-<name>` segment for results, one ZMQ
SUB socket, and a `detect()` that blocks until its own result comes back. A
camera therefore pays `regions * inference` serially on every frame even when
the detector pool is idle, which caps it at `(1000 / detect.fps) / inference_ms`
regions per frame no matter how many detectors are configured.

`ParallelRemoteObjectDetector` holds `detect.lanes` independent detectors --
each with its own name, its own pair of shm segments and its own socket -- and
runs a frame's regions across them on a thread pool. Every blocking step
releases the GIL (`zmq.select` in the subscriber, the numpy copy into shm, the
cv2 work in `create_tensor_input`), so the threads overlap for real.

Nothing on the detector side changes. `DetectorRunner` already creates
`out-<connection_id>` lazily for ids it has not seen, and the detection queue is
already shared across every camera and every detector.

Two things this deliberately does not do:

- It does not add detector capacity. It lets one camera claim more of a pool
  that is otherwise idle, so under saturation it trades fairness for latency:
  upstream's one-outstanding-request-per-camera is accidental fairness that
  lanes give up. Keep `lanes` at 1 for cameras that do not need it.
- It does not free the shm segments it uses. Lane segments are created and
  tracked alongside the camera's own in `app.py` / `camera/maintainer.py`, which
  is where they are released.
"""

import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue
from multiprocessing.synchronize import Event as MpEvent
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from frigate.detectors.detector_config import ModelConfig
from frigate.object_detection.base import RemoteObjectDetector

logger = logging.getLogger(__name__)

# "#" cannot appear in a camera name (frigate.const.REGEX_CAMERA_NAME is
# ^[a-zA-Z0-9_-]+$), so a lane name can never collide with a real camera's shm
# segment. It is also safe against the ZMQ SUB prefix match used to route
# results, because subscriptions carry a trailing "/": "object_detector/cam/"
# is not a prefix of "object_detector/cam#1/", and "...#1/" is not a prefix of
# "...#10/". Changing the separator or dropping the slash would silently
# cross-deliver detections between lanes.
LANE_SEPARATOR = "#"


def lane_names(camera_name: str, lanes: int) -> list[str]:
    """Shm and topic names for a camera's detector lanes.

    Lane 0 keeps the bare camera name, so a camera at the default `lanes: 1` is
    byte for byte what upstream does and reuses the shm that already exists for
    it rather than orphaning it.
    """
    return [camera_name] + [
        f"{camera_name}{LANE_SEPARATOR}{i}" for i in range(1, lanes)
    ]


class AggregateEventsPerSecond:
    """`eps()` summed across lanes.

    `camera_metrics.detection_fps` is read off the detector's `fps` attribute.
    Each lane counts only its own inferences, so reporting a single lane would
    silently under-report the camera by up to a factor of `lanes`.
    """

    def __init__(self, detectors: list[RemoteObjectDetector]) -> None:
        self.detectors = detectors

    def eps(self) -> float:
        return sum(detector.fps.eps() for detector in self.detectors)


class ParallelRemoteObjectDetector:
    def __init__(
        self,
        name: str,
        labels: dict[int, str],
        detection_queue: Queue,
        model_config: ModelConfig,
        stop_event: MpEvent,
        lanes: int = 1,
    ) -> None:
        self.name = name
        self.detectors = [
            RemoteObjectDetector(
                lane_name, labels, detection_queue, model_config, stop_event
            )
            for lane_name in lane_names(name, lanes)
        ]
        self.fps = AggregateEventsPerSecond(self.detectors)

        # A lane is handed to exactly one thread at a time. The pool is sized to
        # the lane count, so this never actually blocks; it is what makes the
        # invariant explicit rather than implied by the pool size.
        self._idle: queue.Queue[RemoteObjectDetector] = queue.Queue()
        for detector in self.detectors:
            self._idle.put(detector)

        self._bound = threading.local()
        self._pool = (
            ThreadPoolExecutor(
                max_workers=len(self.detectors),
                thread_name_prefix=f"detect:{name}",
            )
            if len(self.detectors) > 1
            else None
        )

        if len(self.detectors) > 1:
            logger.info(f"{name}: detecting across {len(self.detectors)} lanes")

    def detect(self, tensor_input: np.ndarray, threshold: float = 0.4) -> list[Any]:
        """Detect on the lane bound to the calling thread.

        This is what keeps `video.detect()` unchanged: it still gets one object
        with a `detect()` method and never learns that lanes exist. Calls from
        outside `map()` (single-lane cameras, or any future caller) fall through
        to lane 0.
        """
        detector = getattr(self._bound, "detector", None) or self.detectors[0]
        return detector.detect(tensor_input, threshold)

    def map(self, fn: Callable[[Any], Any], items: Iterable[Any]) -> list[Any]:
        """Apply `fn` across lanes, returning results in input order.

        Order is preserved so the detection list handed to `reduce_detections`
        is identical to the serial one.
        """
        items = list(items)

        if self._pool is None or len(items) < 2:
            return [fn(item) for item in items]

        return list(self._pool.map(self._on_a_lane(fn), items))

    def _on_a_lane(self, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def run(item: Any) -> Any:
            detector = self._idle.get()
            self._bound.detector = detector
            try:
                return fn(item)
            finally:
                self._bound.detector = None
                self._idle.put(detector)

        return run
