import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "redis" not in sys.modules:
    stub = ModuleType("redis")

    class Redis:  # pragma: no cover - shim for test environment
        def __init__(self, *args, **kwargs) -> None:
            pass

        def close(self) -> None:
            pass

    def from_url(*args, **kwargs):  # pragma: no cover - shim for tests
        return Redis()

    stub.Redis = Redis
    stub.from_url = from_url
    sys.modules["redis"] = stub

if "cv2" not in sys.modules:
    cv2_stub = ModuleType("cv2")

    class VideoCapture:  # pragma: no cover - shim for tests
        def __init__(self, path: str) -> None:
            self.path = path
            self._opened = Path(path).exists()

        def isOpened(self) -> bool:
            return self._opened

        def get(self, prop: int) -> float:
            return 0.0

        def release(self) -> None:
            pass

    cv2_stub.VideoCapture = VideoCapture
    cv2_stub.CAP_PROP_FRAME_COUNT = 7
    cv2_stub.CAP_PROP_FPS = 5
    sys.modules["cv2"] = cv2_stub
