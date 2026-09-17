"""Wrappers around MyGOChat so the Discord event loop never blocks on torch.

MyGOChat (https://github.com/qaz45647/MyGOChat) is a RoBERTa classifier that
maps a line of Traditional Chinese text onto one of ~157 MyGO!!!!! screenshots.
It is a synchronous, CPU/GPU-bound library with a multi-second first call, so
every call here is dispatched to a single-worker thread pool: one worker keeps
inference serialized (torch modules are not safe to share across threads) while
still freeing the event loop to keep the gateway heartbeat alive.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

log = logging.getLogger(__name__)

# Used only by RandomEngine when the vendored checkout is missing entirely.
_BUILTIN_FALLBACK: tuple[tuple[str, str], ...] = (
    ("不可能吧", "https://truth.bahamut.com.tw/s01/202502/d2f7862abe1e286262eda2f355c4dbbc.JPG"),
    (
        "一旦加入就無法回頭了喔",
        "https://truth.bahamut.com.tw/s01/202502/eb090662200d3fca876f69450bb2aa5d.JPG",
    ),
)


@dataclass(frozen=True)
class Candidate:
    """One ranked suggestion returned by the model."""

    label: str
    quote: str
    image_url: str
    probability: float  # percent, 0-100

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Candidate":
        label = str(payload.get("label") or payload.get("quote") or "")
        return cls(
            label=label,
            quote=str(payload.get("quote") or label),
            image_url=str(payload.get("image_url") or ""),
            probability=float(payload.get("probability") or 0.0),
        )


class EngineError(RuntimeError):
    """Raised when the backing model cannot be loaded or queried."""


class Engine(Protocol):
    """The surface the cogs depend on."""

    name: str

    async def load(self) -> None: ...

    async def predict(self, text: str, k: int = 5) -> list[Candidate]: ...

    async def close(self) -> None: ...


class MyGOChatEngine:
    """Runs the real MyGOChat model in a dedicated worker thread."""

    name = "mygochat"

    def __init__(self, mygochat_path: Path, max_input_chars: int = 300) -> None:
        self._path = Path(mygochat_path)
        self._max_input_chars = max_input_chars
        self._chat: Any | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mygochat")
        self._load_lock = asyncio.Lock()

    async def load(self) -> None:
        """Import and warm the model. Safe to call more than once."""
        async with self._load_lock:
            if self._chat is not None:
                return
            loop = asyncio.get_running_loop()
            self._chat = await loop.run_in_executor(self._executor, self._load_blocking)
            log.info("MyGOChat model ready (%s)", self._path)

    def _load_blocking(self) -> Any:
        if not self._path.exists():
            raise EngineError(
                f"MyGOChat checkout not found at {self._path}. "
                "Run scripts/setup.sh, or set MYGOCHAT_PATH to an existing clone."
            )

        # MyGOChat ships no setup.py, so the clone is imported off sys.path
        # rather than installed into site-packages.
        path_str = str(self._path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

        try:
            from mygochat import MyGOChat  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise EngineError(
                f"Could not import mygochat from {self._path}: {exc}. "
                "The model needs torch and transformers: "
                "`pip install -r requirements-model.txt`. "
                "Set MYGO_ENGINE=random to run without them."
            ) from exc

        try:
            return MyGOChat()
        except Exception as exc:  # pragma: no cover - torch/transformers failure
            raise EngineError(
                f"MyGOChat failed to initialize: {exc}. The 1.3 GB model.safetensors is a "
                "git-lfs object -- if `git lfs pull` was skipped it is only a pointer file."
            ) from exc

    async def predict(self, text: str, k: int = 5) -> list[Candidate]:
        text = text.strip()[: self._max_input_chars]
        if not text:
            return []
        if self._chat is None:
            await self.load()

        loop = asyncio.get_running_loop()
        try:
            raw = await loop.run_in_executor(self._executor, self._predict_blocking, text, k)
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"MyGOChat inference failed: {exc}") from exc

        return [Candidate.from_payload(item) for item in raw]

    def _predict_blocking(self, text: str, k: int) -> list[dict[str, Any]]:
        assert self._chat is not None
        result = self._chat.chat_with_candidates(text, k=k)
        return list(result.get("candidates", []))

    async def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._chat = None


class RandomEngine:
    """Zero-dependency stand-in: picks a random line from Label_Path.json.

    Useful for wiring up permissions and command plumbing before pulling the
    1.3 GB model, and for tests. Set MYGO_ENGINE=random to select it.
    """

    name = "random"

    def __init__(
        self,
        mygochat_path: Path | None = None,
        seed: int | None = None,
        pairs: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        self._path = Path(mygochat_path) if mygochat_path else None
        self._random = random.Random(seed)
        self._pairs: list[tuple[str, str]] = list(pairs) if pairs else []

    async def load(self) -> None:
        if self._pairs:
            return
        self._pairs = self._load_pairs()
        log.info("RandomEngine ready with %d quotes", len(self._pairs))

    def _load_pairs(self) -> list[tuple[str, str]]:
        if self._path:
            label_file = self._path / "mygochat" / "data" / "Label_Path.json"
            try:
                data = json.loads(label_file.read_text(encoding="utf-8"))
                pairs = [
                    (str(row["title"]), str(row["Image_Path"]))
                    for row in data
                    if row.get("title") and row.get("Image_Path")
                ]
                if pairs:
                    return pairs
            except (OSError, ValueError, KeyError, TypeError) as exc:
                log.warning("Falling back to built-in quotes (%s): %s", label_file, exc)
        return list(_BUILTIN_FALLBACK)

    async def predict(self, text: str, k: int = 5) -> list[Candidate]:
        if not text.strip():
            return []
        if not self._pairs:
            await self.load()

        picks = self._random.sample(self._pairs, k=min(k, len(self._pairs)))
        # Descending, plausible-looking confidences so downstream thresholds work.
        return [
            Candidate(
                label=quote,
                quote=quote,
                image_url=url,
                probability=round(max(1.0, 90.0 / (index + 1)), 2),
            )
            for index, (quote, url) in enumerate(picks)
        ]

    async def close(self) -> None:
        return None


def build_engine(config: Any) -> Engine:
    """Pick an engine from `Config`."""
    if getattr(config, "engine", "mygochat") == "random":
        return RandomEngine(mygochat_path=config.mygochat_path)
    return MyGOChatEngine(
        mygochat_path=config.mygochat_path,
        max_input_chars=config.max_input_chars,
    )


__all__ = [
    "Candidate",
    "Engine",
    "EngineError",
    "MyGOChatEngine",
    "RandomEngine",
    "build_engine",
]
