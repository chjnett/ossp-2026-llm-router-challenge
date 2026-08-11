# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""3층 캐시 — 유연성의 실체.

"많이 시험한다"의 병목은 반복 시간이다. 특징 추출과 트랙별 예측을 디스크에
붙잡아 두면, 게이트 임계·μ·사용률 같은 값을 바꾸는 실험이 초 단위로 떨어진다.

    특징            추출기 버전 + 입력 해시    [F] 바뀔 때만
    트랙별 예측      트랙 버전 + fit 자료 해시  해당 트랙 바뀔 때만
    할당·게이트      캐시 없음                 매번

키는 전부 내용에서 유도한다. 파일 경로나 실행 순서는 키에 들어가지 않는다.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "experiments" / "cache"


def digest(*parts: object) -> str:
    """인자들을 순서대로 이어붙여 고정 해시를 만든다."""

    h = hashlib.blake2b(digest_size=16)
    for part in parts:
        if isinstance(part, np.ndarray):
            h.update(np.ascontiguousarray(part).tobytes())
            h.update(str(part.dtype).encode())
            h.update(str(part.shape).encode())
        elif isinstance(part, (list, tuple)):
            for item in part:
                h.update(str(item).encode("utf-8"))
                h.update(b"\x1f")
        else:
            h.update(str(part).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def texts_digest(texts: Sequence[str]) -> str:
    """프롬프트 묶음의 내용 해시. 순서가 바뀌면 다른 키가 된다.

    캐시는 오프라인 도구이므로 순서 의존이 문제되지 않는다. 라우터 런타임의
    순서 불변성과는 별개다.
    """

    return digest("texts", len(texts), texts)


class ArrayCache:
    """numpy 배열 묶음을 내용 주소로 저장한다."""

    def __init__(self, root: Path | None = None, *, enabled: bool = True) -> None:
        self.root = Path(root or DEFAULT_ROOT)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def _path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / f"{key}.npz"

    def get_or_compute(
        self,
        namespace: str,
        key: str,
        compute: Callable[[], Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        path = self._path(namespace, key)
        if self.enabled and path.exists():
            with np.load(path, allow_pickle=False) as loaded:
                self.hits += 1
                return {name: loaded[name] for name in loaded.files}

        self.misses += 1
        value = compute()
        if self.enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            # 부분 파일이 유효한 캐시로 보이지 않도록 원자적으로 교체한다.
            # np.savez는 이름이 .npz로 끝나지 않으면 확장자를 덧붙이므로
            # 임시 파일 이름도 .npz로 끝나야 한다.
            tmp = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
            try:
                with tmp.open("wb") as handle:
                    np.savez(handle, **value)
                os.replace(tmp, path)
            finally:
                if tmp.exists():
                    tmp.unlink()
        return value

    def stats(self) -> str:
        total = self.hits + self.misses
        rate = self.hits / total if total else 0.0
        return f"캐시 적중 {self.hits}/{total} ({rate:.0%})"
