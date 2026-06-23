from __future__ import annotations

import base64
from functools import lru_cache
from typing import Any, Dict, List

import cv2
import numpy as np

from ssm.config import load_yaml
from ssm.paths import get_paths


@lru_cache
def _load_prompt(name: str) -> dict:
    path = get_paths().configs_vlm / "prompts" / f"{name}.yaml"
    if not path.exists():
        path = get_paths().configs_vlm / "prompts" / "default.yaml"
    return load_yaml(path)


def build_vlm_system_prompt(*, robot_toy_mode: bool = False) -> str:
    key = "robot_toy" if robot_toy_mode else "default"
    data = _load_prompt(key)
    return (data.get("system") or "").strip()


def build_vlm_rules(*, robot_toy_mode: bool = False) -> str:
    key = "robot_toy" if robot_toy_mode else "default"
    data = _load_prompt(key)
    return (data.get("rules") or "").strip()


def should_stop_vlm_repeat(text: str, *, min_unit: int = 10, repeats: int = 2) -> bool:
    """流式生成出现同一段文字反复拼接时提前截断。"""
    n = len(text)
    if n < min_unit * (repeats + 1):
        return False
    max_unit = min(96, n // (repeats + 1))
    for unit_len in range(min_unit, max_unit + 1):
        unit = text[-unit_len:]
        if text.endswith(unit * (repeats + 1)):
            return True
        tail = text[-unit_len * (repeats + 2) :]
        if tail and tail.count(unit) > repeats:
            return True
    return False


def _delta_content_piece(chunk: Dict[str, Any]) -> str:
    """OpenAI 流式 chunk 里 choices[0].delta 的文本增量；无则返回空串。"""
    try:
        choices = chunk.get("choices")
        if not choices:
            return ""
        ch0: Any = choices[0]
        if not isinstance(ch0, dict):
            return ""
        delta = ch0.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(str(p.get("text", "")))
                    elif isinstance(p, dict) and "text" in p:
                        parts.append(str(p.get("text", "")))
                return "".join(parts)
        text = ch0.get("text")
        if isinstance(text, str):
            return text
    except (IndexError, TypeError, AttributeError):
        pass
    return ""


def _assistant_content_message(raw: Dict[str, Any]) -> str:
    """OpenAI 风格 API 中 assistant 的 content 可能是 str 或 content-parts 列表。"""
    try:
        msg = raw["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
        return "".join(parts)
    if content is not None:
        return str(content)
    return ""


def _resize_bgr_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    return cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _bgr_to_data_url(img: np.ndarray, image_format: str, jpeg_quality: int) -> str:
    fmt = image_format.lower().strip()
    if fmt == "png":
        ok, buf = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        if not ok:
            raise RuntimeError("cv2.imencode png failed")
        mime = "image/png"
    elif fmt in ("jpg", "jpeg"):
        ok, buf = cv2.imencode(
            ".jpg",
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("cv2.imencode jpg failed")
        mime = "image/jpeg"
    else:
        raise ValueError("unsupported image_format: %s" % image_format)
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return "data:%s;base64,%s" % (mime, b64)


def _image_data_url_from_jpeg_bytes(
    data: bytes,
    max_side: int,
    jpeg_quality: int,
    image_format: str = "png",
) -> str:
    """解码相机快照等压缩图后，按最长边可选缩放，再编码为 PNG 或 JPEG。"""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("无法解码图像字节（快照应为 JPEG/PNG 等）")
    img = _resize_bgr_max_side(img, max_side)
    return _bgr_to_data_url(img, image_format, jpeg_quality)
