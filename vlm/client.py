#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _resdata_to_objects(res_data: dict, shape: Tuple[int, int, int]) -> List[dict]:
    objects: List[dict] = []
    det_id = 0
    for _label in sorted(res_data.keys()):
        for item in res_data[_label]:
            one_label, one_box, _one_score = item
            bbox_xyxy = [
                int(one_box[0]),
                int(one_box[1]),
                int(one_box[2]),
                int(one_box[3]),
            ]
            objects.append(
                {
                    "id": det_id,
                    "label": str(one_label),
                    "bbox_xyxy": bbox_xyxy,
                }
            )
            det_id += 1
    return objects


def _build_cv_payload(objects: list) -> dict:
    return {"objects": objects}


# 写入「画面信息」摘要：检测标签英->中（不影响 JSON 里原始 label 字段）。
_VLM_SUMMARY_LABEL_ALIASES: Dict[str, str] = {
    "robot_toy": "机器人玩偶",
}


def _is_robot_toy_label(raw: str) -> bool:
    s = (raw or "").strip()
    if s == "机器人玩偶":
        return True
    return s.lower() == "robot_toy"


def _label_for_vlm_summary(label: str) -> str:
    s = (label or "").strip()
    if s in _VLM_SUMMARY_LABEL_ALIASES:
        return _VLM_SUMMARY_LABEL_ALIASES[s]
    low = s.lower()
    return _VLM_SUMMARY_LABEL_ALIASES.get(low, s)


def _normalize_detection_bucket_key(raw: str) -> str:
    """同类合并用键；机器人玩偶统一到 robot_toy。"""
    s = (raw or "").strip()
    if _is_robot_toy_label(s):
        return "robot_toy"
    return s


def _bbox_grid_spot(bbox: List[Any], w: int, h: int) -> str:
    """框中心相对整图九宫格，返回如「上左」；无尺寸则空串。"""
    if w <= 1 or h <= 1:
        return ""
    try:
        x1, y1, x2, y2 = [float(bbox[i]) for i in range(4)]
    except (TypeError, ValueError, IndexError):
        return ""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    col = "左" if cx / w < 0.33 else ("中" if cx / w < 0.66 else "右")
    row = "上" if cy / h < 0.33 else ("中" if cy / h < 0.66 else "下")
    return row + col


def _summarize_cv(cv_payload: Optional[Dict[str, Any]]) -> str:
    """中文画面信息：类别、数量与整体方位（不含外观条列）。"""
    if not cv_payload:
        return ""
    pl = cv_payload
    items = pl.get("detections") or pl.get("objects") or []
    if not items:
        return ""
    w = max(int(pl.get("image_width") or 0), 1)
    h = max(int(pl.get("image_height") or 0), 1)
    bucket: Dict[str, List[str]] = {}
    for det in items:
        lbl = str(det.get("label", "?"))
        key = _normalize_detection_bucket_key(lbl)
        bbox = det.get("bbox_xyxy") or [0, 0, 0, 0]
        bucket.setdefault(key, []).append(_bbox_grid_spot(bbox, w, h))

    parts: List[str] = []
    keys_ordered = sorted(bucket.keys(), key=lambda k: (0 if k == "robot_toy" else 1, k))
    for key in keys_ordered:
        spots = bucket[key]
        spots_clean = [s for s in spots if s]
        loc = ""
        if spots_clean:
            loc = "(%s)" % Counter(spots_clean).most_common(1)[0][0]
        cn = _label_for_vlm_summary(key)
        n = len(spots)
        seg = "%s×%d%s" % (cn, n, loc)
        parts.append(seg)
    return "；".join(parts) if parts else ""


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
        ok, buf = cv2.imencode(
            ".png",
            img,
            [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
        )
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
        raise ValueError(f"unsupported image_format: {image_format}")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _image_data_url_resized(
    image_path: str, max_side: int, jpeg_quality: int, image_format: str = "png"
) -> str:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = _resize_bgr_max_side(img, max_side)
    return _bgr_to_data_url(img, image_format, jpeg_quality)


def _image_data_url_raw(image_path: str) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(image_path)
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _image_data_url_from_jpeg_bytes(
    data: bytes,
    max_side: int,
    jpeg_quality: int,
    image_format: str = "png",
) -> str:
    """解码相机快照等压缩图后，按最长边可选缩放，再编码为 PNG（默认，避免二次 JPEG）或 JPEG。"""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("无法解码图像字节（快照应为 JPEG/PNG 等）")
    img = _resize_bgr_max_side(img, max_side)
    return _bgr_to_data_url(img, image_format, jpeg_quality)


DEFAULT_VLM_SYSTEM_PROMPT = """你根据用户所附图像直接作答，语气自然、一两段口语即可。"""

DEFAULT_VLM_RULES = """规则：
- 只写图像里看得见的内容；不存在、看不清的一律不写，禁止编造、禁止猜测。
- 禁止否定句，如「没有看到XX」「画面里没有XX」——没有的就不写。
- 严禁在回答中出现：机器人、卡通、玩偶、玩具，以及包含上述任一字词的任何短语。
- 仅当用户原话已含禁词时，才可复述用户用词。
- 一两段说完即停；禁止重复同一句，禁止按角落/方位逐块罗列。"""

ROBOT_TOY_VLM_SYSTEM_PROMPT = """你根据用户所附图像直接作答，语气自然口语，一两段即可。

下列「机器人玩偶」仅在画面里能清楚看到时才写入回答；看不见或不确定时，完全忽略对机器人玩偶的任何说明，只描述其他可见内容（内化要点后自然叙述，勿逐条念说明书；不提资料、说明、训练、检测等来源）：

这是一款Q版卡通机器人玩偶，整体大头小身、造型圆润，白色云朵状头部搭配浅蓝色大护目镜，黑色镜框，镜片内两只竖椭圆黑眼与弧形微笑。身体为红白赛车服，胸口红衣、橙黄V领与圆环徽章，白腹印黑色「01」与橙色横条，红袖白掌、白腿黑圆头靴，具未来感与潮玩风格。"""

ROBOT_TOY_VLM_RULES = """规则：
- 只写图像里看得见的内容；禁止编造，禁止「没有/没看到/未发现」等否定句。
- 机器人玩偶全画面最多一个；只有符合上述造型的才算，椅子、网布、家具等不算。
- 画面里没有玩偶时，只答环境等实际可见物，不得提玩偶及相关词。
- 一两段说完即停；禁止重复同一句，禁止按角落/方位逐块罗列。"""


def build_vlm_system_prompt(*, robot_toy_mode: bool = False) -> str:
    return ROBOT_TOY_VLM_SYSTEM_PROMPT if robot_toy_mode else DEFAULT_VLM_SYSTEM_PROMPT


def build_vlm_rules(*, robot_toy_mode: bool = False) -> str:
    return ROBOT_TOY_VLM_RULES if robot_toy_mode else DEFAULT_VLM_RULES


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


def _http_get_bytes(url: str, timeout: float) -> bytes:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"摄像头服务不可用: {url} ({e})") from e


def _http_get_json(url: str, timeout: float) -> Dict[str, Any]:
    data = _http_get_bytes(url, timeout)
    try:
        obj = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"摄像头服务返回的 JSON 无法解析: {url}") from e
    if not isinstance(obj, dict):
        raise RuntimeError(f"摄像头服务返回的不是 JSON 对象: {url}")
    return obj


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


def _post_chat_completions(base_url: str, body: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/v1/chat/completions"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {err_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Request failed (is llama-server running?): {e}") from e


def _post_chat_completions_stream(base_url: str, body: Dict[str, Any], timeout: float) -> str:
    """POST chat/completions，stream=true，按 SSE 解析并边到边打印到 stdout；返回合并后的助手全文。"""
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = dict(body)
    payload["stream"] = True
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    acc: List[str] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            while True:
                raw_line = resp.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                err = obj.get("error")
                if err is not None:
                    raise RuntimeError(f"Stream error: {err}")
                piece = _delta_content_piece(obj)
                if piece:
                    acc.append(piece)
                    print(piece, end="", flush=True)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {err_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Request failed (is llama-server running?): {e}") from e

    full = "".join(acc)
    if full and not full.endswith("\n"):
        print("", flush=True)
    return full


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CV + user message -> VLM (llama-server OpenAI API)",
    )
    parser.add_argument("-i", "--image", type=str, default=None, help="输入图片路径")
    parser.add_argument("-m", "--message", type=str, required=True, help="用户文字（与网页里输入等同）")
    parser.add_argument(
        "--camera",
        action="store_true",
        help="使用已启动的 TensorRT 摄像头服务（默认 http://127.0.0.1:9080）",
    )
    parser.add_argument("--camera-url", type=str, default="http://127.0.0.1:9080")
    parser.add_argument(
        "--weight",
        type=str,
        default="runs/detect/ssg_a/run.bin",
        help="加密 ONNX run.bin 路径（默认 runs/detect/ssg_a/run.bin，一般不必传）",
    )
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--nms", type=float, default=0.5)
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="不写「画面信息」摘要；摄像头不加 GET /cv_result.json；-i 图不调 ssDet",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=0,
        help="发给 VLM 前最长边缩放，0=按解码/原图尺寸（由视觉塔侧再做 smart_resize）",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        choices=("png", "jpeg"),
        default="png",
        help="发给 VLM 的 data URL 编码：png 无二次 JPEG 损失（默认）；jpeg 体积更小",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="仅当 --image-format jpeg 时有效",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="system 提示词；不设则用内置默认",
    )
    parser.add_argument(
        "--system-file",
        type=str,
        default=None,
        help="从文件读 system 提示词（UTF-8），优先于 --system",
    )
    parser.add_argument(
        "--save-cv-json",
        type=str,
        default=None,
        help="可选：把 /cv_result.json 同结构的 JSON 写到该路径",
    )
    parser.add_argument(
        "--save-full",
        type=str,
        default=None,
        help="可选：把 API 完整 JSON 响应写入该路径",
    )
    parser.add_argument(
        "--print-cv",
        action="store_true",
        help="stderr 额外打一行画面信息摘要（由检测 JSON 压成）",
    )
    parser.add_argument(
        "--verbose-cv",
        action="store_true",
        help="打印每框的 box/label（默认关闭，避免与模型回答混在一起）",
    )
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        help="流式输出（默认）：stream=true，按 SSE 即时打印",
    )
    stream_group.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="关闭流式：等完整响应后再打印",
    )
    parser.set_defaults(stream=True)
    args = parser.parse_args()

    if args.system_file:
        system_override = Path(args.system_file).read_text(encoding="utf-8").strip()
    elif args.system:
        system_override = args.system
    else:
        system_override = None

    cv_payload: Optional[Dict[str, Any]] = None

    if args.camera:
        base = args.camera_url.rstrip("/")
        try:
            snapshot = _http_get_bytes(base + "/snapshot.jpg", args.timeout)
            if not args.no_cv:
                cv_payload = _http_get_json(base + "/cv_result.json", args.timeout)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            print(
                '请先启动摄像头推理服务：cd "/home/sunshink/ssdet VLM" && "camera/scripts/start_camera.sh"',
                file=sys.stderr,
            )
            return 1

        out_vis = Path("cv.jpg")
        out_vis.write_bytes(snapshot)
        print(f"[cv] camera snapshot -> {out_vis.resolve()}", file=sys.stderr)
        image_url = _image_data_url_from_jpeg_bytes(
            snapshot,
            args.max_image_side,
            args.jpeg_quality,
            args.image_format,
        )
    else:
        if not args.image:
            print("请传入 -i/--image，或使用 --camera", file=sys.stderr)
            return 1

        src = cv2.imread(args.image)
        if src is None:
            print(f"无法读图: {args.image}", file=sys.stderr)
            return 1

        if not args.no_cv:
            from cv.inference import ssDet

            model = ssDet(
                conf=args.conf,
                nms=args.nms,
                weight=args.weight,
                verbose=args.verbose_cv,
            )
            res_data, vis = model.detect(src)
            out_vis = Path("cv.jpg")
            if not cv2.imwrite(str(out_vis), vis):
                print(f"[cv] failed to write {out_vis}", file=sys.stderr)
                return 1
            print(f"[cv] wrote {out_vis.resolve()}", file=sys.stderr)
            del vis
            objects = _resdata_to_objects(res_data, src.shape)
            cv_payload = _build_cv_payload(objects)
            cv_payload["image_width"] = int(src.shape[1])
            cv_payload["image_height"] = int(src.shape[0])

        if args.max_image_side <= 0:
            image_url = _image_data_url_raw(args.image)
        else:
            image_url = _image_data_url_resized(
                args.image, args.max_image_side, args.jpeg_quality, args.image_format
            )

    cv_summary = _summarize_cv(cv_payload)

    if args.save_cv_json and cv_payload is not None:
        Path(args.save_cv_json).write_text(
            json.dumps(cv_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[cv] wrote {args.save_cv_json}", file=sys.stderr)

    if args.print_cv:
        print("[cv_summary]", cv_summary or "(empty)", file=sys.stderr)

    msg = (args.message or "").strip()
    user_text = msg if msg else args.message
    if cv_summary:
        user_text = f"{user_text}\n\n画面信息：{cv_summary}"

    if system_override is not None:
        system = system_override
    else:
        # CLI 默认不带 system 提示词；网页视频对话由 camera/server 的 /ask 注入。
        system = ""

    messages: List[Dict[str, Any]] = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": user_text},
            ],
        }
    )

    req_body = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "cache_prompt": False,
        "n_cache_reuse": 0,
    }

    print("[vlm] calling llama-server …", file=sys.stderr)
    try:
        if args.stream:
            content = _post_chat_completions_stream(
                args.base_url, req_body, args.timeout
            )
            raw: Optional[Dict[str, Any]] = None
        else:
            raw = _post_chat_completions(args.base_url, req_body, args.timeout)
            content = _assistant_content_message(raw)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    if args.save_full:
        if args.stream:
            save_obj: Dict[str, Any] = {
                "stream": True,
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": content,
                        }
                    }
                ],
            }
        else:
            save_obj = raw
        Path(args.save_full).write_text(
            json.dumps(save_obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[vlm] full response -> {args.save_full}", file=sys.stderr)

    if not content:
        print("未能解析助手回复（choices[0].message.content 为空或非预期类型）", file=sys.stderr)
        if raw is not None:
            print(json.dumps(raw, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    if not args.stream:
        print(content, end="" if content.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
