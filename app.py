#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
════════════════════════════════════════════════════════════════════════════
  مِرآة — استوديو الصور المتقدم
  Mirror — Advanced Image Studio (Backend)
════════════════════════════════════════════════════════════════════════════
  المشروع: محرر صور يحاكي المحررات الشهيرة.
  التقنية : باك-اند Python (بدون إطارات معقدة - مكتبة قياسية فقط + Pillow + numpy)
  الواجهة : ملف واحد index.html

  التشغيل في Visual Studio Code (بسيط جداً):
     1)  افتح مجلد المشروع في VS Code.
     2)  افتح الطرفية (Terminal).
     3)  نفّذ الأمر:          python app.py
     4)  افتح المتصفح على:     http://localhost:8000

  المتطلبات (تُثبَّت مرة واحدة فقط):
     pip install pillow numpy

  ─────────────────────────────────────────────────────────────────────────
  تقسيم المشروع (كما طلبت):
     • جهة «المعالجة والتحسين والتنعيم» ⟵ بكسلات + فلتر تنعيم/تحديد/حواف.
     • جهة «العمليات الأخرى»            ⟵ هندسة، تحويل صيغ، إزالة خلفية، رسم.
     • جهة «الميزات الإبداعية»           ⟵ دمج، خلفيات منتجات، مؤثرات فنية.
  ─────────────────────────────────────────────────────────────────────────
  الخوارزميات الواردة في المحاضرات المرفقة:
      Lab1: قراءة/حفظ، معلومات، تحويل صيغ.                  ✔
      Lab2: قص، تكبير/تصغير، تدوير، نقل، دمج، خلفية منتج.    ✔
      Lab3: Negative، Logarithm، Equalization (Histogram).   ✔
      Lab4: Mean & Gaussian (Linear)، Median (Non-Linear)،
            Custom Kernel.                                   ✔
      Lab5: Laplacian، Difference، Laplacian Edge، Sobel،
            Canny.                                           ✔
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8000"))
ROOT = Path(__file__).resolve().parent
INDEX_FILE = ROOT / "index.html"
BACKGROUND_DIR = ROOT / "backgrounds"
GUIDE_PDF = ROOT / "دليل_الاستخدام_الشامل_مرآة.pdf"
GUIDE_DOCX = ROOT / "دليل_الاستخدام_الشامل_مرآة.docx"
# أسماء بديلة لدعم التحميل من الموقع
GUIDE_ALIASES = {
    "/guide.pdf": GUIDE_PDF,
    "/دليل_الاستخدام_الشامل_مرآة.pdf": GUIDE_PDF,
    "/دليل_الاستخدام.pdf": GUIDE_PDF,
    "/دليل-الاستخدام.pdf": GUIDE_PDF,
    "/help.pdf": GUIDE_PDF,
    "/guide.docx": GUIDE_DOCX,
    "/دليل_الاستخدام_الشامل_مرآة.docx": GUIDE_DOCX,
    "/دليل_الاستخدام.docx": GUIDE_DOCX,
}
MAX_BODY = 64 * 1024 * 1024


# ═══════════════════════════════════════════════════════════════════════════
#  أدوات مساعدة عامة
# ═══════════════════════════════════════════════════════════════════════════

def clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        return max(low, min(high, int(float(value))))
    except (TypeError, ValueError):
        return default


def clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def decode_data_url(value: str) -> Image.Image:
    """تحويل data URL إلى صورة RGBA مع تصحيح اتجاه EXIF."""
    if not isinstance(value, str) or "," not in value:
        raise ValueError("لم تصل صورة صالحة")
    header, encoded = value.split(",", 1)
    if not encoded:
        raise ValueError("بيانات الصورة فارغة")
    try:
        raw = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(raw))
        image = ImageOps.exif_transpose(image)
        return image.convert("RGBA")
    except Exception as exc:
        raise ValueError("تعذر قراءة الصورة؛ تأكد من أنها PNG أو JPG أو WEBP") from exc


def _flatten_for_jpeg(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGB")
    bg = Image.new("RGB", image.size, (255, 255, 255))
    bg.paste(image, mask=image.getchannel("A"))
    return bg


def encode_data_url(image: Image.Image, output_format: str = "PNG", quality: int = 92) -> str:
    fmt = (output_format or "PNG").upper().replace("JPG", "JPEG")
    if fmt not in {"PNG", "JPEG", "WEBP", "BMP", "GIF"}:
        fmt = "PNG"
    buffer = io.BytesIO()
    kwargs: dict[str, Any] = {}
    if fmt == "JPEG":
        image = _flatten_for_jpeg(image)
        kwargs.update(quality=max(10, min(100, int(quality))), optimize=True)
    elif fmt == "WEBP":
        kwargs.update(quality=max(10, min(100, int(quality))), method=5)
    elif fmt == "PNG":
        kwargs.update(optimize=True)
    if fmt == "GIF":
        image = image.convert("P")
    image.save(buffer, format=fmt, **kwargs)
    mime = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp", "BMP": "bmp", "GIF": "gif"}[fmt]
    return "data:image/%s;base64," % mime + base64.b64encode(buffer.getvalue()).decode("ascii")


def np_from(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGBA")).astype(np.float32)


def to_image(arr: np.ndarray, preserve_alpha: np.ndarray | None = None) -> Image.Image:
    arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    if preserve_alpha is not None and arr.shape[2] == 3:
        rgba = np.dstack([arr, np.clip(np.round(preserve_alpha), 0, 255).astype(np.uint8)])
        return Image.fromarray(rgba, "RGBA")
    if arr.shape[2] == 4:
        return Image.fromarray(arr, "RGBA")
    return Image.fromarray(arr, "RGB").convert("RGBA")


def grayscale_array(image: Image.Image) -> np.ndarray:
    return np.asarray(ImageOps.grayscale(image)).astype(np.float32)


def put_alpha(base: Image.Image, alpha: Image.Image | None = None) -> Image.Image:
    """إرجاع صورة RGB مع الحفاظ على قناة الشفافية الأصلية."""
    if alpha is None:
        alpha = base.getchannel("A")
    rgb = base.convert("RGB")
    out = Image.new("RGBA", rgb.size)
    out.paste(rgb, mask=alpha)
    return out


def convolve_channel(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """التفاف ثنائي الأبعاد متجهي (سريع) لمصفوفة رمادية عبر النواة."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(channel, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(channel, dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + channel.shape[0], j:j + channel.shape[1]]
    return out


def normalize_edges(magnitude: np.ndarray) -> np.ndarray:
    """تطبيع خريطة حواف إلى نطاق 0..255."""
    m = magnitude
    low, high = float(m.min()), float(m.max())
    if high - low < 1e-6:
        return np.zeros_like(m)
    return (m - low) / (high - low) * 255.0


# ═══════════════════════════════════════════════════════════════════════════
#  تحسينات البكسل (تحويلات نقطية)  —  Lab3
# ═══════════════════════════════════════════════════════════════════════════

def apply_gamma(image: Image.Image, gamma: float) -> Image.Image:
    gamma = clamp_float(gamma, 0.05, 5.0, 1.0)
    lut = [max(0, min(255, int(((i / 255.0) ** gamma) * 255.0 + 0.5))) for i in range(256)]
    rgb = image.convert("RGB").point(lut * 3)
    return put_alpha(rgb, image.getchannel("A"))


def apply_logarithm(image: Image.Image, c: float | None = None) -> Image.Image:
    """تحويل اللوغاريتم: s = c * log(1 + r)  (يسلّط الضوء على التفاصيل الداكنة)."""
    r = grayscale_array(image)
    c = c if c is not None else 255.0 / (math.log(1.0 + r.max()) if r.max() > 0 else 1.0)
    out = np.clip(c * np.log1p(np.maximum(r, 0.0)), 0, 255)
    return put_alpha(Image.fromarray(out.astype(np.uint8)), image.getchannel("A"))


def apply_negative(image: Image.Image) -> Image.Image:
    """Negative Transform: s = 255 - r"""
    rgb = 255.0 - np.asarray(image.convert("RGB")).astype(np.float32)
    return to_image(rgb, np.asarray(image.getchannel("A")).astype(np.float32))


def apply_equalize(image: Image.Image) -> Image.Image:
    """تسوية الهيستوغرام (Histogram Equalization) — تحسين التباين تلقائياً."""
    gray = ImageOps.equalize(ImageOps.grayscale(image))
    # لتطبيق التسوية بطريقة ملونة نحافظ على اللون من خلال تحويل LCh بسيط.
    rgb = image.convert("RGB")
    # تُطبَّق التسوية على التشبع-السطوع في نموذج HSV للحفاظ على الألوان.
    hsv = rgb.convert("HSV")
    h, s, v = hsv.split()
    v = ImageOps.equalize(v)
    out = Image.merge("HSV", (h, s, v)).convert("RGB")
    return put_alpha(out, image.getchannel("A"))


def apply_sepia(image: Image.Image, strength: float = 1.0) -> Image.Image:
    strength = clamp_float(strength, 0.0, 1.0, 1.0)
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    out = np.stack(
        [
            gray * 1.07,  # أحمر
            gray * 0.98,  # أخضر
            gray * 0.80,  # أزرق
        ],
        axis=-1,
    )
    arr = arr * (1 - strength) + out * strength
    return put_alpha(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), image.getchannel("A"))


def apply_hsv_adjust(image: Image.Image, hue: float = 0.0, saturation: float = 0.0, value: float = 0.0) -> Image.Image:
    """تعديل درجة اللون / التشبع / الإضاءة في نموذج HSV."""
    hsv = image.convert("HSV")
    h, s, v = hsv.split()
    h = h.point(lambda p: int((p + hue) % 256))
    s = s.point(lambda p: clamp_int(int(p * (1 + saturation / 100)), 0, 255, p))
    v = v.point(lambda p: clamp_int(int(p * (1 + value / 100)), 0, 255, p))
    out = Image.merge("HSV", (h, s, v)).convert("RGB")
    return put_alpha(out, image.getchannel("A"))


def apply_color_balance(image: Image.Image, red: float, green: float, blue: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    for idx, amount in ((0, red), (1, green), (2, blue)):
        arr[..., idx] *= (1 + amount / 100)
    return put_alpha(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), image.getchannel("A"))


def apply_vibrance(image: Image.Image, amount: float) -> Image.Image:
    """إحياء الألوان (Vibrance): يزيد التشبع بشكل قوي للألوان الباهتة فقط."""
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    gray = arr.mean(axis=-1, keepdims=True)
    sat = np.maximum(np.max(arr, axis=-1, keepdims=True) - np.min(arr, axis=-1, keepdims=True), 1e-6)
    # أوزان أقل للألوان المشبعة أصلاً حتى لا تحترق البشرة.
    weight = 1 - np.clip(sat / 255.0, 0, 1)
    amount = clamp_float(amount, -100, 100, 0) / 100
    out = gray + (arr - gray) * (1 + amount * weight)
    return put_alpha(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), image.getchannel("A"))


# ═══════════════════════════════════════════════════════════════════════════
#  الفلاتر (تنعيم / تحديد / حواف)  —  Lab4 & Lab5
# ═══════════════════════════════════════════════════════════════════════════

def apply_custom_filter(image: Image.Image, kernel: list[list[float]], normalize: bool = True) -> Image.Image:
    """فلتر مخصص بالنواة (إدخال المستخدم) — Custom Convolution Kernel.
    يجب أن تكون الأبعاد فردية. اختيارياً تُقسَّم على مجموعها حتى لا تُغيَّر الإضاءة."""
    k = np.array(kernel, dtype=np.float32)
    if k.ndim != 2 or k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise ValueError("النواة يجب أن تكون مصفوفة ثنائية الأبعاد بأبعاد فردية (مثل 3×3 أو 5×5)")
    if normalize:
        total = float(k.sum())
        if abs(total) > 1e-6:
            k = k / total
    gray = grayscale_array(image)
    out = convolve_channel(gray, k)
    out = np.clip(out, 0, 255)
    return put_alpha(Image.fromarray(out.astype(np.uint8)), image.getchannel("A"))


def apply_mean_filter(image: Image.Image, size: int) -> Image.Image:
    """Linear Mean Filter — متوسط نافذة size×size (تنعيم)."""
    size = clamp_int(size, 3, 41, 3)
    if size % 2 == 0:
        size += 1
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    alpha = np.asarray(image.getchannel("A")).astype(np.float32)
    out = np.zeros_like(arr)
    for c in range(3):
        out[..., c] = convolve_channel(arr[..., c], np.full((size, size), 1.0 / (size * size)))
    return to_image(out, alpha)


def apply_gaussian_filter(image: Image.Image, radius: float) -> Image.Image:
    """Linear Gaussian Filter — تجانسي عبر نافذة غاوسية (تنعيم ناعم)."""
    radius = clamp_float(radius, 0.1, 30.0, 2.0)
    rgb = image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))
    return put_alpha(rgb, image.getchannel("A"))


def apply_median_filter(image: Image.Image, size: int) -> Image.Image:
    """Non-Linear Median Filter — يستبدل كل بكسل بوسيط جيرانه (يزيل الضجيج)."""
    size = clamp_int(size, 3, 25, 3)
    if size % 2 == 0:
        size += 1
    rgb = image.convert("RGB").filter(ImageFilter.MedianFilter(size=size))
    return put_alpha(rgb, image.getchannel("A"))


def apply_laplacian(image: Image.Image, mode: str = "abs") -> Image.Image:
    """Laplacian Filter — مشتقة ثانية تبرز الحواف تماماً (وتؤدي لون رمادي بالوسط)."""
    gray = grayscale_array(image)
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    lap = convolve_channel(gray, kernel)
    if mode == "abs":
        lap = np.abs(lap)
    out = np.clip(lap, 0, 255)
    return put_alpha(Image.fromarray(out.astype(np.uint8)), image.getchannel("A"))


def apply_laplacian_edge(image: Image.Image) -> Image.Image:
    """Laplacian Edge Detection — إبراز الحواف عبر مشتقة ثانية (خلفية سوداء)."""
    gray = grayscale_array(image)
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    lap = convolve_channel(gray, kernel)
    edge = normalize_edges(np.abs(lap)) * 255.0
    # نحافظ فقط على الحواف القوية
    edge = np.where(edge > 12, 255, 0)
    return put_alpha(Image.fromarray(edge.astype(np.uint8)), image.getchannel("A"))


def apply_difference_filter(image: Image.Image) -> Image.Image:
    """Difference / High-Pass Filter — الصورة فرق الغاوسي (تفاصيل دقيقة)."""
    gray = grayscale_array(image)
    smoothed = np.asarray(image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=2))).astype(np.float32)
    g = 0.299 * smoothed[..., 0] + 0.587 * smoothed[..., 1] + 0.114 * smoothed[..., 2]
    diff = gray - g
    out = np.clip(diff + 128, 0, 255)
    return put_alpha(Image.fromarray(out.astype(np.uint8)), image.getchannel("A"))


def _sobel_kernels() -> tuple[np.ndarray, np.ndarray]:
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    return gx, gy


def apply_sobel(image: Image.Image, axis: str | None = None) -> Image.Image:
    """Sobel — كاشف حواف مشتقة أولى (X، Y، أو المقدار الإجمالي)."""
    gx, gy = _sobel_kernels()
    gray = grayscale_array(image)
    sx = convolve_channel(gray, gx)
    sy = convolve_channel(gray, gy)
    if axis == "x":
        out = normalize_edges(np.abs(sx))
    elif axis == "y":
        out = normalize_edges(np.abs(sy))
    else:
        # المقدار: sqrt(Gx² + Gy²) ثم تطبيع.
        magnitude = np.sqrt(sx ** 2 + sy ** 2)
        out = normalize_edges(magnitude)
    return put_alpha(Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8)), image.getchannel("A"))


def apply_canny(image: Image.Image, low: int = 60, high: int = 160) -> Image.Image:
    """Canny Edge Detection كامل من الصفر:
       1) غاوسي (تنعيم)  2) Sobel (تدرج)  3) Non-Maximum Suppression
       4) عتبتان (Hysteresis) لربط الحواف."""
    gray = grayscale_array(image)
    h, w = gray.shape
    # 1) تنعيم غاوسي
    radius = clamp_float(2.0, 0.5, 8.0, 2.0)
    smooth = np.asarray(Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius))).astype(np.float32)
    # 2) Sobel
    gx, gy = _sobel_kernels()
    sx = convolve_channel(smooth, gx)
    sy = convolve_channel(smooth, gy)
    magnitude = np.sqrt(sx ** 2 + sy ** 2)
    angle = np.degrees(np.arctan2(sy, sx)) % 180.0
    # 3) Non-Maximum Suppression
    nms = np.zeros((h, w), dtype=np.float32)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            a = angle[y, x]
            m = magnitude[y, x]
            if (0 <= a < 22.5) or (157.5 <= a < 180):
                n1, n2 = magnitude[y, x - 1], magnitude[y, x + 1]
            elif 22.5 <= a < 67.5:
                n1, n2 = magnitude[y - 1, x + 1], magnitude[y + 1, x - 1]
            elif 67.5 <= a < 112.5:
                n1, n2 = magnitude[y - 1, x], magnitude[y + 1, x]
            else:
                n1, n2 = magnitude[y - 1, x - 1], magnitude[y + 1, x + 1]
            if m >= n1 and m >= n2:
                nms[y, x] = m
    # قياس الدرجات في نطاق 0..255
    scale = 255.0 / (magnitude.max() if magnitude.max() > 0 else 1.0)
    nms *= scale
    low = clamp_int(low, 5, 255, 60)
    high = clamp_int(high, low, 255, 160)
    strong = nms >= high
    weak = (nms >= low) & (nms < high)
    # 4) Hysteresis: ربط الحواف الضعيفة المتصلة بحواف قوية.
    out = strong.copy()
    visited = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for y in range(h):
        for x in range(w):
            if strong[y, x] and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))
                while queue:
                    cy, cx = queue.popleft()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                                if weak[ny, nx]:
                                    visited[ny, nx] = True
                                    out[ny, nx] = True
                                    queue.append((ny, nx))
    edge = np.where(out, 255, 0)
    return put_alpha(Image.fromarray(edge.astype(np.uint8)), image.getchannel("A"))


# ═══════════════════════════════════════════════════════════════════════════
#  المؤثرات الفنية والإبداعية  —  قسم «الميزات»
# ═══════════════════════════════════════════════════════════════════════════

def apply_vignette(image: Image.Image, strength: float = 0.6, size: float = 0.55) -> Image.Image:
    """تعتيم الأطراف (Vignette) لإبراز مركز الصورة."""
    strength = clamp_float(strength, 0.0, 1.0, 0.6)
    size = clamp_float(size, 0.1, 1.0, 0.55)
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt(((xx - cx) / (w / 2.0)) ** 2 + ((yy - cy) / (h / 2.0)) ** 2) / size
    mask = np.clip(1.0 - (r ** 2) * strength, 0.0, 1.0)[..., None]
    out = arr * mask
    return put_alpha(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), image.getchannel("A"))


def apply_grain(image: Image.Image, amount: float = 0.15) -> Image.Image:
    """حبّة فيلم (Film Grain) — ضجيج عشوائي، يمنح مظهراً سينمائياً."""
    amount = clamp_float(amount, 0.0, 1.0, 0.15)
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 255 * amount, arr.shape).astype(np.float32)
    out = arr + noise
    return put_alpha(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), image.getchannel("A"))


def apply_split_tone(image: Image.Image, shadow: tuple[int, int, int], highlight: tuple[int, int, int]) -> Image.Image:
    """Split-Tone: تلوين الظلال بلون والإضاءات بلون آخر (مظهر سينمائي)."""
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]) / 255.0
    out = np.zeros_like(arr)
    for c in range(3):
        sh = shadow[c] * (1 - gray)
        hi = highlight[c] * gray
        out[..., c] = sh + hi
    return put_alpha(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), image.getchannel("A"))


def apply_color_pop(image: Image.Image, color: tuple[int, int, int], tolerance: float = 90) -> Image.Image:
    """Color Pop: يُخفي الألوان عدا اللون المختار (يبرزه فقط)."""
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])[..., None]
    dist = np.sqrt(((arr - np.array(color)) ** 2).sum(axis=-1, keepdims=True))
    keep = np.clip(1.0 - dist / tolerance, 0, 1)
    out = gray + (arr - gray) * keep
    return put_alpha(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), image.getchannel("A"))


def apply_oil_paint(image: Image.Image, radius: int = 3) -> Image.Image:
    """مؤثر رسم زيتي — يستبدل كل بكسل بأكثر لون تكراراً داخل جيرانه."""
    radius = clamp_int(radius, 1, 8, 3)
    arr = np.asarray(image.convert("RGB"))
    alpha = np.asarray(image.getchannel("A"))
    h, w = arr.shape[:2]
    out = arr.copy()
    step = radius
    for y in range(h):
        for x in range(w):
            y0, y1 = max(0, y - radius), min(h, y + radius + 1)
            x0, x1 = max(0, x - radius), min(w, x + radius + 1)
            block = arr[y0:y1, x0:x1].reshape(-1, 3)
            # نستخدم تمثيل مقسّم لتجميع الألوان المتقاربة.
            quant = (block // (step * 8)) * (step * 8)
            colors, counts = np.unique(quant, axis=0, return_counts=True)
            out[y, x] = colors[int(np.argmax(counts))]
    return to_image(out.astype(np.float32), alpha.astype(np.float32))


def apply_cartoon(image: Image.Image, smooth: int = 4, edges: bool = True) -> Image.Image:
    """مؤثر كرتوني: تنعيم قوي + تراكب حواف سوداء مبالغ فيها."""
    rgb = image.convert("RGB").filter(ImageFilter.MedianFilter(size=5))
    rgb = rgb.filter(ImageFilter.GaussianBlur(radius=float(smooth)))
    if edges:
        gray = ImageOps.grayscale(rgb)
        edge = gray.filter(ImageFilter.FIND_EDGES)
        edge = ImageOps.invert(edge).point(lambda p: 0 if p > 40 else 255)
        edge = edge.convert("L")
        rgb = Image.composite(rgb, Image.new("RGB", rgb.size, (0, 0, 0)), edge)
    return put_alpha(rgb, image.getchannel("A"))


def apply_reflection(image: Image.Image, alpha: float = 0.4, gap_ratio: float = 0.04) -> Image.Image:
    """انعكاس زجاجي أسفل الصورة (مناسب للمنتجات — تأثير عرض رف)."""
    alpha = clamp_float(alpha, 0.0, 1.0, 0.4)
    w, h = image.size
    gap = max(4, int(h * gap_ratio))
    flip = image.transpose(Image.FLIP_TOP_BOTTOM)
    canvas = Image.new("RGBA", (w, h * 2 + gap), (0, 0, 0, 0))
    canvas.alpha_composite(image, (0, 0))
    # نُخفي الانعكاس تدريجياً (تدرّج لأسفل).
    mask = Image.new("L", (w, h), 0)
    mask = Image.linear_gradient("L").resize((w, h))
    # نعكس التدرّج ليبدأ من الأعلى.
    mask = ImageOps.invert(mask)
    flip = Image.composite(flip, Image.new("RGBA", (w, h), (0, 0, 0, 0)), mask)
    flipped_alpha = flip.getchannel("A").point(lambda p: int(p * alpha))
    flip.putalpha(flipped_alpha)
    canvas.alpha_composite(flip, (0, h + gap))
    return canvas


def apply_watermark(image: Image.Image, text: str, opacity: float = 0.35, position: str = "br") -> Image.Image:
    """علامة مائية نصية شفافة فوق الصورة."""
    if not text:
        return image
    opacity = clamp_float(opacity, 0.05, 1.0, 0.35)
    w, h = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    size = max(12, int(min(w, h) * 0.05))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = int(min(w, h) * 0.04)
    pos = {
        "tl": (pad, pad),
        "tr": (w - pad - tw, pad),
        "bl": (pad, h - pad - th),
        "br": (w - pad - tw, h - pad - th),
        "center": ((w - tw) // 2, (h - th) // 2),
    }
    x, y = pos.get(position, pos["br"])
    draw.text((x, y), text, font=font, fill=(255, 255, 255, int(255 * opacity)))
    out = Image.alpha_composite(image.convert("RGBA"), overlay)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  إزالة الخلفية + الدمج + خلفيات المنتجات
# ═══════════════════════════════════════════════════════════════════════════

def _median(values: list[int]) -> int:
    values.sort()
    return values[len(values) // 2]


def estimate_border_color(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    w, h = rgb.size
    step = max(1, int(max(w, h) / 80))
    samples: list[tuple[int, int, int]] = []
    for x in range(0, w, step):
        samples.append(rgb.getpixel((x, 0)))
        samples.append(rgb.getpixel((x, h - 1)))
    for y in range(0, h, step):
        samples.append(rgb.getpixel((0, y)))
        samples.append(rgb.getpixel((w - 1, y)))
    return (_median([p[0] for p in samples]), _median([p[1] for p in samples]), _median([p[2] for p in samples]))


def remove_background_simple(image: Image.Image, tolerance: int = 52) -> Image.Image:
    """إزالة الخلفية المتصلة بالحواف عبر flood fill لوني (خفيف وتعليمي)."""
    w, h = image.size
    if w * h > 12_000_000:
        scale = math.sqrt(12_000_000 / (w * h))
        small = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        return remove_background_simple(small, tolerance).resize((w, h), Image.Resampling.LANCZOS)
    rgb = image.convert("RGB")
    px = rgb.load()
    ref = estimate_border_color(image)
    tol = clamp_int(tolerance, 5, 130, 52)
    tol2 = tol * tol
    mask = bytearray(w * h)
    light_bg = min(ref) >= 180 and max(ref) - min(ref) < 55
    for y in range(h):
        row = y * w
        for x in range(w):
            r, g, b = px[x, y]
            distance = (r - ref[0]) ** 2 + (g - ref[1]) ** 2 + (b - ref[2]) ** 2
            near_white = light_bg and r >= 238 and g >= 238 and b >= 238
            if distance <= tol2 or near_white:
                mask[row + x] = 1
    visited = bytearray(w * h)
    queue: deque[tuple[int, int]] = deque()

    def add_edge(x: int, y: int) -> None:
        index = y * w + x
        if mask[index] and not visited[index]:
            visited[index] = 1
            queue.append((x, y))

    for x in range(w):
        add_edge(x, 0)
        add_edge(x, h - 1)
    for y in range(h):
        add_edge(0, y)
        add_edge(w - 1, y)
    while queue:
        x, y = queue.popleft()
        if x > 0:
            add_edge(x - 1, y)
        if x + 1 < w:
            add_edge(x + 1, y)
        if y > 0:
            add_edge(x, y - 1)
        if y + 1 < h:
            add_edge(x, y + 1)
    out = image.copy().convert("RGBA")
    out_px = out.load()
    for y in range(h):
        row = y * w
        for x in range(w):
            if visited[row + x]:
                r, g, b, _ = out_px[x, y]
                out_px[x, y] = (r, g, b, 0)
    return out


def remove_background(image: Image.Image, payload: dict[str, Any]) -> Image.Image:
    tolerance = clamp_int(payload.get("tolerance"), 5, 130, 52)
    if payload.get("method") == "ai":
        try:
            from rembg import remove  # type: ignore
            raw = io.BytesIO()
            image.save(raw, format="PNG")
            res = remove(raw.getvalue())
            return Image.open(io.BytesIO(res)).convert("RGBA")
        except Exception:
            pass  # رجوع صامت إلى الخوارزمية الخفيفة.
    return remove_background_simple(image, tolerance)


def composite_on_background(image: Image.Image, background: Image.Image, fit: str = "cover") -> Image.Image:
    """وضع صورة (بشفافية) فوق خلفية كبيرة — تُحجَّم وفقاً للخلفية."""
    bg = background.convert("RGBA")
    bw, bh = bg.size
    target_w, target_h = bw, bh
    if fit == "contain":
        ratio = min(bw / image.width, bh / image.height)
        target_w, target_h = max(1, round(image.width * ratio)), max(1, round(image.height * ratio))
        base = Image.new("RGBA", bg.size, (0, 0, 0, 0))
        base.alpha_composite(bg)
        x = (bw - target_w) // 2
        y = (bh - target_h) // 2
        subject = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        base.alpha_composite(subject, (x, y))
        return base
    # cover: الخلفية تُقصّ/تُحجَّم لتطابق أبعاد الصورة.
    ratio = max(target_w / image.width, target_h / image.height)
    new_w, new_h = max(1, round(image.width * ratio)), max(1, round(image.height * ratio))
    bg = bg.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x = (new_w - target_w) // 2
    y = (new_h - target_h) // 2
    bg = bg.crop((x, y, x + target_w, y + target_h))
    bg.alpha_composite(image, (0, 0))
    return bg


def blend_images(base: Image.Image, overlay: Image.Image, mode: str, opacity: float) -> Image.Image:
    overlay = ImageOps.contain(overlay, base.size, Image.Resampling.LANCZOS)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    x = (base.width - overlay.width) // 2
    y = (base.height - overlay.height) // 2
    layer.alpha_composite(overlay, (x, y))
    alpha = layer.getchannel("A").point(lambda p: int(p * opacity))
    layer.putalpha(alpha)
    base_rgb = base.convert("RGB")
    layer_rgb = layer.convert("RGB")
    mode = mode.lower()
    if mode == "multiply":
        mixed = ImageChops.multiply(base_rgb, layer_rgb)
    elif mode == "screen":
        mixed = ImageChops.screen(base_rgb, layer_rgb)
    elif mode == "difference":
        mixed = ImageChops.difference(base_rgb, layer_rgb)
    elif mode == "add":
        mixed = ImageChops.add(base_rgb, layer_rgb, scale=2.0, offset=0)
    elif mode == "soft_light":
        # Soft Light مبسّط.
        b = np.asarray(base_rgb).astype(np.float32) / 255.0
        l = np.asarray(layer_rgb).astype(np.float32) / 255.0
        m = np.where(l > 0.5, 1 - (1 - b) * (1 - (l - 0.5) * 2), b * (l * 2))
        mixed = Image.fromarray(np.clip(m * 255, 0, 255).astype(np.uint8))
    else:
        mixed = layer_rgb
    if mode == "normal":
        out = Image.alpha_composite(base.convert("RGBA"), layer)
    else:
        out_rgb = Image.composite(mixed, base_rgb, alpha)
        out = out_rgb.convert("RGBA")
        out.putalpha(base.getchannel("A"))
    return out


def resize_keep_ratio(image: Image.Image, width: int | None, height: int | None, keep: bool) -> Image.Image:
    old_w, old_h = image.size
    if not width and not height:
        return image
    if keep:
        if width and height:
            ratio = min(width / old_w, height / old_h)
            width, height = max(1, round(old_w * ratio)), max(1, round(old_h * ratio))
        elif width:
            height = max(1, round(old_h * width / old_w))
        else:
            width = max(1, round(old_w * height / old_h))
    else:
        width = width or old_w
        height = height or old_h
    return image.resize((max(1, width), max(1, height)), Image.Resampling.LANCZOS)


def crop_image(image: Image.Image, payload: dict[str, Any]) -> Image.Image:
    w, h = image.size
    x = clamp_int(payload.get("x"), 0, w - 1, 0)
    y = clamp_int(payload.get("y"), 0, h - 1, 0)
    cw = clamp_int(payload.get("width"), 1, w - x, w - x)
    ch = clamp_int(payload.get("height"), 1, h - y, h - y)
    return image.crop((x, y, x + cw, y + ch))


def skew_image(image: Image.Image, skew_x: float, skew_y: float) -> Image.Image:
    w, h = image.size
    tx = math.tan(math.radians(max(-45.0, min(45.0, skew_x))))
    ty = math.tan(math.radians(max(-45.0, min(45.0, skew_y))))
    coeffs = (1.0, -tx, tx * h / 2.0, -ty, 1.0, ty * w / 2.0)
    return image.transform((w, h), Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BICUBIC)


def add_border(image: Image.Image, size: int, color: tuple[int, int, int, int]) -> Image.Image:
    size = clamp_int(size, 0, 80, 0)
    if size == 0:
        return image
    return ImageOps.expand(image, border=size, fill=color)


# ═══════════════════════════════════════════════════════════════════════════
#  الرسم على الصور (يقوم الباك-اند بطباعة النص/الأشكال إن أُرسلت إحداثيات)
# ═══════════════════════════════════════════════════════════════════════════

def draw_on_image(image: Image.Image, payload: dict[str, Any]) -> Image.Image:
    """يرسم نصاً أو شكلاً فوق الصورة بإحداثيات نسبية (0..1)."""
    overlay = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size
    items = payload.get("items", [])
    for item in items:
        kind = item.get("kind")
        # إحداثيات نسبة (0..1) ← بكسل.
        nx, ny = item.get("x", 0.5), item.get("y", 0.5)
        px, py = int(nx * w), int(ny * h)
        if kind == "text":
            text = str(item.get("text", ""))
            if not text:
                continue
            size = clamp_int(item.get("size"), 8, 400, 42)
            color = item.get("color", "#ffffff")
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
            except Exception:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((px - tw / 2, py - th / 2), text, font=font, fill=color)
        elif kind == "rect":
            c = item.get("color", "#57d4f2")
            sw = max(1, int(item.get("width", 0.1) * w))
            sh = max(1, int(item.get("height", 0.1) * h))
            draw.rectangle([px - sw // 2, py - sh // 2, px + sw // 2, py + sh // 2], outline=c, width=clamp_int(item.get("stroke", 3), 1, 40, 3))
        elif kind == "circle":
            c = item.get("color", "#57d4f2")
            r = max(2, item.get("radius", 0.05) * min(w, h))
            draw.ellipse([px - r, py - r, px + r, py + r], outline=c, width=clamp_int(item.get("stroke", 3), 1, 40, 3))
    return overlay


# ═══════════════════════════════════════════════════════════════════════════
#  الموجّه الرئيسي (process)
# ═══════════════════════════════════════════════════════════════════════════

def process(payload: dict[str, Any]) -> tuple[Image.Image, str]:
    if not isinstance(payload, dict):
        raise ValueError("الطلب غير صالح")
    image = decode_data_url(payload.get("image", ""))
    op = str(payload.get("op", "")).lower()
    output_format = str(payload.get("output_format", "PNG")).upper()

    # ─── التحويل والصيغ ───────
    if op == "convert":
        return image, output_format

    # ─── العمليات الهندسية ───────
    if op == "resize":
        width = clamp_int(payload.get("width"), 1, 10000, image.width)
        height = clamp_int(payload.get("height"), 1, 10000, image.height)
        return resize_keep_ratio(image, width, height, bool(payload.get("keep", True))), "PNG"
    if op == "rotate":
        angle = clamp_float(payload.get("angle"), -3600, 3600, 0)
        return image.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC), "PNG"
    if op == "flip":
        direction = payload.get("direction", "horizontal")
        return (ImageOps.flip(image) if direction == "vertical" else ImageOps.mirror(image)), "PNG"
    if op == "crop":
        return crop_image(image, payload), "PNG"
    if op == "skew":
        return skew_image(image, clamp_float(payload.get("skew_x"), -45, 45, 0), clamp_float(payload.get("skew_y"), -45, 45, 0)), "PNG"
    if op == "border":
        color = payload.get("color", [30, 41, 59, 255])
        if not isinstance(color, list) or len(color) < 3:
            color = [30, 41, 59, 255]
        rgba = tuple(clamp_int(v, 0, 255, 0) for v in (color + [255])[:4])
        return add_border(image, payload.get("size", 8), rgba), "PNG"

    # ─── تحسينات البكسل ───────
    if op == "brightness":
        return ImageEnhance.Brightness(image).enhance(1 + clamp_float(payload.get("value"), -100, 100, 0) / 100), "PNG"
    if op == "contrast":
        return ImageEnhance.Contrast(image).enhance(1 + clamp_float(payload.get("value"), -100, 100, 0) / 100), "PNG"
    if op == "saturation":
        return ImageEnhance.Color(image).enhance(1 + clamp_float(payload.get("value"), -100, 100, 0) / 100), "PNG"
    if op == "sharpness_amount":
        return ImageEnhance.Sharpness(image).enhance(1 + clamp_float(payload.get("value"), -100, 100, 0) / 100), "PNG"
    if op == "gamma":
        return apply_gamma(image, clamp_float(payload.get("value"), 0.2, 4.0, 1.0)), "PNG"
    if op == "auto":
        out = ImageOps.autocontrast(image.convert("RGB"))
        return put_alpha(out, image.getchannel("A")), "PNG"
    if op == "grayscale":
        gray = ImageOps.grayscale(image)
        return Image.merge("RGBA", (gray, gray, gray, image.getchannel("A"))), "PNG"
    if op == "negative":
        return apply_negative(image), "PNG"
    if op == "logarithm":
        return apply_logarithm(image), "PNG"
    if op == "equalize":
        return apply_equalize(image), "PNG"
    if op == "invert":
        rgb = ImageOps.invert(image.convert("RGB"))
        return put_alpha(rgb, image.getchannel("A")), "PNG"
    if op == "threshold":
        gray = ImageOps.grayscale(image)
        bw = gray.point(lambda p: 255 if p >= clamp_int(payload.get("value"), 0, 255, 128) else 0)
        return Image.merge("RGBA", (bw, bw, bw, image.getchannel("A"))), "PNG"
    if op == "posterize":
        out = ImageOps.posterize(image.convert("RGB"), clamp_int(payload.get("value"), 1, 8, 4))
        return put_alpha(out, image.getchannel("A")), "PNG"
    if op == "solarize":
        out = ImageOps.solarize(image.convert("RGB"), clamp_int(payload.get("value"), 0, 255, 128))
        return put_alpha(out, image.getchannel("A")), "PNG"
    if op == "pixelate":
        block = clamp_int(payload.get("block"), 2, 100, 10)
        small = image.resize((max(1, image.width // block), max(1, image.height // block)), Image.Resampling.BILINEAR)
        return small.resize(image.size, Image.Resampling.NEAREST), "PNG"
    if op == "duotone":
        gray = ImageOps.grayscale(image)
        dark = tuple(payload.get("dark", [25, 35, 68]))
        light = tuple(payload.get("light", [247, 171, 72]))
        dark = tuple(clamp_int(v, 0, 255, 0) for v in dark)
        light = tuple(clamp_int(v, 0, 255, 0) for v in light)
        lut: list[int] = []
        for i in range(256):
            ratio = i / 255
            lut.extend([int(dark[0] * (1 - ratio) + light[0] * ratio),
                        int(dark[1] * (1 - ratio) + light[1] * ratio),
                        int(dark[2] * (1 - ratio) + light[2] * ratio)])
        rgb = gray.convert("RGB").point(lut)
        return put_alpha(rgb, image.getchannel("A")), "PNG"
    if op == "hsv":
        return apply_hsv_adjust(image, clamp_float(payload.get("hue"), -255, 255, 0),
                                clamp_float(payload.get("saturation"), -100, 100, 0),
                                clamp_float(payload.get("value"), -100, 100, 0)), "PNG"
    if op == "color_balance":
        return apply_color_balance(image, clamp_float(payload.get("red"), -100, 100, 0),
                                   clamp_float(payload.get("green"), -100, 100, 0),
                                   clamp_float(payload.get("blue"), -100, 100, 0)), "PNG"
    if op == "vibrance":
        return apply_vibrance(image, clamp_float(payload.get("value"), -100, 100, 0)), "PNG"
    if op == "sepia":
        return apply_sepia(image, clamp_float(payload.get("value"), 0.0, 1.0, 1.0)), "PNG"

    # ─── الفلاتر (تنعيم / تحديد) ───────
    if op == "filter":
        preset = {
            "blur": ImageFilter.BLUR,
            "smooth": ImageFilter.SMOOTH,
            "smooth_more": ImageFilter.SMOOTH_MORE,
            "sharpen": ImageFilter.SHARPEN,
            "detail": ImageFilter.DETAIL,
            "edge": ImageFilter.FIND_EDGES,
            "emboss": ImageFilter.EMBOSS,
            "contour": ImageFilter.CONTOUR,
            "find_edges": ImageFilter.FIND_EDGES,
        }
        name = str(payload.get("filter", "smooth"))
        if name == "unsharp":
            return image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)), "PNG"
        if name not in preset:
            raise ValueError("فلتر غير معروف")
        return image.filter(preset[name]), "PNG"
    if op == "mean":
        return apply_mean_filter(image, clamp_int(payload.get("size"), 3, 41, 3)), "PNG"
    if op == "gaussian":
        return apply_gaussian_filter(image, clamp_float(payload.get("radius"), 0.1, 30, 2.0)), "PNG"
    if op == "median":
        return apply_median_filter(image, clamp_int(payload.get("size"), 3, 25, 3)), "PNG"
    if op == "custom":
        kernel = payload.get("kernel")
        if not isinstance(kernel, list) or not all(isinstance(r, list) for r in kernel):
            raise ValueError("أرسل نواة kernel كقائمة قوائم")
        return apply_custom_filter(image, kernel, normalize=bool(payload.get("normalize", True))), "PNG"

    # ─── كشف الحواف ───────
    if op == "laplacian":
        return apply_laplacian(image, payload.get("mode", "abs")), "PNG"
    if op == "laplacian_edge":
        return apply_laplacian_edge(image), "PNG"
    if op == "difference":
        return apply_difference_filter(image), "PNG"
    if op == "sobel":
        return apply_sobel(image, payload.get("axis")), "PNG"
    if op == "canny":
        return apply_canny(image, clamp_int(payload.get("low"), 5, 255, 60), clamp_int(payload.get("high"), 5, 255, 160)), "PNG"

    # ─── المؤثرات الفنية (الميزات) ───────
    if op == "vignette":
        return apply_vignette(image, clamp_float(payload.get("strength"), 0, 1, 0.6), clamp_float(payload.get("size"), 0.1, 1, 0.55)), "PNG"
    if op == "grain":
        return apply_grain(image, clamp_float(payload.get("amount"), 0, 1, 0.15)), "PNG"
    if op == "split_tone":
        return apply_split_tone(image, tuple(payload.get("shadow", [25, 35, 68])), tuple(payload.get("highlight", [247, 171, 72]))), "PNG"
    if op == "color_pop":
        return apply_color_pop(image, tuple(payload.get("color", [87, 212, 242])), clamp_float(payload.get("tolerance"), 10, 255, 90)), "PNG"
    if op == "oil_paint":
        return apply_oil_paint(image, clamp_int(payload.get("radius"), 1, 8, 3)), "PNG"
    if op == "cartoon":
        return apply_cartoon(image, clamp_int(payload.get("smooth"), 1, 10, 4), bool(payload.get("edges", True))), "PNG"
    if op == "reflection":
        return apply_reflection(image, clamp_float(payload.get("alpha"), 0, 1, 0.4)), "PNG"
    if op == "watermark":
        return apply_watermark(image, str(payload.get("text", "")), clamp_float(payload.get("opacity"), 0.05, 1, 0.35), str(payload.get("position", "br"))), "PNG"

    # ─── الرسم ───────
    if op == "draw":
        return draw_on_image(image, payload), "PNG"

    # ─── إزالة الخلفية + الدمج + خلفيات المنتجات ───────
    if op == "remove_background":
        return remove_background(image, payload), "PNG"
    if op == "blend":
        second = decode_data_url(payload.get("image2", ""))
        return blend_images(image, second, str(payload.get("mode", "normal")), clamp_float(payload.get("opacity"), 0, 1, 0.5)), "PNG"
    if op == "background_replace":
        bg_name = str(payload.get("bg", ""))
        if not bg_name or not (BACKGROUND_DIR / bg_name).exists():
            raise ValueError("الخلفية المطلوبة غير موجودة في المكتبة")
        bg = Image.open(BACKGROUND_DIR / bg_name).convert("RGBA")
        return composite_on_background(image, bg, str(payload.get("fit", "cover"))), "PNG"

    raise ValueError("لم يتم التعرف على العملية المطلوبة")


# ═══════════════════════════════════════════════════════════════════════════
#  خادم HTTP
# ═══════════════════════════════════════════════════════════════════════════

class ImageEditorHandler(BaseHTTPRequestHandler):
    server_version = "MirrorImageStudio/2.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[server] " + (fmt % args) + "\n")
        sys.stdout.flush()

    def _headers(self, content_type: str, length: int | None = None) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        if length is not None:
            self.send_header("Content-Length", str(length))

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._headers("application/json; charset=utf-8", len(raw))
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._headers("text/plain", 0)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _get_path(self) -> str:
        # فك ترميز المسار لدعم الأسماء العربية
        raw_path = urlparse(self.path).path
        try:
            return unquote(raw_path, encoding='utf-8')
        except Exception:
            return raw_path

    def do_HEAD(self) -> None:  # noqa: N802
        # دعم HEAD للمتصفحات التي تطلب معلومات الملف قبل التحميل
        path = self._get_path()
        if path in GUIDE_ALIASES:
            target = GUIDE_ALIASES[path]
            if not target.exists():
                self.send_json({"ok": False, "error": "دليل غير موجود"}, 404)
                return
            self.send_response(200)
            ct = "application/pdf" if target.suffix.lower() == ".pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            self.send_header("Content-Disposition", f'inline; filename="guide{target.suffix}"')
            self._headers(ct, target.stat().st_size)
            self.end_headers()
            return
        if path.startswith("/backgrounds/"):
            name = path[len("/backgrounds/"):]
            target = (BACKGROUND_DIR / name).resolve()
            if BACKGROUND_DIR.resolve() not in target.parents and target != BACKGROUND_DIR.resolve():
                self.send_response(403)
                self._headers("text/plain", 0)
                self.end_headers()
                return
            if not target.exists():
                self.send_response(404)
                self._headers("text/plain", 0)
                self.end_headers()
                return
            self.send_response(200)
            ct = "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
            self._headers(ct, target.stat().st_size)
            self.end_headers()
            return
        # لباقي المسارات نرجع 200 بدون body أو نستدعي GET logic للـ headers فقط
        self.send_response(200)
        self._headers("text/plain", 0)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self._get_path()
        if path in {"/", "/index.html"}:
            try:
                raw = INDEX_FILE.read_bytes()
            except FileNotFoundError:
                self.send_json({"ok": False, "error": "ملف index.html غير موجود"}, 500)
                return
            self.send_response(200)
            self._headers("text/html; charset=utf-8", len(raw))
            self.end_headers()
            self.wfile.write(raw)
            return
        if path == "/api/health":
            self.send_json({"ok": True, "service": "Mirror Image Studio", "python": sys.version.split()[0]})
            return
        if path == "/api/backgrounds":
            names = sorted([p.name for p in BACKGROUND_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
            self.send_json({"ok": True, "backgrounds": names})
            return
        if path.startswith("/backgrounds/"):
            name = path[len("/backgrounds/"):]
            # منع الخروج عن مجلد الخلفيات (Path traversal).
            target = (BACKGROUND_DIR / name).resolve()
            if BACKGROUND_DIR.resolve() not in target.parents and target != BACKGROUND_DIR.resolve():
                self.send_json({"ok": False, "error": "مسار غير مسموح"}, 403)
                return
            if not target.exists():
                self.send_json({"ok": False, "error": "الخلفية غير موجودة"}, 404)
                return
            data = target.read_bytes()
            self.send_response(200)
            ct = "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
            self._headers(ct, len(data))
            self.end_headers()
            self.wfile.write(data)
            return
        # ─── دليل الاستخدام PDF/DOCX — قابل للتحميل من الموقع ───
        if path in GUIDE_ALIASES:
            target = GUIDE_ALIASES[path]
            if not target.exists():
                self.send_json({"ok": False, "error": "دليل الاستخدام غير موجود على الخادم — تأكد من وجود الملف"}, 404)
                return
            data = target.read_bytes()
            self.send_response(200)
            if target.suffix.lower() == ".pdf":
                ct = "application/pdf"
                # استخدام اسم ASCII آمن في الـ header لتجنب خطأ latin-1
                self.send_header("Content-Disposition", f'inline; filename="guide.pdf"; filename*=UTF-8\'\'%D8%AF%D9%84%D9%8A%D9%84-%D8%A7%D9%84%D8%A7%D8%B3%D8%AA%D8%AE%D8%AF%D8%A7%D9%85.pdf')
            else:
                ct = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                self.send_header("Content-Disposition", f'attachment; filename="guide.docx"')
            self._headers(ct, len(data))
            self.end_headers()
            self.wfile.write(data)
            return
        if path in {"/api/guide", "/api/help", "/help"}:
            # معلومات عن الدليل + روابط التحميل
            has_pdf = GUIDE_PDF.exists()
            has_docx = GUIDE_DOCX.exists()
            self.send_json({
                "ok": True,
                "service": "Mirror Image Studio — دليل الاستخدام",
                "files": {
                    "pdf": {
                        "exists": has_pdf,
                        "size": GUIDE_PDF.stat().st_size if has_pdf else 0,
                        "url": "/دليل_الاستخدام_الشامل_مرآة.pdf",
                        "aliases": ["/guide.pdf", "/دليل_الاستخدام.pdf", "/help.pdf"]
                    },
                    "docx": {
                        "exists": has_docx,
                        "size": GUIDE_DOCX.stat().st_size if has_docx else 0,
                        "url": "/دليل_الاستخدام_الشامل_مرآة.docx",
                        "aliases": ["/guide.docx"]
                    }
                },
                "help": "افتح /guide.pdf أو /دليل_الاستخدام_الشامل_مرآة.pdf لتحميل الدليل، أو استخدم زر المساعدة داخل الموقع"
            })
            return
        self.send_json({"ok": False, "error": "المسار غير موجود"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/process":
            self.send_json({"ok": False, "error": "المسار غير موجود"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            self.send_json({"ok": False, "error": "حجم الطلب غير صالح أو كبير جداً"}, 413)
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result, output_format = process(payload)
            data = encode_data_url(result, output_format, payload.get("quality", 92))
            self.send_json({"ok": True, "data": data, "width": result.width, "height": result.height})
        except ValueError as exc:
            self.send_json({"ok": False, "error": str(exc)}, 400)
        except Exception as exc:  # لا نسرب stack trace للمستخدم النهائي
            self.send_json({"ok": False, "error": f"حدث خطأ أثناء المعالجة: {exc}"}, 500)


def main() -> None:
    if not INDEX_FILE.exists():
        print("خطأ: ضع app.py و index.html في المجلد نفسه.")
        raise SystemExit(1)
    if not BACKGROUND_DIR.exists():
        BACKGROUND_DIR.mkdir(exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), ImageEditorHandler)
    print("=" * 64)
    print("  مِرآة — استوديو الصور المتقدم  |  Mirror Image Studio")
    print("=" * 64)
    print(f"  افتح المتصفح على:  http://localhost:{PORT}")
    print("  أوقف الخادم عبر Ctrl+C")
    print("=" * 64)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nتم إيقاف الخادم.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
