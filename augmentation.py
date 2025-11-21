# hw3_q1_augment.py
# Extended augmentations: 18 techniques total
# Top row    : kitty image with 9 different augmentations
# Bottom row : selected image with 9 different augmentations

import io
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt


# -----------------------------
# Basic geometric + noise
# -----------------------------
def aug_flip_horizontal(img: Image.Image) -> Image.Image:
    # Flip (horizontal)
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def aug_scale_zoom_in(img: Image.Image, scale: float = 1.3) -> Image.Image:
    # Scale / zoom-in + center crop back
    w, h = img.size
    new_w, new_h = int(w * scale), int(h * scale)
    img_scaled = img.resize((new_w, new_h), Image.BICUBIC)

    left = (new_w - w) // 2
    top = (new_h - h) // 2
    right = left + w
    bottom = top + h
    return img_scaled.crop((left, top, right, bottom))


def aug_center_crop(img: Image.Image, crop_ratio: float = 0.6) -> Image.Image:
    # Center crop + resize back
    w, h = img.size
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)

    left = (w - cw) // 2
    top = (h - ch) // 2
    right = left + cw
    bottom = top + ch

    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((w, h), Image.BICUBIC)


def aug_translate(img: Image.Image, shift_x: int = 60, shift_y: int = 40) -> Image.Image:
    # Translation (shift image)
    w, h = img.size
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, 0, shift_x, 0, 1, shift_y),
        fillcolor=(0, 0, 0),
    )


def aug_gaussian_noise(img: Image.Image, sigma: float = 25.0) -> Image.Image:
    # Add Gaussian noise
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0.0, sigma, arr.shape)
    noisy = arr + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def aug_rotation(img: Image.Image, angle: float = 30.0) -> Image.Image:
    # Rotation (no expand, fill with black)
    return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))


def aug_shear(img: Image.Image, shear: float = 0.3) -> Image.Image:
    # Shearing along x-axis
    w, h = img.size
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, shear, 0, 0, 1, 0),
        resample=Image.BICUBIC,
        fillcolor=(0, 0, 0),
    )


def aug_brightness_contrast(img: Image.Image, alpha: float = 1.3, beta: float = 20.0) -> Image.Image:
    """
    Brightness / Contrast adjustment
    out = alpha * img + beta
    """
    arr = np.array(img).astype(np.float32)
    arr = alpha * arr + beta
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def aug_gamma(img: Image.Image, gamma: float = 0.5) -> Image.Image:
    """
    Gamma correction
    out = 255 * (img/255)^gamma
    gamma < 1 -> brighter, gamma > 1 -> darker
    """
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.power(arr, gamma)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# -----------------------------
# Advanced photometric + others
# -----------------------------
def aug_hist_equalization(img: Image.Image) -> Image.Image:
    # Histogram equalization on Y channel (YCrCb)
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y_eq = ImageOps.equalize(y)
    return Image.merge("YCbCr", (y_eq, cb, cr)).convert("RGB")


def aug_color_jitter(img: Image.Image) -> Image.Image:
    """
    Color Jitter (strong, deterministic version)
    - Increase brightness and contrast in RGB
    - Increase saturation and slightly shift hue in HSV
    """
    # ----- Brightness / Contrast in RGB -----
    arr = np.array(img).astype(np.float32)

    # ปรับแรงขึ้นให้เห็นชัด: +40% brightness, +50% contrast
    b_factor = 1.4   # brightness
    c_factor = 1.5   # contrast

    # contrast around midpoint 127.5
    arr = (arr - 127.5) * c_factor + 127.5
    arr = arr * b_factor
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img_bc = Image.fromarray(arr)

    # ----- Saturation / Hue ใน HSV -----
    hsv = img_bc.convert("HSV")
    h, s, v = hsv.split()
    h = np.array(h).astype(np.float32)
    s = np.array(s).astype(np.float32)
    v = np.array(v).astype(np.float32)

    # เพิ่ม saturation 60%, เลื่อน hue ประมาณ 10 องศา
    s_factor = 1.6
    h_shift = 10.0 * 255.0 / 360.0  # 10 degrees on hue circle

    s = np.clip(s * s_factor, 0, 255)
    h = (h + h_shift) % 255

    hsv_jit = Image.merge(
        "HSV",
        (Image.fromarray(h.astype(np.uint8)),
         Image.fromarray(s.astype(np.uint8)),
         Image.fromarray(v.astype(np.uint8)))
    )
    return hsv_jit.convert("RGB")



def aug_random_grayscale(img: Image.Image) -> Image.Image:
    # Convert to grayscale then back to RGB
    return img.convert("L").convert("RGB")


# def aug_gaussian_blur(img: Image.Image, radius: float = 2.0) -> Image.Image:
#     return img.filter(ImageFilter.GaussianBlur(radius=radius))

def aug_gaussian_blur(img: Image.Image) -> Image.Image:
    """
    Gaussian blur (strong)
    - radius = 4.0 ทำให้ภาพนุ่มและรายละเอียดหายไปชัดเจน
    """
    return img.filter(ImageFilter.GaussianBlur(radius=4.0))


# def aug_sharpen(img: Image.Image) -> Image.Image:
#     # Unsharp mask for sharpening
#     return img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

def aug_sharpen(img: Image.Image) -> Image.Image:
    """
    Extra-strong sharpening (visible effect)
    - Use a high-pass filter + amplify edges strongly
    - k = 3.0 gives a very clear sharpened result
    """
    arr = np.array(img).astype(np.float32)

    # blur แบบ radius ใหญ่ขึ้นเพื่อให้ส่วนต่างคมชัดกว่าเดิม
    blurred = img.filter(ImageFilter.GaussianBlur(radius=2.0))
    arr_blur = np.array(blurred).astype(np.float32)

    # High-pass component: edge = original - blur
    edge = arr - arr_blur

    # Amplify edge by factor k
    k = 3.0   # ยิ่งมากยิ่งคม (3.0 เห็นชัดมาก)
    sharp = arr + k * edge

    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return Image.fromarray(sharp)


# def aug_salt_pepper(img: Image.Image, amount: float = 0.02, s_vs_p: float = 0.5) -> Image.Image:
#     """
#     Salt & Pepper noise:
#     amount = fraction of pixels to corrupt
#     s_vs_p = salt vs pepper ratio
#     """
#     arr = np.array(img)
#     h, w, c = arr.shape
#     num_pixels = h * w

#     # Salt
#     num_salt = int(np.ceil(amount * num_pixels * s_vs_p))
#     coords = (np.random.randint(0, h, num_salt),
#               np.random.randint(0, w, num_salt))
#     arr[coords[0], coords[1], :] = 255

#     # Pepper
#     num_pepper = int(np.ceil(amount * num_pixels * (1.0 - s_vs_p)))
#     coords = (np.random.randint(0, h, num_pepper),
#               np.random.randint(0, w, num_pepper))
#     arr[coords[0], coords[1], :] = 0

#     return Image.fromarray(arr)

def aug_salt_pepper(img: Image.Image, amount: float = 0.03) -> Image.Image:
    """
    Salt & Pepper noise (white and black dots)
    - amount = 0.03 means 3% of pixels become salt/pepper
    """
    arr = np.array(img).copy()
    h, w, c = arr.shape

    num = int(amount * h * w)

    # salt (white)
    coords = (
        np.random.randint(0, h, num),
        np.random.randint(0, w, num)
    )
    arr[coords] = [255, 255, 255]

    # pepper (black)
    coords = (
        np.random.randint(0, h, num),
        np.random.randint(0, w, num)
    )
    arr[coords] = [0, 0, 0]

    return Image.fromarray(arr)

# def aug_speckle_noise(img: Image.Image, sigma: float = 0.2) -> Image.Image:
#     """
#     Speckle noise: img + img * noise
#     """
#     arr = np.array(img).astype(np.float32) / 255.0
#     noise = np.random.normal(0.0, sigma, arr.shape)
#     noisy = arr + arr * noise
#     noisy = np.clip(noisy, 0, 1.0)
#     return Image.fromarray((noisy * 255.0).astype(np.uint8))

def aug_speckle_noise(img: Image.Image, sigma: float = 0.25) -> Image.Image:
    """
    Speckle noise (multiplicative Gaussian noise)
    - noise = image + image * N(0, sigma)
    - sigma = 0.25 gives strong visible speckle
    """
    arr = np.array(img).astype(np.float32) / 255.0

    # multiplicative noise
    noise = np.random.randn(*arr.shape) * sigma
    noisy = arr + arr * noise

    noisy = np.clip(noisy, 0.0, 1.0)
    noisy = (noisy * 255).astype(np.uint8)

    return Image.fromarray(noisy)

def aug_jpeg_artifacts(img: Image.Image, quality: int = 10) -> Image.Image:
    """
    JPEG compression artifacts:
    save to in-memory JPEG with low quality and reload
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# def aug_motion_blur(img: Image.Image, ksize: int = 9) -> Image.Image:
#     """
#     Motion blur (approximate) ด้วยการเฉลี่ยภาพที่เลื่อนในแนวนอนหลาย ๆ ระยะ
#     - ไม่ใช้ ImageFilter.Kernel เพื่อลดปัญหา compatibility
#     """
#     arr = np.array(img).astype(np.float32)
#     result = np.zeros_like(arr)

#     half = ksize // 2
#     for i in range(ksize):
#         shift = i - half
#         # เลื่อนภาพในแนวนอน (axis=1)
#         shifted = np.roll(arr, shift=shift, axis=1)
#         result += shifted

#     result /= float(ksize)
#     result = np.clip(result, 0, 255).astype(np.uint8)
#     return Image.fromarray(result)

def aug_motion_blur(img: Image.Image, ksize: int = 15) -> Image.Image:
    """
    Motion blur (horizontal) แบบไม่ใช้ ImageFilter.Kernel
    - ใช้การเฉลี่ยภาพที่เลื่อนในแนวนอนหลาย ๆ ระยะ
    - ksize ยิ่งใหญ่ ภาพยิ่งเบลอยาว
    """
    arr = np.array(img).astype(np.float32)
    result = np.zeros_like(arr)

    half = ksize // 2
    for i in range(ksize):
        shift = i - half
        # เลื่อนภาพในแนวนอน (axis=1)
        shifted = np.roll(arr, shift=shift, axis=1)
        result += shifted

    result /= float(ksize)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)

# -----------------------------
# Main script
# -----------------------------
def main():
    np.random.seed(0)

    kitty_path = "kitty.jpg"       # or "kitty55.png"
    selected_path = "selected.jpg" # your own image

    kitty = Image.open(kitty_path).convert("RGB")
    other = Image.open(selected_path).convert("RGB")

    # 9 techniques for kitty (top row)
    kitty_funcs = [
        (aug_flip_horizontal,       "Flip (horizontal)"),
        (aug_scale_zoom_in,         "Scale / zoom-in"),
        (aug_center_crop,           "Center crop"),
        (aug_translate,             "Translation"),
        (aug_gaussian_noise,        "Gaussian noise"),
        (aug_rotation,              "Rotation"),
        (aug_shear,                 "Shearing"),
        (aug_brightness_contrast,   "Brightness / Contrast"),
        (aug_gamma,                 "Gamma correction"),
    ]

    # 9 techniques for selected image (bottom row)
    selected_funcs = [
        (aug_hist_equalization,     "Histogram equalization"),
        (aug_color_jitter,          "Color jitter"),
        (aug_random_grayscale,      "Random grayscale"),
        (aug_gaussian_blur,         "Gaussian blur"),
        (aug_sharpen,               "Sharpening"),
        (aug_salt_pepper,           "Salt & Pepper noise"),
        (aug_speckle_noise,         "Speckle noise"),
        (aug_jpeg_artifacts,        "JPEG compression artifacts"),
        (aug_motion_blur,           "Motion blur"),
    ]

    plt.figure(figsize=(20, 6))

    # Top row: kitty (9 columns)
    for i, (func, label) in enumerate(kitty_funcs):
        img_aug = func(kitty)
        ax = plt.subplot(2, 9, i + 1)
        ax.imshow(img_aug)
        ax.set_title(f"Kitty - {label}", fontsize=8)
        ax.axis("off")

    # Bottom row: selected (9 columns)
    for i, (func, label) in enumerate(selected_funcs):
        img_aug = func(other)
        ax = plt.subplot(2, 9, 9 + i + 1)
        ax.imshow(img_aug)
        ax.set_title(f"Selected - {label}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
