
# Image Augmentation Toolkit

A comprehensive Python library implementing 18 image augmentation techniques for deep learning, computer vision, and data preprocessing pipelines.

## Features

- **Pure Python/PIL implementation** - No heavy dependencies like OpenCV
- **18 augmentation techniques** covering geometric, photometric, and noise-based transformations
- **Easy to integrate** into any ML pipeline

## Installation

```bash
pip install numpy pillow matplotlib
```

## Usage

```python
from PIL import Image
from augmentation import aug_flip_horizontal, aug_gaussian_noise, aug_color_jitter

img = Image.open("your_image.jpg").convert("RGB")

# Apply augmentations
flipped = aug_flip_horizontal(img)
noisy = aug_gaussian_noise(img, sigma=25.0)
jittered = aug_color_jitter(img)
```

## Augmentation Techniques

### Geometric Transformations

| # | Technique | Function | Description |
|---|-----------|----------|-------------|
| 1 | **Horizontal Flip** | `aug_flip_horizontal(img)` | Mirrors the image along the vertical axis |
| 2 | **Scale / Zoom-in** | `aug_scale_zoom_in(img, scale=1.3)` | Enlarges image by scale factor and center-crops to original size |
| 3 | **Center Crop** | `aug_center_crop(img, crop_ratio=0.6)` | Crops center portion and resizes back to original dimensions |
| 4 | **Translation** | `aug_translate(img, shift_x=60, shift_y=40)` | Shifts image by pixel offsets, filling empty space with black |
| 5 | **Rotation** | `aug_rotation(img, angle=30.0)` | Rotates image by specified degrees around center |
| 6 | **Shearing** | `aug_shear(img, shear=0.3)` | Applies horizontal shear transformation |

### Photometric Transformations

| # | Technique | Function | Description |
|---|-----------|----------|-------------|
| 7 | **Brightness/Contrast** | `aug_brightness_contrast(img, alpha=1.3, beta=20.0)` | Linear adjustment: `out = alpha * img + beta` |
| 8 | **Gamma Correction** | `aug_gamma(img, gamma=0.5)` | Non-linear adjustment: `out = 255 * (img/255)^gamma`. Gamma < 1 brightens, > 1 darkens |
| 9 | **Histogram Equalization** | `aug_hist_equalization(img)` | Equalizes Y channel in YCbCr space for improved contrast |
| 10 | **Color Jitter** | `aug_color_jitter(img)` | Adjusts brightness (+40%), contrast (+50%), saturation (+60%), and hue shift (10°) |
| 11 | **Grayscale** | `aug_random_grayscale(img)` | Converts to grayscale while maintaining RGB format |

### Blur & Sharpening

| # | Technique | Function | Description |
|---|-----------|----------|-------------|
| 12 | **Gaussian Blur** | `aug_gaussian_blur(img)` | Applies Gaussian blur with radius=4.0 for strong smoothing |
| 13 | **Sharpening** | `aug_sharpen(img)` | High-pass filter sharpening: `sharp = img + 3.0 * (img - blur)` |
| 14 | **Motion Blur** | `aug_motion_blur(img, ksize=15)` | Simulates horizontal camera motion by averaging shifted frames |

### Noise Injection

| # | Technique | Function | Description |
|---|-----------|----------|-------------|
| 15 | **Gaussian Noise** | `aug_gaussian_noise(img, sigma=25.0)` | Adds random Gaussian noise sampled from N(0, sigma) |
| 16 | **Salt & Pepper** | `aug_salt_pepper(img, amount=0.03)` | Randomly sets 3% of pixels to pure white or black |
| 17 | **Speckle Noise** | `aug_speckle_noise(img, sigma=0.25)` | Multiplicative noise: `out = img + img * N(0, sigma)` |

### Compression Artifacts

| # | Technique | Function | Description |
|---|-----------|----------|-------------|
| 18 | **JPEG Artifacts** | `aug_jpeg_artifacts(img, quality=10)` | Simulates low-quality JPEG compression artifacts |

## Visual Examples

Run the main script to visualize all 18 techniques:

```bash
python -m venv venv
venv\Scripts\activate
python augmentation.py
```

This displays a 2x9 grid showing all augmentations applied to sample images.

## Parameters Reference

| Technique | Key Parameters | Effect |
|-----------|---------------|--------|
| Scale | `scale`: 1.0-2.0 | Higher = more zoom |
| Center Crop | `crop_ratio`: 0.3-0.9 | Lower = more crop |
| Translation | `shift_x`, `shift_y`: pixels | Shift amount |
| Rotation | `angle`: -180 to 180 | Rotation degrees |
| Shear | `shear`: 0.1-0.5 | Shear intensity |
| Brightness/Contrast | `alpha`, `beta` | alpha=contrast, beta=brightness |
| Gamma | `gamma`: 0.3-2.0 | < 1 brighter, > 1 darker |
| Gaussian Noise | `sigma`: 10-50 | Noise intensity |
| Salt & Pepper | `amount`: 0.01-0.1 | Noise density |
| Speckle | `sigma`: 0.1-0.5 | Noise intensity |
| JPEG | `quality`: 1-100 | Lower = more artifacts |
| Motion Blur | `ksize`: 5-25 | Blur length |

## License

MIT License
