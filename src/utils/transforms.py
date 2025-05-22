# src/utils/transforms.py

import random
from typing import Tuple, Union, List

import numpy as np
from PIL import Image, ImageEnhance


class Compose:
    """
    Compose several transforms together.
    """
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms

    def __call__(self, img: Union[np.ndarray, Image.Image]) -> np.ndarray:
        for t in self.transforms:
            img = t(img)
        return img


class Resize:
    """
    Resize an image to the given size.
    """
    def __init__(self, size: Tuple[int, int]):
        self.size = size  # (width, height)

    def __call__(self, img: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(img, np.ndarray):
            pil = Image.fromarray((img * 255).astype(np.uint8))
        else:
            pil = img
        pil = pil.resize(self.size, Image.BILINEAR)
        return np.asarray(pil).astype(np.float32) / 255.0


class RandomCrop:
    """
    Randomly crop a region of the image.
    """
    def __init__(self, crop_size: Tuple[int, int]):
        self.crop_w, self.crop_h = crop_size

    def __call__(self, img: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(img, Image.Image):
            w, h = img.size
        else:
            h, w = img.shape[:2]
            img = Image.fromarray((img * 255).astype(np.uint8))

        if w == self.crop_w and h == self.crop_h:
            return np.asarray(img).astype(np.float32) / 255.0

        left = random.randint(0, w - self.crop_w)
        top = random.randint(0, h - self.crop_h)
        cropped = img.crop((left, top, left + self.crop_w, top + self.crop_h))
        return np.asarray(cropped).astype(np.float32) / 255.0


class RandomHorizontalFlip:
    """
    Horizontally flip the image with a given probability.
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if random.random() < self.p:
            if isinstance(img, np.ndarray):
                return np.fliplr(img).copy()
            else:
                return img.transpose(Image.FLIP_LEFT_RIGHT)
        return np.asarray(img).astype(np.float32) / 255.0 if isinstance(img, np.ndarray) else np.asarray(img).astype(np.float32) / 255.0


class ColorJitter:
    """
    Randomly change brightness, contrast, saturation, and hue.
    """
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(img, np.ndarray):
            img = Image.fromarray((img * 255).astype(np.uint8))

        # Brightness
        if self.brightness > 0:
            factor = 1 + random.uniform(-self.brightness, self.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)

        # Contrast
        if self.contrast > 0:
            factor = 1 + random.uniform(-self.contrast, self.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)

        # Saturation
        if self.saturation > 0:
            factor = 1 + random.uniform(-self.saturation, self.saturation)
            img = ImageEnhance.Color(img).enhance(factor)

        # Hue (convert to HSV and back)
        if self.hue > 0:
            hsv = np.array(img.convert("HSV"), dtype=np.uint8)
            h = hsv[..., 0].astype(int)
            shift = int(random.uniform(-self.hue * 255, self.hue * 255))
            hsv[..., 0] = ((h + shift) % 255).astype(np.uint8)
            img = Image.fromarray(hsv, mode="HSV").convert("RGB")

        return np.asarray(img).astype(np.float32) / 255.0


class Normalize:
    """
    Normalize a float‐valued image by mean/std per channel.
    """
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return (img - self.mean) / self.std
