# Created: 2026-02-01
# Modified: 2026-03-20
# Purpose: Prepare local images for API requests by resizing and encoding them safely.

import io

from PIL import Image


IMAGE_FORMAT_MAP = {
    "png": "png",
    "jpg": "jpeg",
    "jpeg": "jpeg",
    "PNG": "png",
    "JPG": "jpeg",
    "JPEG": "jpeg",
    "webp": "webp",
}


def get_image_size_mb(image, image_format):
    byte_buffer = io.BytesIO()
    image.save(byte_buffer, format=image_format)
    return byte_buffer.tell() / 1024 / 1024


def resize_by_resolution(image, max_resolution):
    width, height = image.size
    if max_resolution is None or (width <= max_resolution and height <= max_resolution):
        return image

    scale_factor = max(width, height) / max_resolution
    new_width = max(1, int(width / scale_factor))
    new_height = max(1, int(height / scale_factor))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def resize_by_file_size(image, image_format, max_size_mb):
    file_size_mb = get_image_size_mb(image, image_format)
    while file_size_mb > max_size_mb:
        scale_factor = (file_size_mb / max_size_mb) ** 0.5
        width, height = image.size
        new_width = max(1, int(width / scale_factor))
        new_height = max(1, int(height / scale_factor))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        file_size_mb = get_image_size_mb(image, image_format)
    return image, file_size_mb


def image_to_bytes(image, image_format):
    byte_buffer = io.BytesIO()
    image.save(byte_buffer, format=image_format)
    byte_buffer.seek(0)
    return byte_buffer


def get_image(image_path, max_size_mb, max_resolution=None):
    try:
        with Image.open(image_path) as raw_image:
            original_format = raw_image.format or image_path.split(".")[-1]
            image = raw_image.convert("RGB")
    except Exception as exc:
        print(f"Failed to read image `{image_path}`: {exc}")
        return None, None, None

    if original_format not in IMAGE_FORMAT_MAP:
        print(f"Unsupported image type: {original_format}")
        return None, None, None

    image_format = IMAGE_FORMAT_MAP[original_format]
    image = resize_by_resolution(image, max_resolution)
    image, file_size_mb = resize_by_file_size(image, image_format, max_size_mb)
    return image_to_bytes(image, image_format), image_format, file_size_mb
