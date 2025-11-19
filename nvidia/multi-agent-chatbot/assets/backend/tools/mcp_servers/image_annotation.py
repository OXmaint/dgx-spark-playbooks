#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
MCP server providing image annotation tools.

This server exposes an `annotate_image` tool that draws bounding boxes with optional tags on images.
It supports multiple image input formats including file paths, base64-encoded images, and URLs.
"""
import base64
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from mcp.server.fastmcp import FastMCP

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

mcp = FastMCP("image-annotation-server")


def _color_name_to_bgr(color_name: str) -> tuple:
    """Convert color name or hex code to BGR tuple for OpenCV.

    Args:
        color_name: Color name (e.g., "red", "green") or hex code (e.g., "#FF0000")

    Returns:
        BGR tuple for OpenCV
    """
    color_map = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "orange": (0, 165, 255),
        "purple": (128, 0, 128),
        "pink": (203, 192, 255),
    }

    color_lower = color_name.lower().strip()

    if color_lower in color_map:
        return color_map[color_lower]

    # Handle hex codes
    if color_lower.startswith("#"):
        hex_color = color_lower[1:]
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (b, g, r)  # BGR for OpenCV

    # Default to red if unknown
    return (0, 0, 255)


@mcp.tool()
def annotate_image(
    image: str,
    bounding_boxes: List[List[float]],
    color: Optional[str] = "red",
    tags: Optional[List[str]] = None,
    coord_format: Optional[str] = "qwen"
) -> str:
    """
    Annotate an image with bounding boxes and optional tags using OpenCV.

    Args:
        image: Image input as either:
               - A file path (e.g., "/path/to/image.jpg")
               - A base64-encoded data URL (e.g., "data:image/jpeg;base64,...")
               - A URL to fetch the image from (e.g., "https://example.com/image.jpg")
        bounding_boxes: List of bounding boxes, where each box is [x1, y1, x2, y2].
                       (x1, y1) is top-left corner, (x2, y2) is bottom-right corner.
        color: Color for the bounding boxes (default: "red").
               Accepts color names: "red", "green", "blue", "yellow", "cyan", "magenta",
               "white", "black", "orange", "purple", "pink"
               Or hex codes: "#FF0000", "#00FF00", etc.
        tags: Optional list of text labels for each bounding box.
              The tag at index i will be drawn above bounding_boxes[i].
        coord_format: Coordinate format for bounding boxes (default: "qwen").
                     - "qwen": Qwen VL format with coordinates in [0, 1000] range
                     - "normalized": Normalized coordinates in [0, 1] range
                     - "pixel": Direct pixel coordinates

    Returns:
        Base64-encoded data URL of the annotated image (PNG format)

    Example:
        # Annotate with Qwen VL format [0, 1000] coordinates (default)
        annotate_image(
            image="/path/to/image.jpg",
            bounding_boxes=[[100, 200, 300, 500], [600, 100, 800, 400]],
            color="green",
            tags=["Person", "Car"]
        )

        # Annotate with pixel coordinates
        annotate_image(
            image="https://example.com/photo.jpg",
            bounding_boxes=[[50, 100, 250, 500]],
            color="#FF5500",
            tags=["Detected Object"],
            coord_format="pixel"
        )
    """
    if not image:
        raise ValueError('Error: annotate_image tool received an empty image string.')

    if not bounding_boxes:
        raise ValueError('Error: annotate_image tool requires at least one bounding box.')

    # Validate bounding boxes format
    for i, box in enumerate(bounding_boxes):
        if len(box) != 4:
            raise ValueError(f'Error: Bounding box {i} must have exactly 4 coordinates [x1, y1, x2, y2], got {len(box)}')

    # Load the image into OpenCV format (numpy array)
    img = None

    if image.startswith("data:image/"):
        # Handle base64-encoded data URL
        _, b64_data = image.split(",", 1)
        image_data = base64.b64decode(b64_data)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    elif image.startswith("http://") or image.startswith("https://"):
        # Handle URL - fetch the image
        try:
            req = urllib.request.Request(
                image,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                image_data = response.read()
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except urllib.error.URLError as e:
            raise ValueError(f'Error fetching image from URL: {e}')
        except Exception as e:
            raise ValueError(f'Error processing image from URL: {e}')

    elif os.path.exists(image):
        # Handle file path
        img = cv2.imread(image)

    else:
        raise ValueError(f'Invalid image input -- could not be identified as a data URL, URL, or filepath: {image}')

    if img is None:
        raise ValueError('Error: Failed to load/decode the image')

    # Get image dimensions for coordinate conversion
    img_height, img_width = img.shape[:2]
    print(f"[IMAGE_ANNOTATION] Image dimensions: {img_width}x{img_height}", flush=True)

    # Convert BGR color
    bgr_color = _color_name_to_bgr(color)

    # Draw bounding boxes and tags
    for i, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = box

        # Convert coordinates to pixels based on specified format
        if coord_format == "normalized":
            # Normalized coordinates (0.0 to 1.0)
            x1 = int(x1 * img_width)
            y1 = int(y1 * img_height)
            x2 = int(x2 * img_width)
            y2 = int(y2 * img_height)
            print(f"[IMAGE_ANNOTATION] Box {i}: Normalized [0-1] {box} -> Pixels [{x1}, {y1}, {x2}, {y2}]", flush=True)
        elif coord_format == "pixel":
            # Already pixel coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(f"[IMAGE_ANNOTATION] Box {i}: Pixel coordinates [{x1}, {y1}, {x2}, {y2}]", flush=True)
        else:
            # Default: Qwen VL format - coordinates in 0-1000 range
            x1 = int((x1 / 1000.0) * img_width)
            y1 = int((y1 / 1000.0) * img_height)
            x2 = int((x2 / 1000.0) * img_width)
            y2 = int((y2 / 1000.0) * img_height)
            print(f"[IMAGE_ANNOTATION] Box {i}: Qwen [0-1000] {box} -> Pixels [{x1}, {y1}, {x2}, {y2}]", flush=True)

        # Draw rectangle with thickness 3
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr_color, thickness=3)

        # Draw tag if provided
        if tags and i < len(tags):
            tag_text = tags[i]

            # Font settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2

            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                tag_text, font, font_scale, font_thickness
            )

            # Calculate background rectangle position (above the bounding box)
            bg_x1 = x1
            bg_y1 = max(0, y1 - text_height - baseline - 8)
            bg_x2 = x1 + text_width + 4
            bg_y2 = y1

            # Draw background rectangle for text
            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), bgr_color, -1)

            # Draw text in white
            text_x = x1 + 2
            text_y = y1 - baseline - 4
            cv2.putText(img, tag_text, (text_x, text_y), font, font_scale,
                       (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Convert annotated image to base64
    success, encoded_img = cv2.imencode('.png', img)
    if not success:
        raise ValueError('Error: Failed to encode annotated image')

    img_str = base64.b64encode(encoded_img.tobytes()).decode("utf-8")
    result = f"data:image/png;base64,{img_str}"

    # Log the result for debugging
    print(f"[IMAGE_ANNOTATION] Successfully created annotated image", flush=True)
    print(f"[IMAGE_ANNOTATION] Result length: {len(result)} characters", flush=True)
    print(f"[IMAGE_ANNOTATION] Result starts with: {result[:50]}...", flush=True)

    # Return as data URL
    return result


if __name__ == "__main__":
    print(f'running {mcp.name} MCP server')
    mcp.run(transport="stdio")
