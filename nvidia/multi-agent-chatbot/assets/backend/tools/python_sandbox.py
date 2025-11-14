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
It supports multiple image input formats including file paths and base64-encoded images.
"""
import base64
import io
import os
import sys
from pathlib import Path
from typing import List, Optional

from mcp.server.fastmcp import FastMCP
from PIL import Image, ImageDraw, ImageFont

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

mcp = FastMCP("image-annotation-server")


@mcp.tool()
def annotate_image(
    image: str,
    bounding_boxes: List[List[float]],
    color: Optional[str] = "red",
    tags: Optional[List[str]] = None
) -> str:
    """
    Annotate an image with bounding boxes and optional tags.

    Args:
        image: Image input as either a file path or base64-encoded data URL
        bounding_boxes: List of bounding boxes, where each box is [x1, y1, x2, y2]
                       (top-left and bottom-right coordinates)
        color: Color for the bounding boxes (default: "red"). Accepts color names or hex codes.
        tags: Optional list of text labels for each bounding box

    Returns:
        Base64-encoded data URL of the annotated image

    Example:
        annotate_image(
            image="/path/to/image.jpg",
            bounding_boxes=[[10, 10, 100, 100], [150, 150, 250, 250]],
            color="green",
            tags=["Object 1", "Object 2"]
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

    # Load the image
    img = None
    if image.startswith("data:image/"):
        # Handle base64-encoded data URL
        _, b64_data = image.split(",", 1)
        image_data = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(image_data))
    elif os.path.exists(image):
        # Handle file path
        img = Image.open(image)
    else:
        raise ValueError(f'Invalid image type -- could not be identified as a data URL or filepath: {image}')

    # Convert to RGB if necessary (for transparency handling)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Create drawing context
    draw = ImageDraw.Draw(img)

    # Try to load a font for tags, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

    # Draw bounding boxes and tags
    for i, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = box

        # Draw rectangle with 3px width
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw tag if provided
        if tags and i < len(tags):
            tag_text = tags[i]
            # Get text bounding box for background
            try:
                bbox = draw.textbbox((x1, y1 - 20), tag_text, font=font)
                # Draw background rectangle for text
                draw.rectangle(bbox, fill=color)
                # Draw text in white
                draw.text((x1, y1 - 20), tag_text, fill="white", font=font)
            except:
                # Fallback for older PIL versions
                draw.text((x1, y1 - 20), tag_text, fill=color, font=font)

    # Convert annotated image to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Return as data URL
    return f"data:image/png;base64,{img_str}"


if __name__ == "__main__":
    print(f'running {mcp.name} MCP server')
    mcp.run(transport="stdio")
