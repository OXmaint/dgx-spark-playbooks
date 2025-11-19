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
MCP server providing image understanding and analysis tools.

This server exposes a `process_image` tool that uses a vision language model to answer queries about images. 
It supports multiple image input formats including URLs, file paths, and base64-encoded images.
"""
import asyncio
import base64
import os
import requests
import sys
from pathlib import Path
import time

from langchain_core.tools import tool, Tool
from langchain_mcp_adapters.tools import to_fastmcp
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI, OpenAI

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from postgres_storage import PostgreSQLConversationStorage


mcp = FastMCP("image-understanding-server")


model_name = "qwen3-vl"
model_client = OpenAI(
    base_url=f"http://qwen3-vl:8000/v1",
    api_key="api_key"
)
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "chatbot")
POSTGRES_USER = os.getenv("POSTGRES_USER", "chatbot_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "chatbot_password")

postgres_storage = PostgreSQLConversationStorage(
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    database=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD
)

@mcp.tool()
def explain_image(query: str, image: str, return_bounding_boxes: bool = False):
    """
    This tool is used to understand an image. It will respond to the user's query based on the image.

    Args:
        query: The question or analysis request about the image
        image: Image input as either:
               - A file path (e.g., "/path/to/image.jpg")
               - A base64-encoded data URL (e.g., "data:image/jpeg;base64,...")
               - A URL to fetch the image from (e.g., "https://example.com/image.jpg")
        return_bounding_boxes: If True, the response will include bounding box coordinates in JSON format
                              for detected objects/defects. Format: {"objects": [{"label": "name", "bbox": [x1, y1, x2, y2]}]}

    Returns:
        Text description of the image, or JSON with bounding boxes if return_bounding_boxes=True
    """
    if not image:
        raise ValueError('Error: explain_image tool received an empty image string.')

    image_url_content = {}

    if image.startswith("http://") or image.startswith("https://"):
        image_url_content = {
            "type": "image_url",
            "image_url": {"url": image}
        }
    else:
        if image.startswith("data:image/"):
            metadata, b64_data = image.split(",", 1)
            filetype = metadata.split(";")[0].split("/")[-1]
        elif os.path.exists(image):
            with open(image, "rb") as image_file:
                filetype = image.split('.')[-1]
                b64_data = base64.b64encode(image_file.read()).decode("utf-8")
        else:
            raise ValueError(f'Invalid image type -- could not be identified as a url or filepath: {image}')
        
        image_url_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{filetype if filetype else 'jpeg'};base64,{b64_data}"
            }
        }

    # Modify query if bounding boxes are requested
    if return_bounding_boxes:
        # Use Qwen VL's native grounding format with 0-1000 coordinate range
        bbox_query = f"""Find all defects in this image with grounding. {query}

Return a JSON object with bounding boxes in the coordinate range [0, 1000].

Format: bbox = [x1, y1, x2, y2] where all values are integers from 0 to 1000:
- (x1, y1) = top-left corner (0,0 is top-left of image)
- (x2, y2) = bottom-right corner (1000,1000 is bottom-right of image)

JSON format:
{{
    "description": "Brief description",
    "objects": [
        {{"label": "name", "bbox": [x1, y1, x2, y2]}}
    ]
}}

Example - rust in upper-right, scratch in lower-center:
{{
    "description": "Ship hull with damage",
    "objects": [
        {{"label": "Rust", "bbox": [700, 100, 900, 300]}},
        {{"label": "Scratch", "bbox": [400, 700, 600, 850]}}
    ]
}}

Rules:
- Use SMALL, TIGHT boxes around specific defects only
- Coordinates must be integers 0-1000
- No overlapping boxes
- Only JSON output"""
        actual_query = bbox_query
    else:
        actual_query = query

    message = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": actual_query},
                image_url_content
            ]
        }
    ]

    try:
        print(f"Sending request to vision model: {actual_query[:100]}...")
        response = model_client.chat.completions.create(
            model=model_name,
            messages=message,
            max_tokens=1024 if return_bounding_boxes else 512,
            temperature=0.1
        )
        print(f"Received response from vision model")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling vision model: {e}")
        raise RuntimeError(f"Failed to process image with vision model: {e}")

if __name__ == "__main__":
    print(f'running {mcp.name} MCP server')
    mcp.run(transport="stdio")