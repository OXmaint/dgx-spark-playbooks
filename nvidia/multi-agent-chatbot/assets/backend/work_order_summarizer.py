#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Work Order Summarization Service"""

import json
from typing import Dict, Any
from openai import AsyncOpenAI
from logger import logger


class WorkOrderSummarizer:
    """Service for summarizing work orders using local LLM"""
    
    def __init__(self, model_host: str = "gpt-oss-120b", model_port: int = 8000):
        """Initialize the summarizer with LLM connection.
        
        Args:
            model_host: Host name of the LLM service
            model_port: Port of the LLM service
        """
        self.model_client = AsyncOpenAI(
            base_url=f"http://{model_host}:{model_port}/v1",
            api_key="api_key"
        )
        self.model_name = model_host
        
    async def summarize_work_order(self, work_order: Dict[str, Any]) -> str:
        """Generate a comprehensive summary of a work order.
        
        Args:
            work_order: Work order JSON payload
            
        Returns:
            Summary text
        """
        try:
            work_order_text = json.dumps(work_order, indent=2)
            
            prompt = f"""Analyze the following work order and provide a comprehensive, well-structured summary.

The summary should:
- Be 2-3 paragraphs
- Highlight key details like work type, priority, location, requirements
- Mention important dates, costs, or resources if present
- Be clear and informative for quick understanding

Work Order:
{work_order_text}

Provide only the summary, no additional commentary:"""

            response = await self.model_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an AI that creates clear, concise summaries of work orders. Provide only the summary without any preamble or additional text."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            summary = response.choices[0].message.content.strip()
            
            logger.info({
                "message": "Work order summarized successfully",
                "summary_length": len(summary)
            })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing work order: {e}", exc_info=True)
            # Fallback: create basic summary
            return f"Work Order Summary: {json.dumps(work_order, indent=2)}"