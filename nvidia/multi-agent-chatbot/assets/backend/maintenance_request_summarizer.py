# maintenance_request_summarizer.py

import json
from typing import Dict, Any
from openai import AsyncOpenAI
from logger import logger


class MaintenanceRequestSummarizer:
    """Summarizes maintenance request JSON into structured summary."""

    def __init__(self, model_host: str = "gpt-oss-120b", model_port: int = 8000):
        self.model_client = AsyncOpenAI(
            base_url=f"http://{model_host}:{model_port}/v1",
            api_key="api_key"
        )
        self.model_name = model_host

    async def summarize_request(self, req: Dict[str, Any]) -> str:
        try:
            data = json.dumps(req, indent=2)

            prompt = f"""
Summarize the following maintenance request in 2–3 short paragraphs.

The summary should:
- Identify what broke or requires service
- Mention severity (if present)
- Describe symptoms or failure mode
- Highlight required actions, technicians, or spare parts
- State urgency, deadlines, or scheduling requirements

MAINTENANCE REQUEST:
{data}

Provide only the summary:
"""

            response = await self.model_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You summarize maintenance requests clearly and professionally."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
                temperature=0.3,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Maintenance request summarization failed: {e}", exc_info=True)
            return f"Maintenance Request Summary (fallback): {json.dumps(req, indent=2)}"
