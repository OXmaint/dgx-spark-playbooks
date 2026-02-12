# pm_schedule_summarizer.py

import json
from typing import Dict, Any
from openai import AsyncOpenAI
from logger import logger


class PmScheduleSummarizer:
    """Summarizes preventive maintenance schedule JSON into structured summary."""

    def __init__(self, model_host: str = "gpt-oss-120b", model_port: int = 8000):
        self.model_client = AsyncOpenAI(
            base_url=f"http://{model_host}:{model_port}/v1",
            api_key="api_key"
        )

        self.model_name = model_host

    async def summarize_pm(self, pm: Dict[str, Any]) -> str:
        try:
            data = json.dumps(pm, indent=2)

            prompt = f"""
Summarize the following preventive maintenance schedule.

Your summary should:
- Identify the asset and maintenance task
- Mention recurrence frequency and due dates
- Highlight required parts/tools
- State estimated time and any safety considerations
- Note any special instructions

PM SCHEDULE:
{data}

Provide only the summary:
"""

            response = await self.model_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You summarize preventive maintenance schedules clearly."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"PM schedule summarization failed: {e}", exc_info=True)
            return f"PM Schedule Summary (fallback): {json.dumps(pm, indent=2)}"
