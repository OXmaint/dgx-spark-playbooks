# inspection_summarizer.py

import json
from typing import Dict, Any
from openai import AsyncOpenAI
from logger import logger


class InspectionSummarizer:
    """Summarizes inspection JSON payload into a structured text summary."""

    def __init__(self, model_host: str = "gpt-oss-120b", model_port: int = 8000):
        self.model_client = AsyncOpenAI(
            base_url=f"http://{model_host}:{model_port}/v1",
            api_key="api_key"
        )
        self.model_name = model_host

    async def summarize_inspection(self, inspection: Dict[str, Any]) -> str:
        try:
            data = json.dumps(inspection, indent=2)

            prompt = f"""
Analyze the following inspection record and summarize it in a clear, structured way.

Your summary must:
- Identify the asset or location inspected
- Summarize key findings (pass/fail, defects, issues)
- Highlight safety or compliance concerns
- Mention any recommended actions and deadlines

INSPECTION RECORD:
{data}

Provide ONLY the summary:
"""

            response = await self.model_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You summarize inspection records in concise form."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )

            summary = response.choices[0].message.content.strip()
            return summary

        except Exception as e:
            logger.error(f"Inspection summarization failed: {e}", exc_info=True)
            return f"Inspection Summary (fallback): {json.dumps(inspection, indent=2)}"
