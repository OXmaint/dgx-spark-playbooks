
"""Work Order Summarization Service"""

import json
from typing import Dict, Any, Tuple
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
        
    async def summarize_and_extract_metadata(self, work_order: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
        """Generate a summary and extract key metadata from a work order.
        
        Args:
            work_order: Work order JSON payload
            
        Returns:
            Tuple of (summary text, metadata dict)
        """
        try:
            work_order_text = json.dumps(work_order, indent=2)
            
            prompt = f"""Analyze the following work order and provide:
1. A comprehensive summary (2-3 paragraphs)
2. Key metadata fields that would be useful for searching and categorizing this work order

Work Order:
{work_order_text}

Respond ONLY with a JSON object in this exact format:
{{
  "summary": "Your detailed summary here...",
  "metadata": {{
    "key1": "value1",
    "key2": "value2"
  }}
}}

For metadata, extract the most important fields like:
- Identifiers (IDs, reference numbers)
- Categories (type, category, department)
- Status/State information
- Priority/Urgency levels
- Dates (creation, due date, etc.)
- People/Teams (assignee, requester, etc.)
- Location information if present

Only include metadata fields that are actually present in the work order. Keep values concise."""

            response = await self.model_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an AI that analyzes work orders and extracts key information. Always respond with valid JSON only, no additional text."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON, handle potential markdown code blocks
            if response_text.startswith("```"):
                # Remove markdown code blocks
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(response_text)
            
            summary = result.get("summary", "")
            metadata = result.get("metadata", {})
            
            # Ensure metadata values are strings and clean them
            clean_metadata = {
                str(k): str(v)[:500]  # Limit metadata value length
                for k, v in metadata.items() 
                if v is not None and str(v).strip()
            }
            
            # Add source and original data
            clean_metadata["source"] = "work_orders"
            clean_metadata["raw_work_order"] = json.dumps(work_order)[:5000]  # Limit size
            
            logger.info({
                "message": "Work order analyzed successfully",
                "summary_length": len(summary),
                "metadata_fields": list(clean_metadata.keys())
            })
            
            return summary, clean_metadata
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response_text}")
            # Fallback: create basic summary
            return self._create_fallback_summary(work_order)
            
        except Exception as e:
            logger.error(f"Error analyzing work order: {e}", exc_info=True)
            raise
    
    def _create_fallback_summary(self, work_order: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
        """Create a basic summary if LLM parsing fails."""
        summary = f"Work Order: {json.dumps(work_order, indent=2)}"
        metadata = {
            "source": "work_orders",
            "raw_work_order": json.dumps(work_order)[:5000]
        }
        
        # Extract any obvious ID fields
        for key in ["id", "work_order_id", "workOrderId", "wo_id", "ticket_id"]:
            if key in work_order:
                metadata["id"] = str(work_order[key])
                break
        
        return summary, metadata