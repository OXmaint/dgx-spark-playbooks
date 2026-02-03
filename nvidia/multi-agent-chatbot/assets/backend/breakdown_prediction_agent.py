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
"""Breakdown Prediction Agent for predictive maintenance analysis.

This agent analyzes asset data to predict potential unplanned breakdowns:
1. Retrieves historical work orders, inspections, and service schedules from vector DB
2. Processes live sensor data
3. Uses LLM to generate comprehensive breakdown predictions with risk scores
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI

from logger import logger
from postgres_storage import PostgreSQLConversationStorage
from prompts import Prompts


class BreakdownPredictionAgent:
    """Agent for predictive maintenance breakdown analysis.

    This agent processes multiple data sources to predict potential equipment failures:
    1. Asset information (metadata, specifications, criticality)
    2. Historical work orders from vector DB
    3. Past inspection reports from vector DB
    4. Service schedules from vector DB
    5. Live sensor data (temperature, vibration, pressure, etc.)
    """

    def __init__(
        self,
        vector_store,
        config_manager,
        postgres_storage: PostgreSQLConversationStorage
    ):
        """Initialize the breakdown prediction agent.

        Args:
            vector_store: VectorStore instance for document retrieval
            config_manager: ConfigManager for reading configuration
            postgres_storage: PostgreSQL storage for persistence
        """
        self.vector_store = vector_store
        self.config_manager = config_manager
        self.storage = postgres_storage
        self.model_client = None
        self.current_model = None
        self.prompts = Prompts()

    async def init(self) -> None:
        """Initialize the agent with model configuration."""
        model_name = self.config_manager.get_selected_model()
        self.set_current_model(model_name)
        logger.info("BreakdownPredictionAgent initialized successfully")

    def set_current_model(self, model_name: str) -> None:
        """Set the current model for completions.

        Args:
            model_name: Name of the model to use
        """
        available_models = self.config_manager.get_available_models()

        if model_name in available_models:
            self.current_model = model_name
            self.model_client = AsyncOpenAI(
                base_url=f"http://{self.current_model}:8000/v1",
                api_key="api_key"
            )
            logger.info(f"BreakdownPredictionAgent using model: {model_name}")
        else:
            raise ValueError(f"Model {model_name} is not available. Available: {available_models}")

    async def retrieve_work_orders(
        self,
        asset_id: str,
        collection_name: Optional[str] = None
    ) -> str:
        """Retrieve recent work orders from vector DB filtered by asset_id and doc_type.

        Args:
            asset_id: Asset identifier for filtering
            collection_name: Optional collection to search in

        Returns:
            Formatted string of recent work orders
        """
        try:
            # Filter by doc_type = "work_order" and asset_id
            metadata_filter = {
                "doc_type": "work_order",
                "asset_id": asset_id
            }

            # Get recent 5 work orders without semantic search (limited to reduce prompt size)
            results = self.vector_store.get_recent_documents(
                k=5,
                collection_name=collection_name,
                metadata_filter=metadata_filter
            )

            if results:
                work_orders = []
                for i, doc in enumerate(results, 1):
                    # Include metadata in output for traceability
                    metadata_str = ""
                    if hasattr(doc, 'metadata') and doc.metadata:
                        wo_id = doc.metadata.get('work_order_id', doc.metadata.get('id', f'WO-{i}'))
                        date = doc.metadata.get('date', doc.metadata.get('created_at', 'N/A'))
                        issue_type = doc.metadata.get('issue_type', doc.metadata.get('type', 'N/A'))
                        metadata_str = f"\n  Work Order ID: {wo_id}\n  Date: {date}\n  Issue Type: {issue_type}"
                    work_orders.append(f"[Work Order {i}]:{metadata_str}\n{doc.page_content}")
                return "\n\n".join(work_orders)
            else:
                return "No historical work orders found in the knowledge base for this asset."
        except Exception as e:
            logger.warning(f"Error retrieving work orders: {e}")
            return f"Error retrieving work orders: {str(e)}"

    async def retrieve_inspections(
        self,
        asset_id: str,
        collection_name: Optional[str] = None
    ) -> str:
        """Retrieve recent inspections from vector DB filtered by asset_id and doc_type.

        Args:
            asset_id: Asset identifier for filtering
            collection_name: Optional collection to search in

        Returns:
            Formatted string of recent inspection reports
        """
        try:
            # Filter by doc_type = "inspection" and asset_id
            metadata_filter = {
                "doc_type": "inspection",
                "asset_id": asset_id
            }

            # Get recent 5 inspections without semantic search (limited to reduce prompt size)
            results = self.vector_store.get_recent_documents(
                k=5,
                collection_name=collection_name,
                metadata_filter=metadata_filter
            )

            if results:
                inspections = []
                for i, doc in enumerate(results, 1):
                    # Include metadata in output for traceability
                    metadata_str = ""
                    if hasattr(doc, 'metadata') and doc.metadata:
                        insp_id = doc.metadata.get('inspection_id', doc.metadata.get('id', f'INSP-{i}'))
                        date = doc.metadata.get('date', doc.metadata.get('inspection_date', 'N/A'))
                        insp_type = doc.metadata.get('inspection_type', 'N/A')
                        condition = doc.metadata.get('condition_rating', doc.metadata.get('condition', 'N/A'))
                        metadata_str = f"\n  Inspection ID: {insp_id}\n  Date: {date}\n  Type: {insp_type}\n  Condition Rating: {condition}"
                    inspections.append(f"[Inspection Report {i}]:{metadata_str}\n{doc.page_content}")
                return "\n\n".join(inspections)
            else:
                return "No historical inspection reports found in the knowledge base for this asset."
        except Exception as e:
            logger.warning(f"Error retrieving inspections: {e}")
            return f"Error retrieving inspections: {str(e)}"

    async def retrieve_service_schedules(
        self,
        asset_id: str,
        collection_name: Optional[str] = None
    ) -> str:
        """Retrieve recent service schedules from vector DB filtered by asset_id and doc_type.

        Args:
            asset_id: Asset identifier for filtering
            collection_name: Optional collection to search in

        Returns:
            Formatted string of recent service schedules
        """
        try:
            # Filter by doc_type = "service_schedule" and asset_id
            metadata_filter = {
                "doc_type": "service_schedule",
                "asset_id": asset_id
            }

            # Get recent 5 service schedules without semantic search (limited to reduce prompt size)
            results = self.vector_store.get_recent_documents(
                k=5,
                collection_name=collection_name,
                metadata_filter=metadata_filter
            )

            if results:
                schedules = []
                for i, doc in enumerate(results, 1):
                    # Include metadata in output for traceability
                    metadata_str = ""
                    if hasattr(doc, 'metadata') and doc.metadata:
                        sched_id = doc.metadata.get('schedule_id', doc.metadata.get('id', f'SCHED-{i}'))
                        frequency = doc.metadata.get('frequency', 'N/A')
                        last_performed = doc.metadata.get('last_performed', 'N/A')
                        next_due = doc.metadata.get('next_due', 'N/A')
                        maintenance_type = doc.metadata.get('maintenance_type', 'N/A')
                        metadata_str = f"\n  Schedule ID: {sched_id}\n  Maintenance Type: {maintenance_type}\n  Frequency: {frequency}\n  Last Performed: {last_performed}\n  Next Due: {next_due}"
                    schedules.append(f"[Service Schedule {i}]:{metadata_str}\n{doc.page_content}")
                return "\n\n".join(schedules)
            else:
                return "No service schedules found in the knowledge base for this asset."
        except Exception as e:
            logger.warning(f"Error retrieving service schedules: {e}")
            return f"Error retrieving service schedules: {str(e)}"

    async def retrieve_maintenance_requests(
        self,
        asset_id: str,
        collection_name: Optional[str] = None
    ) -> str:
        """Retrieve recent maintenance requests from vector DB filtered by asset_id and doc_type.

        Args:
            asset_id: Asset identifier for filtering
            collection_name: Optional collection to search in

        Returns:
            Formatted string of recent maintenance requests
        """
        try:
            # Filter by doc_type = "maintenance_request" and asset_id
            metadata_filter = {
                "doc_type": "maintenance_request",
                "asset_id": asset_id
            }

            # Get recent 5 maintenance requests without semantic search (limited to reduce prompt size)
            results = self.vector_store.get_recent_documents(
                k=5,
                collection_name=collection_name,
                metadata_filter=metadata_filter
            )

            if results:
                requests = []
                for i, doc in enumerate(results, 1):
                    # Include metadata in output for traceability
                    metadata_str = ""
                    if hasattr(doc, 'metadata') and doc.metadata:
                        req_id = doc.metadata.get('request_id', doc.metadata.get('id', f'MR-{i}'))
                        date = doc.metadata.get('date', doc.metadata.get('created_at', 'N/A'))
                        priority = doc.metadata.get('priority', 'N/A')
                        status = doc.metadata.get('status', 'N/A')
                        metadata_str = f"\n  Request ID: {req_id}\n  Date: {date}\n  Priority: {priority}\n  Status: {status}"
                    requests.append(f"[Maintenance Request {i}]:{metadata_str}\n{doc.page_content}")
                return "\n\n".join(requests)
            else:
                return "No maintenance requests found in the knowledge base for this asset."
        except Exception as e:
            logger.warning(f"Error retrieving maintenance requests: {e}")
            return f"Error retrieving maintenance requests: {str(e)}"

    async def retrieve_knowledge_base(
        self,
        collection_name: Optional[str] = None
    ) -> str:
        """Retrieve recent knowledge base articles filtered by doc_type.

        This retrieves general knowledge documents (manuals, troubleshooting guides,
        best practices).

        Args:
            collection_name: Optional collection to search in

        Returns:
            Formatted string of recent knowledge articles
        """
        try:
            # Filter by doc_type = "knowledge"
            metadata_filter = {
                "doc_type": "knowledge"
            }

            # Get recent 3 knowledge base documents without semantic search (limited to reduce prompt size)
            results = self.vector_store.get_recent_documents(
                k=3,
                collection_name=collection_name,
                metadata_filter=metadata_filter
            )

            if results:
                articles = []
                for i, doc in enumerate(results, 1):
                    # Include metadata in output for traceability
                    metadata_str = ""
                    if hasattr(doc, 'metadata') and doc.metadata:
                        doc_id = doc.metadata.get('document_id', doc.metadata.get('id', f'KB-{i}'))
                        title = doc.metadata.get('title', 'N/A')
                        category = doc.metadata.get('category', 'N/A')
                        metadata_str = f"\n  Document ID: {doc_id}\n  Title: {title}\n  Category: {category}"
                    articles.append(f"[Knowledge Base {i}]:{metadata_str}\n{doc.page_content}")
                return "\n\n".join(articles)
            else:
                return "No relevant knowledge base articles found."
        except Exception as e:
            logger.warning(f"Error retrieving knowledge base: {e}")
            return f"Error retrieving knowledge base: {str(e)}"

    def format_sensor_data(self, sensor_data: List[Dict[str, Any]]) -> str:
        """Format sensor data for the prompt.

        Args:
            sensor_data: List of sensor readings

        Returns:
            Formatted string of sensor data
        """
        if not sensor_data:
            return "No live sensor data available."

        formatted_lines = []
        for sensor in sensor_data:
            sensor_name = sensor.get("name", "Unknown Sensor")
            value = sensor.get("value", "N/A")
            unit = sensor.get("unit", "")
            timestamp = sensor.get("timestamp", "N/A")
            status = sensor.get("status", "normal")
            threshold_min = sensor.get("threshold_min")
            threshold_max = sensor.get("threshold_max")

            line = f"- {sensor_name}: {value} {unit}"
            if timestamp != "N/A":
                line += f" (at {timestamp})"
            if status != "normal":
                line += f" [STATUS: {status.upper()}]"
            if threshold_min is not None or threshold_max is not None:
                line += f" [Thresholds: {threshold_min or 'N/A'} - {threshold_max or 'N/A'}]"

            formatted_lines.append(line)

        return "\n".join(formatted_lines)

    def format_asset_info(self, asset_info: Dict[str, Any]) -> str:
        """Format asset information for the prompt.

        Args:
            asset_info: Asset metadata and specifications

        Returns:
            Formatted string of asset information
        """
        if not asset_info:
            return "No asset information provided."

        formatted_lines = []
        priority_fields = ["asset_id", "asset_name", "asset_type", "manufacturer", "model",
                          "installation_date", "criticality", "location", "operating_hours"]

        for field in priority_fields:
            if field in asset_info:
                formatted_lines.append(f"- {field.replace('_', ' ').title()}: {asset_info[field]}")

        for key, value in asset_info.items():
            if key not in priority_fields:
                formatted_lines.append(f"- {key.replace('_', ' ').title()}: {value}")

        return "\n".join(formatted_lines)

    async def generate_prediction(
        self,
        asset_info: Dict[str, Any],
        work_orders: str,
        inspections: str,
        service_schedules: str,
        maintenance_requests: str,
        knowledge_base: str,
        sensor_data: str
    ) -> Dict[str, Any]:
        """Generate breakdown prediction using LLM.

        Args:
            asset_info: Formatted asset information
            work_orders: Retrieved work orders
            inspections: Retrieved inspections
            service_schedules: Retrieved service schedules
            maintenance_requests: Retrieved maintenance requests
            knowledge_base: Retrieved knowledge base articles
            sensor_data: Formatted sensor data

        Returns:
            Parsed prediction results
        """
        # Get the prompt template
        template = Prompts.get_template("breakdown_prediction_agent")

        # Render the prompt with data
        user_prompt = template.render(
            asset_info=self.format_asset_info(asset_info),
            work_orders=work_orders,
            inspections=inspections,
            service_schedules=service_schedules,
            maintenance_requests=maintenance_requests,
            knowledge_base=knowledge_base,
            sensor_data=sensor_data
        )

        system_prompt = self.prompts.breakdown_prediction_system

        try:
            logger.info(f"Sending prediction request to model: {self.current_model}")
            logger.info(f"User prompt length: {len(user_prompt)} characters")

            response = await self.model_client.chat.completions.create(
                model=self.current_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=16000  # Increased to handle full response with thought_process
            )
            logger.info(f"Response finish_reason: {response.choices[0].finish_reason if response.choices else 'N/A'}")

            # Check if response has choices
            if not response.choices:
                logger.error("No choices in model response")
                return {
                    "raw_response": "",
                    "parse_error": "Model returned no choices"
                }

            response_text = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Handle None or empty response
            if response_text is None or response_text.strip() == "":
                logger.error(f"Model returned empty response. Finish reason: {finish_reason}")
                return {
                    "raw_response": "",
                    "parse_error": f"Model returned empty response. Finish reason: {finish_reason}"
                }

            # Warn if response was truncated due to length
            if finish_reason == "length":
                logger.warning(f"Response was truncated due to max_tokens limit. Response length: {len(response_text)}")

            logger.info(f"Received response with {len(response_text)} characters")
            logger.debug(f"Response preview: {response_text[:500]}...")

            # Try to parse JSON from response
            try:
                # Find JSON block in response
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1

                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    prediction = json.loads(json_str)
                    return prediction
                else:
                    # Return raw response if no JSON found
                    logger.warning(f"No JSON found in response. Response: {response_text[:200]}...")
                    return {
                        "raw_response": response_text,
                        "parse_error": "No JSON structure found in response"
                    }
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse prediction JSON: {e}")
                # Attempt to repair truncated JSON
                if finish_reason == "length":
                    logger.info("Attempting to repair truncated JSON...")
                    repaired = self._repair_truncated_json(response_text)
                    if repaired:
                        return repaired
                return {
                    "raw_response": response_text,
                    "parse_error": str(e)
                }

        except Exception as e:
            logger.error(f"Error generating prediction: {e}", exc_info=True)
            raise

    async def predict_breakdown(
        self,
        prediction_id: str,
        asset_info: Dict[str, Any],
        sensor_data: List[Dict[str, Any]],
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Complete prediction pipeline for an asset.

        Args:
            prediction_id: Unique prediction identifier
            asset_info: Asset information dictionary
            sensor_data: List of sensor readings
            collection_name: Optional vector DB collection for RAG

        Returns:
            Complete prediction results
        """
        asset_id = asset_info.get("asset_id", "unknown")
        asset_name = asset_info.get("asset_name", "Unknown Asset")

        logger.info(f"[Prediction {prediction_id}] Starting breakdown prediction for {asset_name}")

        # Step 1: Format sensor data
        logger.info(f"[Prediction {prediction_id}] Step 1: Processing sensor data")
        formatted_sensor_data = self.format_sensor_data(sensor_data)

        # Step 2: Retrieve historical data from vector DB (filtered by doc_type and asset_id)
        logger.info(f"[Prediction {prediction_id}] Step 2: Retrieving work orders")
        work_orders = await self.retrieve_work_orders(asset_id, collection_name)

        logger.info(f"[Prediction {prediction_id}] Step 3: Retrieving inspections")
        inspections = await self.retrieve_inspections(asset_id, collection_name)

        logger.info(f"[Prediction {prediction_id}] Step 4: Retrieving service schedules")
        service_schedules = await self.retrieve_service_schedules(asset_id, collection_name)

        logger.info(f"[Prediction {prediction_id}] Step 5: Retrieving maintenance requests")
        maintenance_requests = await self.retrieve_maintenance_requests(asset_id, collection_name)

        logger.info(f"[Prediction {prediction_id}] Step 6: Retrieving knowledge base")
        knowledge_base = await self.retrieve_knowledge_base(collection_name)

        # Step 7: Generate prediction
        logger.info(f"[Prediction {prediction_id}] Step 7: Generating prediction with LLM")
        prediction = await self.generate_prediction(
            asset_info,
            work_orders,
            inspections,
            service_schedules,
            maintenance_requests,
            knowledge_base,
            formatted_sensor_data
        )

        # Add metadata
        result = {
            "prediction_id": prediction_id,
            "asset_id": asset_id,
            "asset_name": asset_name,
            "generated_at": datetime.utcnow().isoformat(),
            "prediction": prediction,
            "data_sources": {
                "work_orders_retrieved": "No historical work orders" not in work_orders,
                "inspections_retrieved": "No historical inspection" not in inspections,
                "service_schedules_retrieved": "No service schedules" not in service_schedules,
                "maintenance_requests_retrieved": "No maintenance requests" not in maintenance_requests,
                "knowledge_base_retrieved": "No relevant knowledge" not in knowledge_base,
                "sensor_data_available": bool(sensor_data)
            }
        }

        logger.info(f"[Prediction {prediction_id}] Prediction completed successfully")
        return result

    async def predict_batch(
        self,
        assets: List[Dict[str, Any]],
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Run predictions for multiple assets.

        Args:
            assets: List of asset configurations, each containing:
                - asset_info: Asset metadata
                - sensor_data: Live sensor readings
            collection_name: Optional vector DB collection

        Returns:
            List of prediction results for all assets
        """
        results = []

        for i, asset_config in enumerate(assets):
            prediction_id = str(uuid.uuid4())
            asset_info = asset_config.get("asset_info", {})
            sensor_data = asset_config.get("sensor_data", [])

            try:
                logger.info(f"Processing asset {i+1}/{len(assets)}: {asset_info.get('asset_name', 'Unknown')}")

                result = await self.predict_breakdown(
                    prediction_id=prediction_id,
                    asset_info=asset_info,
                    sensor_data=sensor_data,
                    collection_name=collection_name
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Error predicting for asset {asset_info.get('asset_id', 'unknown')}: {e}")
                results.append({
                    "prediction_id": prediction_id,
                    "asset_id": asset_info.get("asset_id", "unknown"),
                    "asset_name": asset_info.get("asset_name", "Unknown"),
                    "error": str(e),
                    "generated_at": datetime.utcnow().isoformat()
                })

        return results

    async def store_prediction(
        self,
        prediction_result: Dict[str, Any]
    ) -> bool:
        """Store prediction result in database.

        Args:
            prediction_result: Prediction results to store

        Returns:
            True if stored successfully
        """
        try:
            prediction_id = prediction_result.get("prediction_id")
            asset_id = prediction_result.get("asset_id")

            await self.storage.store_breakdown_prediction(
                prediction_id=prediction_id,
                asset_id=asset_id,
                prediction_data=json.dumps(prediction_result),
                risk_level=prediction_result.get("prediction", {}).get(
                    "prediction_summary", {}
                ).get("overall_risk_level", "UNKNOWN"),
                breakdown_probability=prediction_result.get("prediction", {}).get(
                    "prediction_summary", {}
                ).get("breakdown_probability_30_days", 0)
            )

            logger.info(f"Stored prediction {prediction_id} for asset {asset_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            return False
