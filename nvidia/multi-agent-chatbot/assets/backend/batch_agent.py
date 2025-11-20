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
"""Batch Image Analysis Agent for processing multiple images and generating reports.

This agent uses a multi-step approach:
1. Extract image description using explain_image tool (vision model) with the user's prompt
2. Retrieve relevant context from vector embeddings
3. Use GPT model to generate comprehensive analysis combining all inputs
"""

import uuid
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI

from client import MCPClient
from logger import logger
from postgres_storage import PostgreSQLConversationStorage


class BatchAnalysisAgent:
    """Agent for batch image analysis that combines vision, RAG, and LLM capabilities.

    This agent processes multiple images by:
    1. Getting visual description from the vision model (explain_image) using the user's prompt
    2. Querying vector store for relevant context based on the image and prompt
    3. Using GPT model to synthesize a comprehensive analysis
    """

    def __init__(
        self,
        vector_store,
        config_manager,
        postgres_storage: PostgreSQLConversationStorage
    ):
        """Initialize the batch analysis agent.

        Args:
            vector_store: VectorStore instance for document retrieval
            config_manager: ConfigManager for reading configuration
            postgres_storage: PostgreSQL storage for persistence
        """
        self.vector_store = vector_store
        self.config_manager = config_manager
        self.storage = postgres_storage
        self.mcp_client = None
        self.tools_by_name = {}
        self.model_client = None
        self.current_model = None

    async def init_tools(self) -> None:
        """Initialize MCP client and load required tools."""
        self.mcp_client = await MCPClient().init()

        try:
            mcp_tools = await self.mcp_client.get_tools()
            self.tools_by_name = {tool.name: tool for tool in mcp_tools}
            logger.info(f"BatchAnalysisAgent loaded {len(mcp_tools)} tools: {list(self.tools_by_name.keys())}")

            if "explain_image" not in self.tools_by_name:
                logger.warning("explain_image tool not found - batch analysis may not work")
        except Exception as e:
            logger.error(f"Failed to load MCP tools for batch agent: {e}")
            raise

        # Initialize model client
        model_name = self.config_manager.get_selected_model()
        self.set_current_model(model_name)

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
            logger.info(f"BatchAnalysisAgent using model: {model_name}")
        else:
            raise ValueError(f"Model {model_name} is not available. Available: {available_models}")

    async def get_image_description(self, image_data: str, user_prompt: str) -> str:
        """Get detailed description of an image using the vision model.

        The user's prompt is passed directly to the vision model so it can answer
        specific questions about the image (e.g., finding specific text, counting objects).

        Args:
            image_data: Base64 encoded image data
            user_prompt: User's analysis prompt - sent directly to vision model

        Returns:
            Detailed image description/analysis from vision model
        """
        if "explain_image" not in self.tools_by_name:
            raise RuntimeError("explain_image tool not available")

        tool = self.tools_by_name["explain_image"]

        try:
            # Pass the user's prompt directly to the vision model
            # This allows specific queries like "find the word DANGER" to be answered
            vision_query = f"""Analyze this image and respond to the following request:

{user_prompt}

In your response:
1. First, address the specific request above with detailed findings
2. Then provide a comprehensive description of all visible elements, objects, text, colors, conditions, and notable features
3. Be specific about locations, quantities, and characteristics when relevant"""

            result = await tool.ainvoke({
                "query": vision_query,
                "image": image_data,
                "return_bounding_boxes": False
            })
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            logger.error(f"Error getting image description: {e}")
            raise

    async def get_relevant_context(
        self,
        query: str,
        image_description: str,
        organization: Optional[str] = None
    ) -> str:
        """Retrieve relevant context from vector store.

        Args:
            query: The analysis prompt
            image_description: Description of the image from vision model
            organization: Optional organization/cluster name to filter results

        Returns:
            Relevant context from vector embeddings
        """
        try:
            # Combine query and image description for better context retrieval
            search_query = f"{query}\n\nImage contents: {image_description}"

            # Query the vector store with optional organization filtering
            results = self.vector_store.get_documents(
                search_query,
                k=3,
                organization=organization
            )

            if results:
                context_parts = []
                for i, doc in enumerate(results, 1):
                    context_parts.append(f"[Context {i}]: {doc.page_content}")
                return "\n\n".join(context_parts)
            else:
                return ""
        except Exception as e:
            logger.warning(f"Error retrieving context from vector store: {e}")
            return ""

    async def generate_analysis(
        self,
        analysis_prompt: str,
        image_description: str,
        vector_context: str,
        filename: str
    ) -> str:
        """Generate comprehensive analysis using GPT model.

        Args:
            analysis_prompt: User's analysis prompt
            image_description: Description from vision model (which already addressed the prompt)
            vector_context: Relevant context from vector store
            filename: Name of the image file

        Returns:
            Comprehensive analysis from GPT model
        """
        system_prompt = """You are an expert image analyst. Your task is to generate a comprehensive,
well-formatted analysis report based on the provided information.

You will receive:
1. A visual analysis from a vision model that has already examined the image
2. Relevant context from a knowledge base (if available)
3. The original analysis request from the user

Generate a professional analysis report that:
- Directly addresses the user's specific requirements
- Presents the vision model's findings clearly
- Incorporates relevant context to provide deeper insights
- Provides actionable observations where appropriate
- Uses clear formatting with headers and bullet points"""

        user_message = f"""## Image: {filename}

### Vision Model Analysis:
{image_description}

### Knowledge Base Context:
{vector_context if vector_context else "No additional context available from knowledge base."}

### Original Analysis Request:
{analysis_prompt}

---

Please synthesize this information into a clear, comprehensive analysis report for this image."""

        try:
            response = await self.model_client.chat.completions.create(
                model=self.current_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating analysis with GPT: {e}")
            raise

    async def analyze_single_image(
        self,
        image_id: str,
        image_data: str,
        filename: str,
        analysis_prompt: str,
        organization: Optional[str] = None
    ) -> str:
        """Complete analysis pipeline for a single image.

        Args:
            image_id: Image identifier
            image_data: Base64 encoded image data
            filename: Original filename
            analysis_prompt: User's analysis prompt
            organization: Optional organization/source name for vector search

        Returns:
            Complete analysis result
        """
        # Step 1: Get image description from vision model (with user's prompt)
        logger.info(f"Step 1: Getting visual analysis for {filename}")
        image_description = await self.get_image_description(image_data, analysis_prompt)

        # Step 2: Get relevant context from vector store
        logger.info(f"Step 2: Retrieving relevant context for {filename}")
        vector_context = await self.get_relevant_context(analysis_prompt, image_description, organization)

        # Step 3: Generate comprehensive analysis with GPT
        logger.info(f"Step 3: Generating final analysis for {filename}")
        analysis = await self.generate_analysis(
            analysis_prompt,
            image_description,
            vector_context,
            filename
        )

        return analysis

    async def process_batch(
        self,
        batch_id: str,
        image_ids: List[str],
        analysis_prompt: str,
        report_format: str = "markdown",
        organization: Optional[str] = None
    ) -> None:
        """Process a batch of images and store results.

        Args:
            batch_id: Unique batch identifier
            image_ids: List of image IDs to process
            analysis_prompt: The prompt to use for analyzing each image
            report_format: Output format for the report
            organization: Optional organization/source name for vector search filtering
        """
        logger.info(f"Starting batch analysis {batch_id} with {len(image_ids)} images")
        if organization:
            logger.info(f"Using organization filter for vector search: {organization}")

        # Create the batch job record
        await self.storage.create_batch_job(
            batch_id=batch_id,
            total_images=len(image_ids),
            analysis_prompt=analysis_prompt,
            report_format=report_format
        )

        # Update status to processing
        await self.storage.update_batch_progress(batch_id, 0, "processing")

        processed_count = 0

        try:
            for i, image_id in enumerate(image_ids):
                result_id = str(uuid.uuid4())
                metadata = None

                try:
                    # Get image data and metadata
                    image_data = await self.storage.get_image(image_id)
                    if not image_data:
                        error_msg = f"Image not found: {image_id}"
                        logger.warning(error_msg)
                        await self.storage.store_batch_result(
                            result_id=result_id,
                            batch_id=batch_id,
                            image_id=image_id,
                            image_filename="unknown",
                            analysis_result="",
                            error_message=error_msg
                        )
                        processed_count += 1
                        await self.storage.update_batch_progress(batch_id, processed_count)
                        continue

                    # Get metadata for filename
                    metadata = await self.storage.get_image_metadata(image_id)
                    filename = metadata.get("filename", f"image_{i+1}") if metadata else f"image_{i+1}"

                    logger.info(f"[Batch {batch_id}] Processing image {i+1}/{len(image_ids)}: {filename}")

                    # Run the complete analysis pipeline
                    analysis_result = await self.analyze_single_image(
                        image_id,
                        image_data,
                        filename,
                        analysis_prompt,
                        organization
                    )

                    # Store the result
                    await self.storage.store_batch_result(
                        result_id=result_id,
                        batch_id=batch_id,
                        image_id=image_id,
                        image_filename=filename,
                        analysis_result=analysis_result
                    )

                    logger.info(f"[Batch {batch_id}] Completed analysis for {filename}")

                except Exception as e:
                    error_msg = f"Error analyzing image {image_id}: {str(e)}"
                    logger.error(error_msg)

                    # Store error result
                    await self.storage.store_batch_result(
                        result_id=result_id,
                        batch_id=batch_id,
                        image_id=image_id,
                        image_filename=metadata.get("filename", "unknown") if metadata else "unknown",
                        analysis_result="",
                        error_message=error_msg
                    )

                # Update progress after each image
                processed_count += 1
                await self.storage.update_batch_progress(batch_id, processed_count)

            # Mark as completed
            await self.storage.update_batch_progress(batch_id, processed_count, "completed")
            logger.info(f"Batch analysis {batch_id} completed successfully")

        except Exception as e:
            error_msg = f"Batch analysis failed: {str(e)}"
            logger.error(error_msg)
            await self.storage.set_batch_error(batch_id, error_msg)
            raise

    async def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a batch job.

        Args:
            batch_id: Batch identifier

        Returns:
            Batch job details including progress
        """
        return await self.storage.get_batch_job(batch_id)

    async def get_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Get all results for a completed batch.

        Args:
            batch_id: Batch identifier

        Returns:
            List of analysis results
        """
        return await self.storage.get_batch_results(batch_id)
