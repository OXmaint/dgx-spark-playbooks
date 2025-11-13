#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Work Order Processing Service with Vector Storage"""

from typing import Dict, Any, List
from langchain_core.documents import Document
from work_order_summarizer import WorkOrderSummarizer
from vector_store import VectorStore
from logger import logger
import json


class WorkOrderService:
    """Service for processing and storing work orders with summaries in Milvus"""
    
    def __init__(self, vector_store: VectorStore, summarizer: WorkOrderSummarizer):
        """Initialize the work order service.
        
        Args:
            vector_store: VectorStore instance for storage (uses Milvus)
            summarizer: WorkOrderSummarizer instance
        """
        self.vector_store = vector_store
        self.summarizer = summarizer
        
    async def process_work_order(self, work_order: Dict[str, Any]) -> Dict[str, Any]:
        """Process a work order: summarize, extract metadata, and store in Milvus with embeddings.
        
        The entire work order is stored in Milvus:
        - Summary text is embedded and stored as the main content
        - LLM-extracted metadata is stored as document metadata
        - Original work order JSON is included in metadata
        
        Args:
            work_order: Work order JSON payload
            
        Returns:
            Dict with summary, metadata, and status
        """
        try:
            logger.info({
                "message": "Processing work order",
                "work_order_keys": list(work_order.keys())
            })
            
            # Let the LLM generate summary AND decide what metadata is important
            summary, metadata = await self.summarizer.summarize_and_extract_metadata(work_order)
            
            # Create document - this will be embedded and stored in Milvus
            document = Document(
                page_content=summary,
                metadata=metadata
            )
            
            # Index the document in Milvus
            # This creates embeddings using qwen3-embedding and stores:
            # - The embedding vector
            # - The summary text
            # - All metadata fields
            self.vector_store.index_documents([document])
            
            logger.info({
                "message": "Work order stored in Milvus",
                "summary_length": len(summary),
                "metadata_fields": list(metadata.keys()),
                "storage": "milvus"
            })
            
            return {
                "status": "success",
                "summary": summary,
                "metadata": metadata,
                "storage_location": "milvus",
                "message": "Work order processed and stored with embeddings in Milvus vector database"
            }
            
        except Exception as e:
            logger.error(f"Error processing work order: {e}", exc_info=True)
            raise
    
    async def search_work_orders(
        self, 
        query: str, 
        k: int = 5,
        filters: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """Search work orders by semantic similarity in Milvus.
        
        Args:
            query: Search query (will be embedded and compared to stored embeddings)
            k: Number of results to return
            filters: Optional metadata filters (e.g., {"priority": "high"})
            
        Returns:
            List of matching work orders with summaries and metadata
        """
        try:
            # Search in Milvus using semantic similarity
            documents = self.vector_store.get_documents(
                query=query,
                k=k,
                sources=["work_orders"]
            )
            
            results = []
            for doc in documents:
                result = {
                    "summary": doc.page_content,
                    "metadata": {}
                }
                
                # Extract all metadata
                for key, value in doc.metadata.items():
                    if key == "raw_work_order":
                        try:
                            result["original_work_order"] = json.loads(value)
                        except:
                            result["original_work_order"] = value
                    elif key != "source":
                        result["metadata"][key] = value
                
                # Apply filters if provided
                if filters:
                    matches_filter = all(
                        result["metadata"].get(k) == v 
                        for k, v in filters.items()
                    )
                    if matches_filter:
                        results.append(result)
                else:
                    results.append(result)
            
            logger.info({
                "message": "Work order search completed",
                "query": query,
                "results_count": len(results),
                "storage": "milvus"
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching work orders: {e}", exc_info=True)
            return []
    
    async def get_work_order_stats(self) -> Dict[str, Any]:
        """Get statistics about stored work orders in Milvus.
        
        Returns:
            Statistics dictionary
        """
        try:
            # This is a simple implementation - you could extend it
            # to query Milvus for actual counts
            return {
                "storage": "milvus",
                "collection": "context",
                "source_filter": "work_orders",
                "note": "Work orders are stored as embeddings in Milvus vector database"
            }
        except Exception as e:
            logger.error(f"Error getting work order stats: {e}")
            return {}