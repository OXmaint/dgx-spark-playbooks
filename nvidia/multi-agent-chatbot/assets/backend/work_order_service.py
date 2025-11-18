#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Work Order Processing Service with Vector Storage"""

from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from work_order_summarizer import WorkOrderSummarizer
from vector_store import VectorStore
from logger import logger
import json
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility


class WorkOrderService:
    """Service for processing and storing work orders per organization in Milvus"""
    
    def __init__(self, vector_store: VectorStore, summarizer: WorkOrderSummarizer):
        """Initialize the work order service.
        
        Args:
            vector_store: VectorStore instance for storage (uses Milvus)
            summarizer: WorkOrderSummarizer instance
        """
        self.vector_store = vector_store
        self.summarizer = summarizer
        self.milvus_uri = vector_store.uri
        
    def _get_collection_name(self, organization_id: str) -> str:
        """Generate collection name from organization_id.
        
        Args:
            organization_id: Organization identifier
            
        Returns:
            Collection name (sanitized)
        """
        # Sanitize collection name: only alphanumeric and underscores
        sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in str(organization_id))
        return f"org_{sanitized}_work_orders"
    
    def _ensure_collection_exists(self, organization_id: str) -> str:
        """Ensure collection exists for organization, create if not.
        
        Args:
            organization_id: Organization identifier
            
        Returns:
            Collection name
        """
        try:
            connections.connect(uri=self.milvus_uri)
            collection_name = self._get_collection_name(organization_id)
            
            if utility.has_collection(collection_name):
                logger.info({
                    "message": "Collection already exists",
                    "organization_id": organization_id,
                    "collection_name": collection_name
                })
                connections.disconnect("default")
                return collection_name
            
            # Create new collection with schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=2560),
                FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=5000),
                FieldSchema(name="organization_id", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="work_order_id", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="work_order_type_id", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="work_order_number", dtype=DataType.VARCHAR, max_length=200),
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description=f"Work orders for organization {organization_id}"
            )
            
            collection = Collection(name=collection_name, schema=schema)
            
            # Create index on vector field
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            
            logger.info({
                "message": "Created new collection for organization",
                "organization_id": organization_id,
                "collection_name": collection_name
            })
            
            connections.disconnect("default")
            return collection_name
            
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}", exc_info=True)
            connections.disconnect("default")
            raise
    
    async def process_work_order(self, work_order: Dict[str, Any]) -> Dict[str, Any]:
        """Process a work order: summarize and store in organization-specific collection.
        
        Required fields in work_order:
        - organization_id
        - work_order_id
        - work_order_type_id
        - work_order_number
        
        Args:
            work_order: Work order JSON payload
            
        Returns:
            Dict with summary, metadata, and status
        """
        try:
            # Extract required metadata fields
            organization_id = work_order.get("organization_id")
            work_order_id = work_order.get("work_order_id")
            work_order_type_id = work_order.get("work_order_type_id")
            work_order_number = work_order.get("work_order_number")
            
            # Validate required fields
            if not all([organization_id, work_order_id, work_order_type_id, work_order_number]):
                raise ValueError(
                    "Missing required fields: organization_id, work_order_id, "
                    "work_order_type_id, work_order_number"
                )
            
            logger.info({
                "message": "Processing work order",
                "organization_id": organization_id,
                "work_order_id": work_order_id
            })
            
            # Ensure collection exists for this organization
            collection_name = self._ensure_collection_exists(organization_id)
            
            # Generate summary using LLM
            summary = await self.summarizer.summarize_work_order(work_order)
            
            # Create embedding for the summary
            embedding = self.vector_store.embeddings.embed_query(summary)
            
            # Store in Milvus
            connections.connect(uri=self.milvus_uri)
            collection = Collection(collection_name)
            
            data = [
                [embedding],  # embedding vector
                [summary],  # summary text
                [str(organization_id)],  # organization_id
                [str(work_order_id)],  # work_order_id
                [str(work_order_type_id)],  # work_order_type_id
                [str(work_order_number)]  # work_order_number
            ]
            
            collection.insert(data)
            collection.flush()
            
            connections.disconnect("default")
            
            logger.info({
                "message": "Work order stored in Milvus",
                "organization_id": organization_id,
                "work_order_id": work_order_id,
                "collection_name": collection_name,
                "summary_length": len(summary)
            })
            
            return {
                "status": "success",
                "organization_id": organization_id,
                "work_order_id": work_order_id,
                "collection_name": collection_name,
                "summary": summary,
                "metadata": {
                    "organization_id": organization_id,
                    "work_order_id": work_order_id,
                    "work_order_type_id": work_order_type_id,
                    "work_order_number": work_order_number
                },
                "message": f"Work order stored in collection '{collection_name}'"
            }
            
        except Exception as e:
            logger.error(f"Error processing work order: {e}", exc_info=True)
            raise
    
    async def search_work_orders(
        self, 
        organization_id: str,
        query: str, 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search work orders within an organization by semantic similarity.
        
        Args:
            organization_id: Organization to search within
            query: Search query (will be embedded and compared)
            k: Number of results to return
            
        Returns:
            List of matching work orders with summaries and metadata
        """
        try:
            collection_name = self._get_collection_name(organization_id)
            
            connections.connect(uri=self.milvus_uri)
            
            if not utility.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} does not exist")
                connections.disconnect("default")
                return []
            
            collection = Collection(collection_name)
            collection.load()
            
            # Create embedding for query
            query_embedding = self.vector_store.embeddings.embed_query(query)
            
            # Search
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=k,
                output_fields=["summary", "organization_id", "work_order_id", 
                              "work_order_type_id", "work_order_number"]
            )
            
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "score": hit.score,
                        "summary": hit.entity.get("summary"),
                        "metadata": {
                            "organization_id": hit.entity.get("organization_id"),
                            "work_order_id": hit.entity.get("work_order_id"),
                            "work_order_type_id": hit.entity.get("work_order_type_id"),
                            "work_order_number": hit.entity.get("work_order_number")
                        }
                    })
            
            connections.disconnect("default")
            
            logger.info({
                "message": "Work order search completed",
                "organization_id": organization_id,
                "query": query,
                "results_count": len(formatted_results)
            })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching work orders: {e}", exc_info=True)
            connections.disconnect("default")
            return []
    
    async def list_organizations(self) -> List[Dict[str, Any]]:
        """List all organizations with work order collections.
        
        Returns:
            List of organizations with their collection info
        """
        try:
            connections.connect(uri=self.milvus_uri)
            
            all_collections = utility.list_collections()
            
            org_collections = []
            for coll_name in all_collections:
                if coll_name.startswith("org_") and coll_name.endswith("_work_orders"):
                    collection = Collection(coll_name)
                    org_id = coll_name.replace("org_", "").replace("_work_orders", "")
                    
                    org_collections.append({
                        "organization_id": org_id,
                        "collection_name": coll_name,
                        "work_order_count": collection.num_entities
                    })
            
            connections.disconnect("default")
            
            return org_collections
            
        except Exception as e:
            logger.error(f"Error listing organizations: {e}", exc_info=True)
            connections.disconnect("default")
            return []
    
    async def get_organization_stats(self, organization_id: str) -> Dict[str, Any]:
        """Get statistics for an organization's work orders.
        
        Args:
            organization_id: Organization identifier
            
        Returns:
            Statistics dictionary
        """
        try:
            collection_name = self._get_collection_name(organization_id)
            
            connections.connect(uri=self.milvus_uri)
            
            if not utility.has_collection(collection_name):
                connections.disconnect("default")
                return {
                    "organization_id": organization_id,
                    "exists": False,
                    "work_order_count": 0
                }
            
            collection = Collection(collection_name)
            
            stats = {
                "organization_id": organization_id,
                "collection_name": collection_name,
                "exists": True,
                "work_order_count": collection.num_entities
            }
            
            connections.disconnect("default")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting organization stats: {e}", exc_info=True)
            connections.disconnect("default")
            return {}