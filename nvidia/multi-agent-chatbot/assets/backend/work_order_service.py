# work_order_service.py
#
# Org-scoped Work Order service.
#
# Uses the shared VectorStore for all Milvus interaction:
#   - Summarize the JSON work order
#   - Store the summary as a Document with metadata (doc_type='work_order')
#   - Search work orders per org
#   - Basic org stats
#

from typing import Any, Dict, List
from langchain_core.documents import Document

from vector_store import VectorStore
from logger import logger

from pymilvus import connections, utility, Collection


class WorkOrderService:
    """
    Service for ingesting and searching work orders on a per-organization basis.

    IMPORTANT:
      - This service does NOT create Milvus collections or indexes directly.
      - It delegates all schema/index management to the shared VectorStore.
      - Collection name is derived from organization_id in a Milvus-safe way and
        is shared with the other org-scoped doc types (inspection, etc.).
    """

    def __init__(self, vector_store: VectorStore, summarizer: Any):
        """
        Args:
            vector_store: Shared VectorStore instance.
            summarizer: WorkOrderSummarizer instance (must expose an async
                        summarize_work_order(...) or summarize(...)).
        """
        self.vector_store = vector_store
        self.summarizer = summarizer
        self.milvus_uri = vector_store.uri

    # -------------------------------------------------------------------------
    # Collection naming
    # -------------------------------------------------------------------------

    def _get_collection_name(self, organization_id: str) -> str:
        """
        Derive the Milvus collection name from organization_id.

        We keep this consistent with OrgJsonIngestService and the MCP tools:
          - Sanitize to [A-Za-z0-9_]
          - Ensure first character is a letter or underscore by prefixing 'org_'
        """
        base = "".join(
            c if c.isalnum() or c == "_" else "_"
            for c in str(organization_id)
        )
        if not base or not (base[0].isalpha() or base[0] == "_"):
            base = "org_" + base
        return base

    # -------------------------------------------------------------------------
    # Ingest
    # -------------------------------------------------------------------------

    async def process_work_order(self, work_order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize the work order JSON and store as a single summary document
        in the per-org collection with doc_type='work_order'.
        """
        organization_id = work_order.get("organization_id")
        if not organization_id:
            raise ValueError("Missing required field: organization_id")

        work_order_id = work_order.get("work_order_id") or work_order.get("id")
        collection_name = self._get_collection_name(organization_id)

        logger.info({
            "message": "Processing work order",
            "organization_id": organization_id,
            "work_order_id": work_order_id,
            "collection_name": collection_name,
        })

        # 1) Summarize using WorkOrderSummarizer
        if hasattr(self.summarizer, "summarize_work_order"):
            summary = await self.summarizer.summarize_work_order(work_order)
        elif hasattr(self.summarizer, "summarize"):
            summary = await self.summarizer.summarize(work_order)
        else:
            raise RuntimeError(
                "WorkOrderSummarizer must expose summarize_work_order(...) or summarize(...)."
            )

        # 2) Metadata for VectorStore
        metadata: Dict[str, Any] = {
            "doc_type": "work_order",
            "organization_id": str(organization_id),
        }
        if work_order_id:
            metadata["work_order_id"] = str(work_order_id)

        doc = Document(page_content=summary, metadata=metadata)

        # 3) Index via VectorStore (no manual Milvus schema/index management)
        self.vector_store.index_documents(
            [doc],
            no_split=True,      # 1 WO → 1 summary chunk
            upsert_by_id=False, # no logical id-based upsert for now
            collection_name=collection_name,
        )

        logger.info({
            "message": "Work order stored via VectorStore",
            "organization_id": organization_id,
            "work_order_id": work_order_id,
            "collection_name": collection_name,
            "summary_length": len(summary),
        })

        return {
            "status": "success",
            "doc_type": "work_order",
            "organization_id": organization_id,
            "work_order_id": work_order_id,
            "collection_name": collection_name,
            "summary": summary,
            "metadata": metadata,
            "message": f"work_order stored in collection '{collection_name}'",
        }

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    async def search_work_orders(
        self,
        organization_id: str,
        query: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over work order summaries for a single organization.
        """
        collection_name = self._get_collection_name(organization_id)

        logger.info({
            "message": "Searching work orders",
            "organization_id": organization_id,
            "collection_name": collection_name,
            "query": query,
            "k": k,
        })

        docs = self.vector_store.get_documents(
            query=query,
            k=k,
            collection_name=collection_name,
        )

        results: List[Dict[str, Any]] = []
        for d in docs:
            md = d.metadata or {}
            # Filter defensively by doc_type
            if md.get("doc_type") not in (None, "work_order"):
                continue
            results.append(
                {
                    "text": d.page_content,
                    "metadata": md,
                }
            )

        logger.info({
            "message": "Work order search completed",
            "organization_id": organization_id,
            "collection_name": collection_name,
            "query": query,
            "returned": len(results),
        })

        return results

    # -------------------------------------------------------------------------
    # Org discovery / stats
    # -------------------------------------------------------------------------

    async def list_organizations(self) -> List[str]:
        """
        List all organizations that have a per-org collection (any doc_type)
        that matches this naming pattern.
        """
        orgs: List[str] = []
        connections.connect(uri=self.milvus_uri)
        try:
            cols = utility.list_collections()
            for name in cols:
                # We treat any collection whose name starts with 'org_' as an org-scoped collection
                if name.startswith("org_"):
                    orgs.append(name)
        finally:
            try:
                connections.disconnect("default")
            except Exception:
                pass

        return orgs

    async def get_organization_stats(self, organization_id: str) -> Dict[str, Any]:
        """
        Basic stats for an organization's collection (entity count).
        """
        collection_name = self._get_collection_name(organization_id)
        connections.connect(uri=self.milvus_uri)
        try:
            if not utility.has_collection(collection_name):
                return {
                    "organization_id": organization_id,
                    "collection": collection_name,
                    "exists": False,
                }

            c = Collection(collection_name)
            c.load()
            num_entities = c.num_entities

            return {
                "organization_id": organization_id,
                "collection": collection_name,
                "exists": True,
                "num_entities": num_entities,
            }
        finally:
            try:
                connections.disconnect("default")
            except Exception:
                pass
