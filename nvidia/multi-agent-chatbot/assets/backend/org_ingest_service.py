# org_ingest_service.py
#
# Generic org-scoped JSON ingest service for inspections, maintenance requests,
# PM schedules, etc. Uses the shared VectorStore and a doc_type-specific
# summarizer to store summarized JSON into Milvus collections per organization.
#

from typing import Dict, Any, Callable, Awaitable
from langchain_core.documents import Document

from vector_store import VectorStore
from logger import logger


class OrgJsonIngestService:
    """
    Generic JSON ingest service for org-scoped entities like inspections,
    maintenance requests, pm_schedules, etc.

    Workflow:
      1. Take a JSON payload that MUST include organization_id.
      2. Summarize it using a provided async summarizer function.
      3. Wrap the summary as a Document with metadata including doc_type.
      4. Index into the VectorStore in the per-org Milvus collection.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        summarizer: Callable[[Dict[str, Any]], Awaitable[str]],
        doc_type: str,
        id_field: str = "id",
    ):
        """
        Args:
            vector_store: Shared VectorStore instance.
            summarizer: Async function: (payload: dict) -> summary string.
            doc_type: Logical type tag ("inspection", "maintenance_request", "pm_schedule", etc.).
            id_field: Field name used as logical id in metadata (optional, e.g. "inspection_id").
        """
        self.vector_store = vector_store
        self.summarizer = summarizer
        self.doc_type = doc_type
        self.id_field = id_field
        self.milvus_uri = vector_store.uri

    def _get_collection_name(self, organization_id: str) -> str:
        """Derive per-org collection name from org_id (sanitized, Milvus-safe)."""
        base = "".join(c if c.isalnum() or c == "_" else "_" for c in str(organization_id))
        # Milvus: first char must be letter or '_'
        if not base or not (base[0].isalpha() or base[0] == "_"):
            base = "org_" + base
        return base

    async def ingest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize JSON payload and store it into the per-org collection with proper doc_type.

        The payload MUST include:
          - organization_id

        If present, the field with name `self.id_field` is used as a logical id.
        """
        try:
            organization_id = payload.get("organization_id")
            if not organization_id:
                raise ValueError("Missing required field: organization_id")

            logical_id = payload.get(self.id_field)
            collection_name = self._get_collection_name(organization_id)

            logger.info({
                "message": f"Processing {self.doc_type}",
                "doc_type": self.doc_type,
                "organization_id": organization_id,
                "logical_id": logical_id,
                "collection_name": collection_name,
            })

            # 1) Summarize payload using provided summarizer
            summary = await self.summarizer(payload)

            # 2) Build metadata for this document
            metadata = {
                "doc_type": self.doc_type,
                "organization_id": str(organization_id),
            }
            if logical_id:
                metadata[self.id_field] = str(logical_id)

            # Optionally attach more doc-type-specific metadata here if you want, e.g.:
            # if self.doc_type == "inspection":
            #     metadata["inspection_type"] = payload.get("inspection_type")

            doc = Document(page_content=summary, metadata=metadata)

            # 3) Index into the per-org collection via VectorStore
            self.vector_store.index_documents(
                [doc],
                no_split=True,          # 1 JSON → 1 summary chunk → 1 vector
                upsert_by_id=False,     # or True if you later add a stable 'id' field
                collection_name=collection_name,
            )

            logger.info({
                "message": f"{self.doc_type} stored via VectorStore",
                "doc_type": self.doc_type,
                "organization_id": organization_id,
                "logical_id": logical_id,
                "collection_name": collection_name,
                "summary_length": len(summary),
            })

            return {
                "status": "success",
                "doc_type": self.doc_type,
                "organization_id": organization_id,
                "logical_id": logical_id,
                "collection_name": collection_name,
                "summary": summary,
                "metadata": metadata,
                "message": f"{self.doc_type} stored in collection '{collection_name}'",
            }

        except Exception as e:
            logger.error(f"Error ingesting {self.doc_type}: {e}", exc_info=True)
            raise
