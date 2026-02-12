#
# RAG Work-Order / Org-Scoped Search MCP Server
#
# Provides search tools for:
#   - search_docs
#   - search_wo
#   - search_inspection
#   - search_maintenance_request
#   - search_pm_schedules
#   - search_knowledge
#
# All tools:
#   • Require org_id (used as Milvus collection name via VectorStore).
#   • Apply a mandatory doc_type filter in metadata.
#   • Return top-k chunks with text + metadata.
#

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import ConfigManager
from vector_store import create_vector_store_with_config
from logger import logger as app_logger  # reuse app logger if you want


# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("rag-wo-mcp")


# ------------------------------------------------------------------------------
# Config + VectorStore
# ------------------------------------------------------------------------------

def _get_config_path() -> str:
    config_path = os.path.join(os.path.dirname(__file__), "../../config.json")
    if not os.path.exists(config_path):
        logger.error("ERROR: config.json not found at %s", config_path)
    return config_path


config_manager = ConfigManager(_get_config_path())
vector_store = create_vector_store_with_config(config_manager)


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

DOC_TYPE_KEYS = ("doc_type", "type")


def _matches_doc_type(metadata: Dict[str, Any], wanted: str) -> bool:
    """Return True if metadata indicates the desired doc_type."""
    if not metadata or not wanted:
        return False

    wanted_lower = str(wanted).lower()
    for key in DOC_TYPE_KEYS:
        if key not in metadata:
            continue
        val = metadata[key]
        if val is None:
            continue
        # support strings, list, or comma-separated values
        if isinstance(val, str):
            # exact match on lowercased string
            if val.lower() == wanted_lower:
                return True
            # also support presence in comma-separated tags
            parts = [p.strip().lower() for p in val.split(",")]
            if wanted_lower in parts:
                return True
        elif isinstance(val, (list, tuple, set)):
            vals = [str(v).lower() for v in val]
            if wanted_lower in vals:
                return True
    return False


def _sanitize_org(org_id: str) -> str:
    base = "".join(c if c.isalnum() or c == "_" else "_" for c in str(org_id))
    if not base or not (base[0].isalpha() or base[0] == "_"):
        base = "org_" + base
    return base



def _search_by_doc_type(
    org_id: str,
    query: str,
    doc_type: Optional[str],
    k: int,
    overshoot_factor: int = 3,
) -> Dict[str, Any]:
    """
    Core search helper used by all tools.

    Steps:
      1. Use VectorStore.get_documents on collection=<org_id> (sanitized).
      2. Filter by metadata doc_type if provided.
      3. Return up to k results with text + metadata.
    """
    collection_name = _sanitize_org(org_id)
    effective_k = max(k * overshoot_factor, k)

    logger.info(
        {
            "message": "MCP search",
            "org_id": org_id,
            "collection": collection_name,
            "query": query,
            "doc_type": doc_type,
            "k": k,
            "effective_k": effective_k,
        }
    )

    try:
        docs = vector_store.get_documents(
            query=query,
            k=effective_k,
            collection_name=collection_name,
        )
    except Exception as e:
        logger.error(f"Error retrieving documents from VectorStore: {e}", exc_info=True)
        return {
            "org_id": org_id,
            "collection": collection_name,
            "doc_type": doc_type,
            "query": query,
            "results": [],
            "count": 0,
            "error": str(e),
        }

    results: List[Dict[str, Any]] = []
    for d in docs:
        md = d.metadata or {}
        if doc_type and not _matches_doc_type(md, doc_type):
            continue

        results.append(
            {
                "text": d.page_content,
                "metadata": md,
            }
        )
        if len(results) >= k:
            break

    logger.info(
        {
            "message": "MCP search completed",
            "org_id": org_id,
            "collection": collection_name,
            "doc_type": doc_type,
            "query": query,
            "returned": len(results),
        }
    )

    return {
        "org_id": org_id,
        "collection": collection_name,
        "doc_type": doc_type,
        "query": query,
        "results": results,
        "count": len(results),
    }


# ------------------------------------------------------------------------------
# MCP server + tools
# ------------------------------------------------------------------------------

mcp = FastMCP("RAG-WorkOrders")


@mcp.tool()
async def search_docs(query: str, org_id: str, k: int = 5) -> Dict[str, Any]:
    """
    Search generic documents (doc_type='docs') for a given organization.

    Args:
        query: Natural language query text.
        org_id: Organization identifier (used as collection name).
        k: Number of results to return.

    Returns:
        JSON dict with org_id, collection, query, doc_type, results, count.
    """
    return _search_by_doc_type(org_id=org_id, query=query, doc_type="docs", k=k)


@mcp.tool()
async def search_wo(query: str, org_id: str, k: int = 5) -> Dict[str, Any]:
    """
    Search work orders (doc_type='work_order') for a given organization.

    Args:
        query: Natural language query text.
        org_id: Organization identifier.
        k: Number of results to return.

    Returns:
        JSON dict with org_id, collection, query, doc_type, results, count.
    """
    return _search_by_doc_type(org_id=org_id, query=query, doc_type="work_order", k=k)


@mcp.tool()
async def search_inspection(query: str, org_id: str, k: int = 5) -> Dict[str, Any]:
    """
    Search inspections (doc_type='inspection') for a given organization.

    Args:
        query: Natural language query text.
        org_id: Organization identifier.
        k: Number of results to return.

    Returns:
        JSON dict with org_id, collection, query, doc_type, results, count.
    """
    return _search_by_doc_type(org_id=org_id, query=query, doc_type="inspection", k=k)


@mcp.tool()
async def search_maintenance_request(query: str, org_id: str, k: int = 5) -> Dict[str, Any]:
    """
    Search maintenance requests (doc_type='maintenance_request') for a given organization.

    Args:
        query: Natural language query text.
        org_id: Organization identifier.
        k: Number of results to return.

    Returns:
        JSON dict with org_id, collection, query, doc_type, results, count.
    """
    return _search_by_doc_type(org_id=org_id, query=query, doc_type="maintenance_request", k=k)


@mcp.tool()
async def search_pm_schedules(query: str, org_id: str, k: int = 5) -> Dict[str, Any]:
    """
    Search preventive maintenance schedules (doc_type='pm_schedule') for a given organization.

    Args:
        query: Natural language query text.
        org_id: Organization identifier.
        k: Number of results to return.

    Returns:
        JSON dict with org_id, collection, query, doc_type, results, count.
    """
    return _search_by_doc_type(org_id=org_id, query=query, doc_type="pm_schedule", k=k)


@mcp.tool()
async def search_knowledge(query: str, org_id: str, k: int = 5) -> Dict[str, Any]:
    """
    Search knowledge base documents (doc_type='knowledge') for a given organization.

    This tool can be used by default to pull additional org-specific context.

    Args:
        query: Natural language query text.
        org_id: Organization identifier.
        k: Number of results to return.

    Returns:
        JSON dict with org_id, collection, query, doc_type, results, count.
    """
    return _search_by_doc_type(org_id=org_id, query=query, doc_type="knowledge", k=k)


if __name__ == "__main__":
    print(f"Starting {mcp.name} MCP server...")
    mcp.run(transport="stdio")
