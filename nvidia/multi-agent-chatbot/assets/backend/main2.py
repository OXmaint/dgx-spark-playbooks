#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""FastAPI backend server for the chatbot application.

This module provides the main HTTP API endpoints and WebSocket connections for:
- Real-time chat via WebSocket
- File upload and document ingestion
- Configuration management (models, sources, chat settings)
- Chat history management
- Vector store operations
"""

import base64
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional, Dict
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agent import ChatAgent
from config import ConfigManager
from logger import logger, log_request, log_response, log_error
from models import ChatIdRequest, ChatRenameRequest, SelectedModelRequest
from postgres_storage import PostgreSQLConversationStorage
from utils import process_and_ingest_files_background
from vector_store import create_vector_store_with_config

from work_order_summarizer import WorkOrderSummarizer
from work_order_service import WorkOrderService

from pymilvus import connections, Collection, utility
import json as _json





POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "chatbot")
POSTGRES_USER = os.getenv("POSTGRES_USER", "chatbot_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "chatbot_password")

config_manager = ConfigManager("./config.json")

postgres_storage = PostgreSQLConversationStorage(
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    database=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD
)

vector_store = create_vector_store_with_config(config_manager)

vector_store._initialize_store()

work_order_summarizer: WorkOrderSummarizer | None = None
work_order_service: WorkOrderService | None = None




agent: ChatAgent | None = None
indexing_tasks: Dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks."""
    global agent, work_order_summarizer, work_order_service
    logger.debug("Initializing PostgreSQL storage and agent...")
    
    try:
        await postgres_storage.init_pool()
        logger.info("PostgreSQL storage initialized successfully")
        
        logger.debug("Initializing ChatAgent...")
        agent = await ChatAgent.create(
            vector_store=vector_store,
            config_manager=config_manager,
            postgres_storage=postgres_storage
        )
        logger.info("ChatAgent initialized successfully.")
        
        # Initialize work order services
        logger.debug("Initializing Work Order Summarizer...")
        work_order_summarizer = WorkOrderSummarizer(
            model_host=config_manager.get_selected_model(),
            model_port=8000
        )
        
        work_order_service = WorkOrderService(
            vector_store=vector_store,
            summarizer=work_order_summarizer
        )
        logger.info("Work Order Service initialized successfully.")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield
    
    try:
        await postgres_storage.close()
        logger.debug("PostgreSQL storage closed successfully")
    except Exception as e:
        logger.error(f"Error closing PostgreSQL storage: {e}")


app = FastAPI(
    title="Chatbot API",
    description="Backend API for LLM-powered chatbot with RAG capabilities",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint for the API."""
    try:
        # Check database connection
        db_healthy = False
        try:
            await postgres_storage.init_pool() if not postgres_storage.pool else None
            db_healthy = postgres_storage.pool is not None
        except:
            pass
        
        # Check vector store
        vector_store_healthy = vector_store._store is not None
        
        # Check agent
        agent_healthy = agent is not None
        
        # Check work order service
        work_order_service_healthy = work_order_service is not None
        
        overall_healthy = all([
            db_healthy,
            vector_store_healthy,
            agent_healthy,
            work_order_service_healthy
        ])
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "components": {
                "database": "healthy" if db_healthy else "unhealthy",
                "vector_store": "healthy" if vector_store_healthy else "unhealthy",
                "agent": "healthy" if agent_healthy else "unhealthy",
                "work_order_service": "healthy" if work_order_service_healthy else "unhealthy"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# ---------------------------------------------------------------------------------------
# Work Order (org-scoped) endpoints
# ---------------------------------------------------------------------------------------

@app.post("/work-orders/process")
async def process_work_order(work_order: Dict[str, Any]):
    """Process a work order: generate summary and store in organization-specific collection."""
    try:
        log_request({
            "organization_id": work_order.get("organization_id"),
            "work_order_id": work_order.get("work_order_id")
        }, "/work-orders/process")
        
        if not work_order_service:
            raise HTTPException(
                status_code=503,
                detail="Work order service not initialized"
            )
        
        result = await work_order_service.process_work_order(work_order)
        
        log_response({
            "status": result["status"],
            "collection": result["collection_name"]
        }, "/work-orders/process")
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log_error(e, "/work-orders/process")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing work order: {str(e)}"
        )


@app.get("/work-orders/search")
async def search_work_orders(organization_id: str, query: str, k: int = 5):
    """Search work orders within an organization by semantic similarity."""
    try:
        log_request({
            "organization_id": organization_id,
            "query": query,
            "k": k
        }, "/work-orders/search")
        
        if not work_order_service:
            raise HTTPException(
                status_code=503,
                detail="Work order service not initialized"
            )
        
        results = await work_order_service.search_work_orders(organization_id, query, k)
        
        log_response({"result_count": len(results)}, "/work-orders/search")
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        log_error(e, "/work-orders/search")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching work orders: {str(e)}"
        )


@app.get("/work-orders/organizations")
async def list_organizations():
    """List all organizations that have work order collections."""
    try:
        if not work_order_service:
            raise HTTPException(
                status_code=503,
                detail="Work order service not initialized"
            )
        
        orgs = await work_order_service.list_organizations()
        return {"organizations": orgs, "count": len(orgs)}
        
    except Exception as e:
        log_error(e, "/work-orders/organizations")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing organizations: {str(e)}"
        )


@app.get("/work-orders/organizations/{organization_id}/stats")
async def get_organization_stats(organization_id: str):
    """Get statistics for an organization's work orders."""
    try:
        if not work_order_service:
            raise HTTPException(
                status_code=503,
                detail="Work order service not initialized"
            )
        
        stats = await work_order_service.get_organization_stats(organization_id)
        return stats
        
    except Exception as e:
        log_error(e, f"/work-orders/organizations/{organization_id}/stats")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting organization stats: {str(e)}"
        )

# ---------------------------------------------------------------------------------------
# Generic document ingestion (collection-aware)
# ---------------------------------------------------------------------------------------

@app.post("/ingest")
async def ingest_files(
    background_tasks: BackgroundTasks,
    files: Optional[List[UploadFile]] = File(None),
    collection: Optional[str] = Form(None),  # NEW
    doc_type: Optional[str] = Form(None),
):
    """Ingest documents for vector search and RAG (optionally to a specific collection)."""
    try:
        log_request({"file_count": len(files) if files else 0, "collection": collection}, "/ingest")
        
        if not files:
            raise HTTPException(status_code=400, detail="No files were uploaded.")
        
        task_id = str(uuid.uuid4())
        
        file_info = []
        for file in files:
            content = await file.read()
            file_info.append({
                "filename": file.filename,
                "content": content
            })
        
        indexing_tasks[task_id] = "queued"
        
        background_tasks.add_task(
            process_and_ingest_files_background,
            file_info,
            vector_store,
            config_manager,
            task_id,
            indexing_tasks,
            collection,  # NEW
            doc_type
        )
        
        response = {
            "message": f"Files queued for processing. Indexing {len(files)} files in the background.",
            "files": [file.filename for file in files],
            "status": "queued",
            "task_id": task_id,
            "collection": collection or vector_store.default_collection_name(),
        }
        
        log_response(response, "/ingest")
        return response
            
    except Exception as e:
        log_error(e, "/ingest")
        raise HTTPException(
            status_code=500,
            detail=f"Error queuing files for ingestion: {str(e)}"
        )


@app.get("/ingest/status/{task_id}")
async def get_indexing_status(task_id: str):
    """Get the status of a file ingestion task."""
    if task_id in indexing_tasks:
        return {"status": indexing_tasks[task_id]}
    else:
        raise HTTPException(status_code=404, detail="Task not found")

# ---------------------------------------------------------------------------------------
# Collection inspection + search (per-company)
# ---------------------------------------------------------------------------------------

@app.get("/collections/{collection}/info")
async def get_collection_info(collection: str, sample: int = 3):
    """Return existence, num_entities, schema fields, index summary, and a small sample."""
    coll = "".join(c if c.isalnum() or c == "_" else "_" for c in collection)
    try:
        connections.connect(uri=vector_store.uri)
        exists = utility.has_collection(coll)
        if not exists:
            return {"collection": coll, "exists": False}

        c = Collection(coll)
        c.load()
        info = {
            "collection": coll,
            "exists": True,
            "num_entities": c.num_entities,
            "schema_fields": [
                {"name": f.name, "dtype": str(f.dtype), "is_primary": getattr(f, "is_primary", False)}
                for f in c.schema.fields
            ],
            "indexes": [
                {"field": idx.field_name, "index_name": idx.index_name, "params": idx.params}
                for idx in c.indexes
            ],
        }

        # Try a tiny sample (adjust output_fields to your schema)
        try:
            rows = c.query(
                expr="pk >= 0",
                limit=max(0, sample),
                output_fields=["text", "source", "chunk_index", "chunk_total"]
            )
            info["sample"] = rows
        except Exception:
            info["sample"] = []

        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading collection '{coll}': {e}")
    finally:
        try:
            connections.disconnect("default")
        except Exception:
            pass


@app.get("/collections/{collection}/search")
async def search_collection(collection: str, query: str, k: int = 5):
    """Semantic search in a specific collection using the configured embeddings."""
    try:
        docs = vector_store.get_documents(query=query, k=k, collection_name=collection)
        results = []
        for d in docs:
            results.append({
                "text": d.page_content,
                "metadata": d.metadata,
            })
        return {"collection": collection, "query": query, "k": k, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching in '{collection}': {e}")

# ---------------------------------------------------------------------------------------
# Chat + config endpoints (unchanged)
# ---------------------------------------------------------------------------------------

@app.get("/sources")
async def get_sources():
    """Get all available document sources."""
    try:
        config = config_manager.read_config()
        return {"sources": config.sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sources: {str(e)}")


@app.get("/selected_sources")
async def get_selected_sources():
    """Get currently selected document sources for RAG."""
    try:
        config = config_manager.read_config()
        return {"sources": config.selected_sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting selected sources: {str(e)}")


@app.post("/selected_sources")
async def update_selected_sources(selected_sources: List[str]):
    """Update the selected document sources for RAG."""
    try:
        config_manager.updated_selected_sources(selected_sources)
        return {"status": "success", "message": "Selected sources updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating selected sources: {str(e)}")


@app.get("/selected_model")
async def get_selected_model():
    """Get the currently selected LLM model."""
    try:
        model = config_manager.get_selected_model()
        return {"model": model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting selected model: {str(e)}")


@app.post("/selected_model")
async def update_selected_model(request: SelectedModelRequest):
    """Update the selected LLM model."""
    try:
        logger.debug(f"Updating selected model to: {request.model}")
        config_manager.updated_selected_model(request.model)
        return {"status": "success", "message": "Selected model updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating selected model: {str(e)}")


@app.get("/available_models")
async def get_available_models():
    """Get list of all available LLM models."""
    try:
        models = config_manager.get_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting available models: {str(e)}")


@app.get("/chats")
async def list_chats():
    """Get list of all chat conversations."""
    try:
        chat_ids = await postgres_storage.list_conversations()
        return {"chats": chat_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing chats: {str(e)}")


@app.get("/chat_id")
async def get_chat_id():
    """Get the current active chat ID, creating a conversation if it doesn't exist."""
    try:
        config = config_manager.read_config()
        current_chat_id = config.current_chat_id
        
        if current_chat_id and await postgres_storage.exists(current_chat_id):
            return {
                "status": "success",
                "chat_id": current_chat_id
            }
        
        new_chat_id = str(uuid.uuid4())
        
        await postgres_storage.save_messages_immediate(new_chat_id, [])
        await postgres_storage.set_chat_metadata(new_chat_id, f"Chat {new_chat_id[:8]}")
        
        config_manager.updated_current_chat_id(new_chat_id)
        
        return {
            "status": "success",
            "chat_id": new_chat_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting chat ID: {str(e)}"
        )


@app.post("/chat_id")
async def update_chat_id(request: ChatIdRequest):
    """Update the current active chat ID."""
    try:
        config_manager.updated_current_chat_id(request.chat_id)
        return {
            "status": "success",
            "message": f"Current chat ID updated to {request.chat_id}",
            "chat_id": request.chat_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating chat ID: {str(e)}"
        )


@app.get("/chat/{chat_id}/metadata")
async def get_chat_metadata(chat_id: str):
    """Get metadata for a specific chat."""
    try:
        metadata = await postgres_storage.get_chat_metadata(chat_id)
        return metadata
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting chat metadata: {str(e)}"
        )


@app.post("/chat/rename")
async def rename_chat(request: ChatRenameRequest):
    """Rename a chat conversation."""
    try:
        await postgres_storage.set_chat_metadata(request.chat_id, request.new_name)
        return {
            "status": "success",
            "message": f"Chat {request.chat_id} renamed to {request.new_name}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error renaming chat: {str(e)}"
        )


@app.post("/chat/new")
async def create_new_chat():
    """Create a new chat conversation and set it as current."""
    try:
        new_chat_id = str(uuid.uuid4())
        await postgres_storage.save_messages_immediate(new_chat_id, [])
        await postgres_storage.set_chat_metadata(new_chat_id, f"Chat {new_chat_id[:8]}")
        
        config_manager.updated_current_chat_id(new_chat_id)
        
        return {
            "status": "success",
            "message": "New chat created",
            "chat_id": new_chat_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating new chat: {str(e)}"
        )


@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a specific chat and its messages."""
    try:
        success = await postgres_storage.delete_conversation(chat_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Chat {chat_id} deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Chat {chat_id} not found"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting chat: {str(e)}"
        )


@app.delete("/chats/clear")
async def clear_all_chats():
    """Clear all chat conversations and create a new default chat."""
    try:
        chat_ids = await postgres_storage.list_conversations()
        cleared_count = 0
        
        for chat_id in chat_ids:
            if await postgres_storage.delete_conversation(chat_id):
                cleared_count += 1
        
        new_chat_id = str(uuid.uuid4())
        await postgres_storage.save_messages_immediate(new_chat_id, [])
        await postgres_storage.set_chat_metadata(new_chat_id, f"Chat {new_chat_id[:8]}")
        
        config_manager.updated_current_chat_id(new_chat_id)
        
        return {
            "status": "success",
            "message": f"Cleared {cleared_count} chats and created new chat",
            "new_chat_id": new_chat_id,
            "cleared_count": cleared_count
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing all chats: {str(e)}"
        )


@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a document collection from the vector store."""
    try:
        success = vector_store.delete_collection(collection_name)
        if success:
            return {"status": "success", "message": f"Collection '{collection_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found or could not be deleted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")


@app.websocket("/ws/chat/{chat_id}")
async def websocket_endpoint(websocket: WebSocket, chat_id: str):
    """WebSocket endpoint for real-time chat communication."""
    logger.debug(f"WebSocket connection attempt for chat_id: {chat_id}")
    try:
        await websocket.accept()
        logger.debug(f"WebSocket connection accepted for chat_id: {chat_id}")
        
        history_messages = await postgres_storage.get_messages(chat_id)
        history = [postgres_storage._message_to_dict(msg) for i, msg in enumerate(history_messages) if i != 0]
        await websocket.send_json({"type": "history", "messages": history})
        
        while True:
            data = await websocket.receive_text()
            client_message = json.loads(data)
            new_message = client_message.get("message")
            image_id = client_message.get("image_id")
            
            image_data = None
            if image_id:
                image_data = await postgres_storage.get_image(image_id)
                logger.debug(f"Retrieved image data for image_id: {image_id}, data length: {len(image_data) if image_data else 0}")
            
            try:
                async for event in agent.query(query_text=new_message, chat_id=chat_id, image_data=image_data):
                    await websocket.send_json(event)
            except Exception as query_error:
                logger.error(f"Error in agent.query: {str(query_error)}", exc_info=True)
                await websocket.send_json({"type": "error", "content": f"Error processing request: {str(query_error)}"})
        
            final_messages = await postgres_storage.get_messages(chat_id)
            final_history = [postgres_storage._message_to_dict(msg) for i, msg in enumerate(final_messages) if i != 0]
            await websocket.send_json({"type": "history", "messages": final_history})
            
    except WebSocketDisconnect:
        logger.debug(f"Client disconnected from chat {chat_id}")
    except Exception as e:
        logger.error(f"WebSocket error for chat {chat_id}: {str(e)}", exc_info=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
