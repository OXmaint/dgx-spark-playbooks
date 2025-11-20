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

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agent import ChatAgent
from batch_agent import BatchAnalysisAgent
from config import ConfigManager
from logger import logger, log_request, log_response, log_error
from models import ChatIdRequest, ChatRenameRequest, SelectedModelRequest, ImageDescriptionRequest, BatchAnalysisRequest
from postgres_storage import PostgreSQLConversationStorage
from report_generator import ReportGenerator
from utils import process_and_ingest_files_background, process_batch_analysis_background
from vector_store import create_vector_store_with_config

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

agent: ChatAgent | None = None
batch_agent: BatchAnalysisAgent | None = None
report_generator: ReportGenerator | None = None
indexing_tasks: Dict[str, str] = {}
batch_tasks: Dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks."""
    global agent, batch_agent, report_generator
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

        # Initialize batch analysis agent
        logger.debug("Initializing BatchAnalysisAgent...")
        batch_agent = BatchAnalysisAgent(
            vector_store=vector_store,
            config_manager=config_manager,
            postgres_storage=postgres_storage
        )
        await batch_agent.init_tools()
        logger.info("BatchAnalysisAgent initialized successfully.")

        # Initialize report generator
        report_generator = ReportGenerator(postgres_storage)
        logger.info("ReportGenerator initialized successfully.")

    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL storage: {e}")
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


@app.websocket("/ws/chat/{chat_id}")
async def websocket_endpoint(websocket: WebSocket, chat_id: str):
    """WebSocket endpoint for real-time chat communication.
    
    Args:
        websocket: WebSocket connection
        chat_id: Unique chat identifier
    """
    logger.debug(f"WebSocket connection attempt for chat_id: {chat_id}")
    try:
        await websocket.accept()
        logger.debug(f"WebSocket connection accepted for chat_id: {chat_id}")

        history_messages = await postgres_storage.get_messages(chat_id)
        history = [postgres_storage._message_to_dict(msg) for i, msg in enumerate(history_messages) if i != 0]

        # Resolve image attachments - fetch actual image data for each attachment
        for msg in history:
            if msg.get('attachments'):
                for attachment in msg['attachments']:
                    if attachment.get('type') == 'image' and attachment.get('image_id'):
                        image_id = attachment['image_id']
                        image_data = await postgres_storage.get_image(image_id)
                        if image_data:
                            # Add resolved image data to the message
                            msg['image'] = image_data
                            logger.debug(f"Resolved image attachment: {image_id}")

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
            
            has_image_response = False
            try:
                async for event in agent.query(query_text=new_message, chat_id=chat_id, image_data=image_data):
                    # Log event details for debugging
                    if isinstance(event, dict):
                        event_type = event.get('type', 'unknown')
                        # Check for nested final_response in token events
                        if event_type == 'token' and isinstance(event.get('data'), dict):
                            nested_data = event.get('data', {})
                            if nested_data.get('type') == 'final_response':
                                logger.info(f"[WEBSOCKET] Sending token with nested final_response")
                                logger.info(f"[WEBSOCKET] Nested data has 'text': {'text' in nested_data}, length: {len(nested_data.get('text', ''))}")
                                logger.info(f"[WEBSOCKET] Nested data has 'image': {'image' in nested_data}, length: {len(nested_data.get('image', ''))}")
                                # Mark that we sent an image response
                                if nested_data.get('image'):
                                    has_image_response = True
                            else:
                                logger.debug(f"[WEBSOCKET] Sending event type: {event_type}")
                        else:
                            logger.debug(f"[WEBSOCKET] Sending event type: {event_type}")
                    else:
                        logger.debug(f"[WEBSOCKET] Sending non-dict event: {type(event)}, length: {len(str(event))}")
                    await websocket.send_json(event)
            except Exception as query_error:
                logger.error(f"Error in agent.query: {str(query_error)}", exc_info=True)
                await websocket.send_json({"type": "error", "content": f"Error processing request: {str(query_error)}"})

            # Don't send history update if we sent an image response
            # (the history from DB doesn't include the image, it would overwrite the frontend state)
            if not has_image_response:
                final_messages = await postgres_storage.get_messages(chat_id)
                final_history = [postgres_storage._message_to_dict(msg) for i, msg in enumerate(final_messages) if i != 0]
                await websocket.send_json({"type": "history", "messages": final_history})
            
    except WebSocketDisconnect:
        logger.debug(f"Client disconnected from chat {chat_id}")
    except Exception as e:
        logger.error(f"WebSocket error for chat {chat_id}: {str(e)}", exc_info=True)


@app.post("/upload-image")
async def upload_image(image: UploadFile = File(...), chat_id: str = Form(...)):
    """Upload and store an image for chat processing with metadata.

    Args:
        image: Uploaded image file
        chat_id: Chat identifier for context

    Returns:
        Dictionary with generated image_id and metadata
    """
    image_data = await image.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    data_uri = f"data:{image.content_type};base64,{image_base64}"
    image_id = str(uuid.uuid4())

    # Store image with metadata for context awareness
    await postgres_storage.store_image_with_metadata(
        image_id=image_id,
        image_base64=data_uri,
        chat_id=chat_id,
        filename=image.filename,
        content_type=image.content_type,
        persistent=True  # Chat-associated images don't expire
    )

    logger.debug(f"Image uploaded: {image_id}, filename: {image.filename}, chat_id: {chat_id}")

    return {
        "image_id": image_id,
        "filename": image.filename,
        "content_type": image.content_type,
        "chat_id": chat_id
    }


@app.post("/ingest")
async def ingest_files(files: Optional[List[UploadFile]] = File(None), background_tasks: BackgroundTasks = None):
    """Ingest documents for vector search and RAG.
    
    Args:
        files: List of uploaded files to process
        background_tasks: FastAPI background tasks manager
        
    Returns:
        Task information for tracking ingestion progress
    """
    try:
        log_request({"file_count": len(files) if files else 0}, "/ingest")
        
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
            indexing_tasks
        )
        
        response = {
            "message": f"Files queued for processing. Indexing {len(files)} files in the background.",
            "files": [file.filename for file in files],
            "status": "queued",
            "task_id": task_id
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
    """Get the status of a file ingestion task.
    
    Args:
        task_id: Unique task identifier
        
    Returns:
        Current task status
    """
    if task_id in indexing_tasks:
        return {"status": indexing_tasks[task_id]}
    else:
        raise HTTPException(status_code=404, detail="Task not found")


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
    """Update the selected document sources for RAG.
    
    Args:
        selected_sources: List of source names to use for retrieval
    """
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
    """Update the selected LLM model.
    
    Args:
        request: Model selection request with model name
    """
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
    """Update the current active chat ID.
    
    Args:
        request: Chat ID update request
    """
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
    """Get metadata for a specific chat.
    
    Args:
        chat_id: Unique chat identifier
        
    Returns:
        Chat metadata including name
    """
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
    """Rename a chat conversation.
    
    Args:
        request: Chat rename request with chat_id and new_name
    """
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
    """Delete a specific chat and its messages.
    
    Args:
        chat_id: Unique chat identifier to delete
    """
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
    """Delete a document collection from the vector store.

    Args:
        collection_name: Name of the collection to delete
    """
    try:
        success = vector_store.delete_collection(collection_name)
        if success:
            return {"status": "success", "message": f"Collection '{collection_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found or could not be deleted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")


# Image management endpoints

@app.get("/images")
async def list_images(limit: int = 50):
    """List all available images with metadata.

    Args:
        limit: Maximum number of images to return (default 50)

    Returns:
        List of image metadata
    """
    try:
        images = await postgres_storage.list_images(limit=limit)
        return {"images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing images: {str(e)}")


@app.get("/chat/{chat_id}/images")
async def list_chat_images(chat_id: str):
    """List all images associated with a specific chat.

    Args:
        chat_id: Chat identifier

    Returns:
        List of image metadata for the chat
    """
    try:
        images = await postgres_storage.list_images_for_chat(chat_id)
        return {"images": images, "chat_id": chat_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing chat images: {str(e)}")


@app.get("/image/{image_id}")
async def get_image_data(image_id: str):
    """Get image data and metadata by ID.

    Args:
        image_id: Unique image identifier

    Returns:
        Image data and metadata
    """
    try:
        image_data = await postgres_storage.get_image(image_id)
        metadata = await postgres_storage.get_image_metadata(image_id)

        if not image_data:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found or expired")

        return {
            "image_id": image_id,
            "image_data": image_data,
            "metadata": metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting image: {str(e)}")


@app.get("/image/{image_id}/metadata")
async def get_image_metadata(image_id: str):
    """Get metadata for a specific image.

    Args:
        image_id: Unique image identifier

    Returns:
        Image metadata
    """
    try:
        metadata = await postgres_storage.get_image_metadata(image_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Image metadata for {image_id} not found")
        return metadata
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting image metadata: {str(e)}")


@app.post("/image/{image_id}/description")
async def update_image_description(image_id: str, request: ImageDescriptionRequest):
    """Update the description of an image.

    Args:
        image_id: Unique image identifier
        request: Request containing the new description

    Returns:
        Success status
    """
    try:
        success = await postgres_storage.update_image_description(image_id, request.description)
        if success:
            return {"status": "success", "message": f"Description updated for image {image_id}"}
        else:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating image description: {str(e)}")


@app.delete("/image/{image_id}")
async def delete_image(image_id: str):
    """Delete an image and its metadata.

    Args:
        image_id: Unique image identifier

    Returns:
        Success status
    """
    try:
        success = await postgres_storage.delete_image(image_id)
        if success:
            return {"status": "success", "message": f"Image {image_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting image: {str(e)}")


# Batch Analysis Endpoints

@app.post("/batch-analyze")
async def create_batch_analysis(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Create a new batch image analysis job.

    Args:
        request: Batch analysis request with image IDs and prompt
        background_tasks: FastAPI background tasks manager

    Returns:
        Batch job information for tracking
    """
    try:
        if not request.image_ids:
            raise HTTPException(status_code=400, detail="No image IDs provided")

        if len(request.image_ids) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 images per batch")

        batch_id = str(uuid.uuid4())

        # Queue the batch processing task
        batch_tasks[batch_id] = "queued"

        background_tasks.add_task(
            process_batch_analysis_background,
            batch_id,
            request.image_ids,
            request.analysis_prompt,
            request.report_format,
            batch_agent,
            batch_tasks,
            request.organization
        )

        logger.info(f"Batch analysis queued: {batch_id} with {len(request.image_ids)} images")

        return {
            "batch_id": batch_id,
            "status": "queued",
            "total_images": len(request.image_ids),
            "analysis_prompt": request.analysis_prompt,
            "report_format": request.report_format
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating batch analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating batch analysis: {str(e)}")


@app.post("/batch-analyze/upload")
async def create_batch_analysis_with_upload(
    files: List[UploadFile] = File(...),
    analysis_prompt: str = Form(...),
    report_format: str = Form("markdown"),
    organization: str = Form(None),
    background_tasks: BackgroundTasks = None
):
    """Create a batch analysis job by uploading images directly.

    Args:
        files: List of image files to analyze
        analysis_prompt: The prompt for analysis
        report_format: Output format (markdown, html)
        background_tasks: FastAPI background tasks manager

    Returns:
        Batch job information for tracking
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        if len(files) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 images per batch")

        # Upload and store all images first
        image_ids = []
        for file in files:
            image_data = await file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            data_uri = f"data:{file.content_type};base64,{image_base64}"
            image_id = str(uuid.uuid4())

            await postgres_storage.store_image_with_metadata(
                image_id=image_id,
                image_base64=data_uri,
                chat_id=None,  # Not associated with a chat
                filename=file.filename,
                content_type=file.content_type,
                persistent=True
            )
            image_ids.append(image_id)

        batch_id = str(uuid.uuid4())

        # Queue the batch processing task
        batch_tasks[batch_id] = "queued"

        background_tasks.add_task(
            process_batch_analysis_background,
            batch_id,
            image_ids,
            analysis_prompt,
            report_format,
            batch_agent,
            batch_tasks,
            organization
        )

        logger.info(f"Batch analysis with upload queued: {batch_id} with {len(image_ids)} images")

        return {
            "batch_id": batch_id,
            "status": "queued",
            "total_images": len(image_ids),
            "image_ids": image_ids,
            "analysis_prompt": analysis_prompt,
            "report_format": report_format
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating batch analysis with upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating batch analysis: {str(e)}")


@app.get("/batch-analyze/{batch_id}/status")
async def get_batch_status(batch_id: str):
    """Get the status of a batch analysis job.

    Args:
        batch_id: Batch identifier

    Returns:
        Current batch job status and progress
    """
    try:
        # Get from database for detailed status
        batch_job = await postgres_storage.get_batch_job(batch_id)

        if not batch_job:
            # Check in-memory tasks
            if batch_id in batch_tasks:
                return {"batch_id": batch_id, "status": batch_tasks[batch_id]}
            raise HTTPException(status_code=404, detail=f"Batch job {batch_id} not found")

        return batch_job

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting batch status: {str(e)}")


@app.get("/batch-analyze/{batch_id}/results")
async def get_batch_results(batch_id: str):
    """Get the results of a completed batch analysis.

    Args:
        batch_id: Batch identifier

    Returns:
        List of analysis results for each image
    """
    try:
        batch_job = await postgres_storage.get_batch_job(batch_id)
        if not batch_job:
            raise HTTPException(status_code=404, detail=f"Batch job {batch_id} not found")

        results = await postgres_storage.get_batch_results(batch_id)

        return {
            "batch_id": batch_id,
            "status": batch_job["status"],
            "total_images": batch_job["total_images"],
            "processed_count": batch_job["processed_count"],
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting batch results: {str(e)}")


@app.get("/batch-analyze/{batch_id}/report")
async def get_batch_report(
    batch_id: str,
    format: str = "markdown",
    include_images: bool = True
):
    """Generate and return the analysis report for a batch job.

    Args:
        batch_id: Batch identifier
        format: Report format (markdown, html)
        include_images: Whether to embed images in the report

    Returns:
        Generated report content
    """
    try:
        batch_job = await postgres_storage.get_batch_job(batch_id)
        if not batch_job:
            raise HTTPException(status_code=404, detail=f"Batch job {batch_id} not found")

        if batch_job["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Batch job not completed. Current status: {batch_job['status']}"
            )

        report_content = await report_generator.generate_report(
            batch_id=batch_id,
            format=format,
            include_images=include_images
        )

        # Determine content type based on format
        if format == "html":
            content_type = "text/html"
        else:
            content_type = "text/markdown"

        from fastapi.responses import Response
        return Response(
            content=report_content,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=report_{batch_id[:8]}.{format if format != 'markdown' else 'md'}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


@app.get("/batch-analyze")
async def list_batch_jobs(limit: int = 50):
    """List all batch analysis jobs.

    Args:
        limit: Maximum number of jobs to return

    Returns:
        List of batch job summaries
    """
    try:
        jobs = await postgres_storage.list_batch_jobs(limit=limit)
        return {"batch_jobs": jobs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing batch jobs: {str(e)}")


@app.delete("/batch-analyze/{batch_id}")
async def delete_batch_job(batch_id: str):
    """Delete a batch analysis job and its results.

    Args:
        batch_id: Batch identifier

    Returns:
        Success status
    """
    try:
        success = await postgres_storage.delete_batch_job(batch_id)

        # Also clean up in-memory tracking
        batch_tasks.pop(batch_id, None)

        if success:
            return {"status": "success", "message": f"Batch job {batch_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Batch job {batch_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting batch job: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)