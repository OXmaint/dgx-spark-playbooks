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
"""PostgreSQL-based conversation storage with caching and I/O optimization."""

import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import asyncpg
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage

from logger import logger


@dataclass
class CacheEntry:
    """Cache entry with TTL support."""
    data: Any
    timestamp: float
    ttl: float = 300
    
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


class PostgreSQLConversationStorage:
    """PostgreSQL-based conversation storage with intelligent caching and I/O optimization."""
    
    def __init__(
        self, 
        host: str = 'postgres', 
        port: int = 5432, 
        database: str = 'chatbot', 
        user: str = 'chatbot_user', 
        password: str = 'chatbot_password',
        pool_size: int = 10,
        cache_ttl: int = 300
    ):
        """Initialize PostgreSQL connection pool and caching.
        
        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Database user
            password: Database password
            pool_size: Connection pool size
            cache_ttl: Cache TTL in seconds
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool_size = pool_size
        self.cache_ttl = cache_ttl
        
        self.pool: Optional[asyncpg.Pool] = None
        
        self._message_cache: Dict[str, CacheEntry] = {}
        self._metadata_cache: Dict[str, CacheEntry] = {}
        self._image_cache: Dict[str, CacheEntry] = {}
        self._chat_list_cache: Optional[CacheEntry] = None
        
        self._pending_saves: Dict[str, List[BaseMessage]] = {}
        self._save_lock = asyncio.Lock()
        self._batch_save_task: Optional[asyncio.Task] = None
        
        self._cache_hits = 0
        self._cache_misses = 0
        self._db_operations = 0

    async def init_pool(self) -> None:
        """Initialize the connection pool and create tables."""
        try:
            await self._ensure_database_exists()
            
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=2,
                max_size=self.pool_size,
                command_timeout=30
            )
            
            await self._create_tables()
            logger.debug("PostgreSQL connection pool initialized successfully")
            
            self._batch_save_task = asyncio.create_task(self._batch_save_worker())
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise

    async def _ensure_database_exists(self) -> None:
        """Ensure the target database exists, create if it doesn't."""
        try:
            conn = await asyncpg.connect(
                host=self.host,
                port=self.port,
                database='postgres',
                user=self.user,
                password=self.password
            )
            
            try:
                result = await conn.fetchval(
                    "SELECT 1 FROM pg_database WHERE datname = $1",
                    self.database
                )
                
                if not result:
                    await conn.execute(f'CREATE DATABASE "{self.database}"')
                    logger.debug(f"Created database: {self.database}")
                else:
                    logger.debug(f"Database {self.database} already exists")
                    
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"Error ensuring database exists: {e}")
            pass

    async def close(self) -> None:
        """Close the connection pool and cleanup."""
        if self._batch_save_task:
            self._batch_save_task.cancel()
            try:
                await self._batch_save_task
            except asyncio.CancelledError:
                pass
        
        if self.pool:
            await self.pool.close()
            logger.debug("PostgreSQL connection pool closed")

    async def _create_tables(self) -> None:
        """Create necessary tables if they don't exist."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    chat_id VARCHAR(255) PRIMARY KEY,
                    messages JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_metadata (
                    chat_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(500),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES conversations(chat_id) ON DELETE CASCADE
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    image_id VARCHAR(255) PRIMARY KEY,
                    image_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + INTERVAL '1 hour')
                )
            """)

            # Image metadata table for making model context-aware about available images
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS image_metadata (
                    image_id VARCHAR(255) PRIMARY KEY,
                    chat_id VARCHAR(255),
                    filename VARCHAR(500),
                    description TEXT,
                    content_type VARCHAR(100),
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
                )
            """)

            await conn.execute("CREATE INDEX IF NOT EXISTS idx_image_metadata_chat_id ON image_metadata(chat_id)")
            
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_images_expires_at ON images(expires_at)")

            # Batch analysis tables
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS batch_analysis_jobs (
                    batch_id VARCHAR(255) PRIMARY KEY,
                    status VARCHAR(50) NOT NULL DEFAULT 'queued',
                    total_images INTEGER NOT NULL DEFAULT 0,
                    processed_count INTEGER NOT NULL DEFAULT 0,
                    analysis_prompt TEXT,
                    report_format VARCHAR(50) DEFAULT 'markdown',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS batch_analysis_results (
                    result_id VARCHAR(255) PRIMARY KEY,
                    batch_id VARCHAR(255) NOT NULL,
                    image_id VARCHAR(255) NOT NULL,
                    image_filename VARCHAR(500),
                    analysis_result TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_message TEXT,
                    FOREIGN KEY (batch_id) REFERENCES batch_analysis_jobs(batch_id) ON DELETE CASCADE
                )
            """)

            await conn.execute("CREATE INDEX IF NOT EXISTS idx_batch_results_batch_id ON batch_analysis_results(batch_id)")
            
            await conn.execute("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql'
            """)
            
            await conn.execute("""
                DROP TRIGGER IF EXISTS update_conversations_updated_at ON conversations
            """)
            await conn.execute("""
                CREATE TRIGGER update_conversations_updated_at
                    BEFORE UPDATE ON conversations
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column()
            """)

    def _message_to_dict(self, message: BaseMessage) -> Dict:
        """Convert a message object to a dictionary for storage."""
        result = {
            "type": message.__class__.__name__,
            "content": message.content,
        }

        if hasattr(message, "tool_calls") and message.tool_calls:
            result["tool_calls"] = message.tool_calls

        if isinstance(message, ToolMessage):
            result["tool_call_id"] = getattr(message, "tool_call_id", None)
            result["name"] = getattr(message, "name", None)

        # Include attachments (image references) if present
        if hasattr(message, "additional_kwargs") and message.additional_kwargs.get("attachments"):
            result["attachments"] = message.additional_kwargs["attachments"]

        return result

    def _dict_to_message(self, data: Dict) -> BaseMessage:
        """Convert a dictionary back to a message object."""
        msg_type = data["type"]
        content = data["content"]
        
        if msg_type == "AIMessage":
            msg = AIMessage(content=content)
            if "tool_calls" in data:
                msg.tool_calls = data["tool_calls"]
            return msg
        elif msg_type == "HumanMessage":
            return HumanMessage(content=content)
        elif msg_type == "SystemMessage":
            return SystemMessage(content=content)
        elif msg_type == "ToolMessage":
            return ToolMessage(
                content=content,
                tool_call_id=data.get("tool_call_id", ""),
                name=data.get("name", "")
            )
        else:
            return HumanMessage(content=content)

    def _get_cached_messages(self, chat_id: str) -> Optional[List[BaseMessage]]:
        """Get messages from cache if available and not expired."""
        cache_entry = self._message_cache.get(chat_id)
        if cache_entry and not cache_entry.is_expired():
            self._cache_hits += 1
            return cache_entry.data
        
        self._cache_misses += 1
        return None

    def _cache_messages(self, chat_id: str, messages: List[BaseMessage]) -> None:
        """Cache messages with TTL."""
        self._message_cache[chat_id] = CacheEntry(
            data=messages.copy(),
            timestamp=time.time(),
            ttl=self.cache_ttl
        )

    def _invalidate_cache(self, chat_id: str) -> None:
        """Invalidate cache entries for a chat."""
        self._message_cache.pop(chat_id, None)
        self._metadata_cache.pop(chat_id, None)
        self._chat_list_cache = None

    async def exists(self, chat_id: str) -> bool:
        """Check if a conversation exists (with caching)."""
        cached_messages = self._get_cached_messages(chat_id)
        if cached_messages is not None:
            return len(cached_messages) > 0
        
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM conversations WHERE chat_id = $1)",
                chat_id
            )
            self._db_operations += 1
            return result

    async def get_messages(self, chat_id: str, limit: Optional[int] = None) -> List[BaseMessage]:
        """Retrieve messages for a chat session with caching."""
        cached_messages = self._get_cached_messages(chat_id)
        if cached_messages is not None:
            return cached_messages[-limit:] if limit else cached_messages
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT messages FROM conversations WHERE chat_id = $1",
                chat_id
            )
            self._db_operations += 1
            
            if not row:
                return []
            
            messages_data = row['messages']
            if isinstance(messages_data, str):
                messages_data = json.loads(messages_data)
            messages = [self._dict_to_message(msg_data) for msg_data in messages_data]
            
            self._cache_messages(chat_id, messages)
            
            return messages[-limit:] if limit else messages

    async def save_messages(self, chat_id: str, messages: List[BaseMessage]) -> None:
        """Save messages with batching for performance."""
        async with self._save_lock:
            self._pending_saves[chat_id] = messages.copy()
        
        self._cache_messages(chat_id, messages)
    
    async def save_messages_immediate(self, chat_id: str, messages: List[BaseMessage]) -> None:
        """Save messages immediately without batching - for critical operations."""
        serialized_messages = [self._message_to_dict(msg) for msg in messages]
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO conversations (chat_id, messages, message_count)
                VALUES ($1, $2, $3)
                ON CONFLICT (chat_id)
                DO UPDATE SET 
                    messages = EXCLUDED.messages,
                    message_count = EXCLUDED.message_count,
                    updated_at = CURRENT_TIMESTAMP
            """, chat_id, json.dumps(serialized_messages), len(messages))
            self._db_operations += 1
        
        self._cache_messages(chat_id, messages)
        self._chat_list_cache = None

    async def _batch_save_worker(self) -> None:
        """Background worker to batch save operations."""
        while True:
            try:
                await asyncio.sleep(1.0)
                
                async with self._save_lock:
                    if not self._pending_saves:
                        continue
                    
                    saves_to_process = self._pending_saves.copy()
                    self._pending_saves.clear()
                
                async with self.pool.acquire() as conn:
                    async with conn.transaction():
                        for chat_id, messages in saves_to_process.items():
                            serialized_messages = [self._message_to_dict(msg) for msg in messages]
                            
                            await conn.execute("""
                                INSERT INTO conversations (chat_id, messages, message_count)
                                VALUES ($1, $2, $3)
                                ON CONFLICT (chat_id)
                                DO UPDATE SET 
                                    messages = EXCLUDED.messages,
                                    message_count = EXCLUDED.message_count,
                                    updated_at = CURRENT_TIMESTAMP
                            """, chat_id, json.dumps(serialized_messages), len(messages))
                
                self._db_operations += len(saves_to_process)
                if saves_to_process:
                    logger.debug(f"Batch saved {len(saves_to_process)} conversations")
                    self._chat_list_cache = None
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch save worker: {e}")

    async def add_message(self, chat_id: str, message: BaseMessage) -> None:
        """Add a single message to conversation (optimized)."""
        current_messages = await self.get_messages(chat_id)
        current_messages.append(message)
        
        await self.save_messages(chat_id, current_messages)

    async def delete_conversation(self, chat_id: str) -> bool:
        """Delete a conversation by chat_id."""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM conversations WHERE chat_id = $1",
                    chat_id
                )
                self._db_operations += 1
                
                self._invalidate_cache(chat_id)
                
                return "DELETE 1" in result
        except Exception as e:
            logger.error(f"Error deleting conversation {chat_id}: {e}")
            return False

    async def list_conversations(self) -> List[str]:
        """List all conversation IDs with caching."""
        if self._chat_list_cache and not self._chat_list_cache.is_expired():
            self._cache_hits += 1
            return self._chat_list_cache.data
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT chat_id FROM conversations ORDER BY updated_at DESC"
            )
            self._db_operations += 1
            
            chat_ids = [row['chat_id'] for row in rows]
            
            self._chat_list_cache = CacheEntry(
                data=chat_ids,
                timestamp=time.time(),
                ttl=60
            )
            self._cache_misses += 1
            
            return chat_ids

    async def store_image(self, image_id: str, image_base64: str) -> None:
        """Store base64 image data with TTL."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO images (image_id, image_data)
                VALUES ($1, $2)
                ON CONFLICT (image_id)
                DO UPDATE SET 
                    image_data = EXCLUDED.image_data,
                    created_at = CURRENT_TIMESTAMP,
                    expires_at = CURRENT_TIMESTAMP + INTERVAL '1 hour'
            """, image_id, image_base64)
            self._db_operations += 1
        
        self._image_cache[image_id] = CacheEntry(
            data=image_base64,
            timestamp=time.time(),
            ttl=3600
        )

    async def get_image(self, image_id: str) -> Optional[str]:
        """Retrieve base64 image data with caching."""
        cache_entry = self._image_cache.get(image_id)
        if cache_entry and not cache_entry.is_expired():
            self._cache_hits += 1
            return cache_entry.data

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT image_data FROM images
                WHERE image_id = $1 AND expires_at > CURRENT_TIMESTAMP
            """, image_id)
            self._db_operations += 1

            if row:
                image_data = row['image_data']
                self._image_cache[image_id] = CacheEntry(
                    data=image_data,
                    timestamp=time.time(),
                    ttl=3600
                )
                self._cache_misses += 1
                return image_data

            return None

    async def store_image_with_metadata(
        self,
        image_id: str,
        image_base64: str,
        chat_id: str = None,
        filename: str = None,
        content_type: str = None,
        persistent: bool = False
    ) -> None:
        """Store base64 image data with metadata for context awareness.

        Args:
            image_id: Unique image identifier
            image_base64: Base64 encoded image data
            chat_id: Associated chat identifier
            filename: Original filename
            content_type: MIME type of the image
            persistent: If True, image won't expire (for chat-associated images)
        """
        async with self.pool.acquire() as conn:
            # Store image data with optional persistence
            if persistent:
                await conn.execute("""
                    INSERT INTO images (image_id, image_data, expires_at)
                    VALUES ($1, $2, NULL)
                    ON CONFLICT (image_id)
                    DO UPDATE SET
                        image_data = EXCLUDED.image_data,
                        created_at = CURRENT_TIMESTAMP,
                        expires_at = NULL
                """, image_id, image_base64)
            else:
                await conn.execute("""
                    INSERT INTO images (image_id, image_data)
                    VALUES ($1, $2)
                    ON CONFLICT (image_id)
                    DO UPDATE SET
                        image_data = EXCLUDED.image_data,
                        created_at = CURRENT_TIMESTAMP,
                        expires_at = CURRENT_TIMESTAMP + INTERVAL '1 hour'
                """, image_id, image_base64)

            # Store metadata
            await conn.execute("""
                INSERT INTO image_metadata (image_id, chat_id, filename, content_type)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (image_id)
                DO UPDATE SET
                    chat_id = EXCLUDED.chat_id,
                    filename = EXCLUDED.filename,
                    content_type = EXCLUDED.content_type,
                    uploaded_at = CURRENT_TIMESTAMP
            """, image_id, chat_id, filename, content_type)

            self._db_operations += 2

        # Cache the image data
        self._image_cache[image_id] = CacheEntry(
            data=image_base64,
            timestamp=time.time(),
            ttl=3600 if not persistent else 86400  # 24 hours for persistent
        )

    async def get_image_metadata(self, image_id: str) -> Optional[Dict]:
        """Get metadata for a specific image.

        Args:
            image_id: Unique image identifier

        Returns:
            Dictionary with image metadata or None if not found
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT image_id, chat_id, filename, description, content_type, uploaded_at, is_active
                FROM image_metadata
                WHERE image_id = $1
            """, image_id)
            self._db_operations += 1

            if row:
                return {
                    "image_id": row['image_id'],
                    "chat_id": row['chat_id'],
                    "filename": row['filename'],
                    "description": row['description'],
                    "content_type": row['content_type'],
                    "uploaded_at": row['uploaded_at'].isoformat() if row['uploaded_at'] else None,
                    "is_active": row['is_active']
                }
            return None

    async def update_image_description(self, image_id: str, description: str) -> bool:
        """Update the description for an image.

        Args:
            image_id: Unique image identifier
            description: New description text

        Returns:
            True if updated successfully, False otherwise
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE image_metadata
                SET description = $2
                WHERE image_id = $1
            """, image_id, description)
            self._db_operations += 1
            return "UPDATE 1" in result

    async def list_images(self, limit: int = 50) -> List[Dict]:
        """List all active images with metadata.

        Args:
            limit: Maximum number of images to return

        Returns:
            List of image metadata dictionaries
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT m.image_id, m.chat_id, m.filename, m.description, m.content_type, m.uploaded_at, m.is_active
                FROM image_metadata m
                INNER JOIN images i ON m.image_id = i.image_id
                WHERE m.is_active = TRUE
                AND (i.expires_at IS NULL OR i.expires_at > CURRENT_TIMESTAMP)
                ORDER BY m.uploaded_at DESC
                LIMIT $1
            """, limit)
            self._db_operations += 1

            return [
                {
                    "image_id": row['image_id'],
                    "chat_id": row['chat_id'],
                    "filename": row['filename'],
                    "description": row['description'],
                    "content_type": row['content_type'],
                    "uploaded_at": row['uploaded_at'].isoformat() if row['uploaded_at'] else None,
                    "is_active": row['is_active']
                }
                for row in rows
            ]

    async def list_images_for_chat(self, chat_id: str) -> List[Dict]:
        """List all active images for a specific chat.

        Args:
            chat_id: Chat identifier

        Returns:
            List of image metadata dictionaries for the chat
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT m.image_id, m.chat_id, m.filename, m.description, m.content_type, m.uploaded_at, m.is_active
                FROM image_metadata m
                INNER JOIN images i ON m.image_id = i.image_id
                WHERE m.chat_id = $1
                AND m.is_active = TRUE
                AND (i.expires_at IS NULL OR i.expires_at > CURRENT_TIMESTAMP)
                ORDER BY m.uploaded_at DESC
            """, chat_id)
            self._db_operations += 1

            return [
                {
                    "image_id": row['image_id'],
                    "chat_id": row['chat_id'],
                    "filename": row['filename'],
                    "description": row['description'],
                    "content_type": row['content_type'],
                    "uploaded_at": row['uploaded_at'].isoformat() if row['uploaded_at'] else None,
                    "is_active": row['is_active']
                }
                for row in rows
            ]

    async def delete_image(self, image_id: str) -> bool:
        """Delete an image and its metadata.

        Args:
            image_id: Unique image identifier

        Returns:
            True if deleted successfully, False otherwise
        """
        async with self.pool.acquire() as conn:
            # Due to CASCADE, deleting from images will also delete from image_metadata
            result = await conn.execute(
                "DELETE FROM images WHERE image_id = $1",
                image_id
            )
            self._db_operations += 1

            # Clear from cache
            self._image_cache.pop(image_id, None)

            return "DELETE 1" in result

    async def deactivate_image(self, image_id: str) -> bool:
        """Soft delete an image by marking it inactive.

        Args:
            image_id: Unique image identifier

        Returns:
            True if deactivated successfully, False otherwise
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE image_metadata
                SET is_active = FALSE
                WHERE image_id = $1
            """, image_id)
            self._db_operations += 1
            return "UPDATE 1" in result

    async def search_images_for_chat(self, chat_id: str, search_term: str) -> List[Dict]:
        """Search images in a chat by filename or description.

        Args:
            chat_id: Chat identifier
            search_term: Term to search for in filename or description

        Returns:
            List of matching image metadata dictionaries
        """
        async with self.pool.acquire() as conn:
            # Case-insensitive search in filename and description
            rows = await conn.fetch("""
                SELECT m.image_id, m.chat_id, m.filename, m.description, m.content_type, m.uploaded_at, m.is_active
                FROM image_metadata m
                INNER JOIN images i ON m.image_id = i.image_id
                WHERE m.chat_id = $1
                AND m.is_active = TRUE
                AND (i.expires_at IS NULL OR i.expires_at > CURRENT_TIMESTAMP)
                AND (
                    LOWER(m.filename) LIKE LOWER($2)
                    OR LOWER(m.description) LIKE LOWER($2)
                )
                ORDER BY m.uploaded_at DESC
            """, chat_id, f"%{search_term}%")
            self._db_operations += 1

            return [
                {
                    "image_id": row['image_id'],
                    "chat_id": row['chat_id'],
                    "filename": row['filename'],
                    "description": row['description'],
                    "content_type": row['content_type'],
                    "uploaded_at": row['uploaded_at'].isoformat() if row['uploaded_at'] else None,
                    "is_active": row['is_active']
                }
                for row in rows
            ]

    async def get_image_by_filename(self, chat_id: str, filename: str) -> Optional[Dict]:
        """Get image metadata by exact filename match within a chat.

        Args:
            chat_id: Chat identifier
            filename: Exact filename to match

        Returns:
            Image metadata dictionary or None if not found
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT m.image_id, m.chat_id, m.filename, m.description, m.content_type, m.uploaded_at, m.is_active
                FROM image_metadata m
                INNER JOIN images i ON m.image_id = i.image_id
                WHERE m.chat_id = $1
                AND m.filename = $2
                AND m.is_active = TRUE
                AND (i.expires_at IS NULL OR i.expires_at > CURRENT_TIMESTAMP)
            """, chat_id, filename)
            self._db_operations += 1

            if row:
                return {
                    "image_id": row['image_id'],
                    "chat_id": row['chat_id'],
                    "filename": row['filename'],
                    "description": row['description'],
                    "content_type": row['content_type'],
                    "uploaded_at": row['uploaded_at'].isoformat() if row['uploaded_at'] else None,
                    "is_active": row['is_active']
                }
            return None

    async def get_chat_metadata(self, chat_id: str) -> Optional[Dict]:
        """Get chat metadata with caching."""
        cache_entry = self._metadata_cache.get(chat_id)
        if cache_entry and not cache_entry.is_expired():
            self._cache_hits += 1
            return cache_entry.data
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT name, created_at FROM chat_metadata WHERE chat_id = $1",
                chat_id
            )
            self._db_operations += 1
            
            if row:
                metadata = {
                    "name": row['name'],
                    "created_at": row['created_at'].isoformat()
                }
            else:
                metadata = {"name": f"Chat {chat_id[:8]}"}
            
            self._metadata_cache[chat_id] = CacheEntry(
                data=metadata,
                timestamp=time.time(),
                ttl=self.cache_ttl
            )
            self._cache_misses += 1
            
            return metadata

    async def set_chat_metadata(self, chat_id: str, name: str) -> None:
        """Set chat metadata."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO chat_metadata (chat_id, name)
                VALUES ($1, $2)
                ON CONFLICT (chat_id)
                DO UPDATE SET 
                    name = EXCLUDED.name,
                    updated_at = CURRENT_TIMESTAMP
            """, chat_id, name)
            self._db_operations += 1
        
        self._metadata_cache[chat_id] = CacheEntry(
            data={"name": name},
            timestamp=time.time(),
            ttl=self.cache_ttl
        )

    async def cleanup_expired_images(self) -> int:
        """Clean up expired images and return count of deleted images."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM images WHERE expires_at < CURRENT_TIMESTAMP"
            )
            self._db_operations += 1
            
            expired_keys = [
                key for key, entry in self._image_cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._image_cache[key]
            
            deleted_count = int(result.split()[-1]) if result else 0
            if deleted_count > 0:
                logger.debug(f"Cleaned up {deleted_count} expired images")
            
            return deleted_count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "db_operations": self._db_operations,
            "cached_conversations": len(self._message_cache),
            "cached_metadata": len(self._metadata_cache),
            "cached_images": len(self._image_cache)
        }

    def load_conversation_history(self, chat_id: str) -> List[Dict]:
        """Legacy method - converts to async call."""
        import asyncio
        return asyncio.create_task(self._load_conversation_history_dict(chat_id))

    async def _load_conversation_history_dict(self, chat_id: str) -> List[Dict]:
        """Load conversation history in dict format for compatibility."""
        messages = await self.get_messages(chat_id)
        return [self._message_to_dict(msg) for msg in messages]

    def save_conversation_history(self, chat_id: str, messages: List[Dict]) -> None:
        """Legacy method - converts to async call."""
        import asyncio
        message_objects = [self._dict_to_message(msg) for msg in messages]
        return asyncio.create_task(self.save_messages(chat_id, message_objects))

    # Batch Analysis Methods

    async def create_batch_job(
        self,
        batch_id: str,
        total_images: int,
        analysis_prompt: str,
        report_format: str = "markdown"
    ) -> None:
        """Create a new batch analysis job.

        Args:
            batch_id: Unique batch identifier
            total_images: Total number of images to process
            analysis_prompt: The prompt to use for analyzing each image
            report_format: Output format (markdown, html, pdf)
        """
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO batch_analysis_jobs (batch_id, total_images, analysis_prompt, report_format, status)
                VALUES ($1, $2, $3, $4, 'queued')
            """, batch_id, total_images, analysis_prompt, report_format)
            self._db_operations += 1

    async def update_batch_progress(
        self,
        batch_id: str,
        processed_count: int,
        status: str = "processing"
    ) -> None:
        """Update batch job progress.

        Args:
            batch_id: Batch identifier
            processed_count: Number of images processed so far
            status: Current status (queued, processing, completed, failed)
        """
        async with self.pool.acquire() as conn:
            if status == "completed":
                await conn.execute("""
                    UPDATE batch_analysis_jobs
                    SET processed_count = $2, status = $3, updated_at = CURRENT_TIMESTAMP, completed_at = CURRENT_TIMESTAMP
                    WHERE batch_id = $1
                """, batch_id, processed_count, status)
            else:
                await conn.execute("""
                    UPDATE batch_analysis_jobs
                    SET processed_count = $2, status = $3, updated_at = CURRENT_TIMESTAMP
                    WHERE batch_id = $1
                """, batch_id, processed_count, status)
            self._db_operations += 1

    async def set_batch_error(self, batch_id: str, error_message: str) -> None:
        """Mark a batch job as failed with error message.

        Args:
            batch_id: Batch identifier
            error_message: Error description
        """
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE batch_analysis_jobs
                SET status = 'failed', error_message = $2, updated_at = CURRENT_TIMESTAMP
                WHERE batch_id = $1
            """, batch_id, error_message)
            self._db_operations += 1

    async def store_batch_result(
        self,
        result_id: str,
        batch_id: str,
        image_id: str,
        image_filename: str,
        analysis_result: str,
        error_message: str = None
    ) -> None:
        """Store analysis result for a single image in the batch.

        Args:
            result_id: Unique result identifier
            batch_id: Batch identifier
            image_id: Image identifier
            image_filename: Original filename
            analysis_result: The analysis text from the model
            error_message: Optional error if analysis failed
        """
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO batch_analysis_results (result_id, batch_id, image_id, image_filename, analysis_result, error_message)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, result_id, batch_id, image_id, image_filename, analysis_result, error_message)
            self._db_operations += 1

    async def get_batch_job(self, batch_id: str) -> Optional[Dict]:
        """Get batch job details.

        Args:
            batch_id: Batch identifier

        Returns:
            Batch job details or None if not found
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT batch_id, status, total_images, processed_count, analysis_prompt,
                       report_format, created_at, updated_at, completed_at, error_message
                FROM batch_analysis_jobs
                WHERE batch_id = $1
            """, batch_id)
            self._db_operations += 1

            if row:
                return {
                    "batch_id": row['batch_id'],
                    "status": row['status'],
                    "total_images": row['total_images'],
                    "processed_count": row['processed_count'],
                    "analysis_prompt": row['analysis_prompt'],
                    "report_format": row['report_format'],
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                    "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None,
                    "completed_at": row['completed_at'].isoformat() if row['completed_at'] else None,
                    "error_message": row['error_message'],
                    "progress_percent": round((row['processed_count'] / row['total_images'] * 100) if row['total_images'] > 0 else 0, 1)
                }
            return None

    async def get_batch_results(self, batch_id: str) -> List[Dict]:
        """Get all results for a batch job.

        Args:
            batch_id: Batch identifier

        Returns:
            List of analysis results
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT result_id, batch_id, image_id, image_filename, analysis_result, processed_at, error_message
                FROM batch_analysis_results
                WHERE batch_id = $1
                ORDER BY processed_at ASC
            """, batch_id)
            self._db_operations += 1

            return [
                {
                    "result_id": row['result_id'],
                    "batch_id": row['batch_id'],
                    "image_id": row['image_id'],
                    "image_filename": row['image_filename'],
                    "analysis_result": row['analysis_result'],
                    "processed_at": row['processed_at'].isoformat() if row['processed_at'] else None,
                    "error_message": row['error_message']
                }
                for row in rows
            ]

    async def list_batch_jobs(self, limit: int = 50) -> List[Dict]:
        """List all batch jobs.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of batch job summaries
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT batch_id, status, total_images, processed_count, report_format,
                       created_at, completed_at
                FROM batch_analysis_jobs
                ORDER BY created_at DESC
                LIMIT $1
            """, limit)
            self._db_operations += 1

            return [
                {
                    "batch_id": row['batch_id'],
                    "status": row['status'],
                    "total_images": row['total_images'],
                    "processed_count": row['processed_count'],
                    "report_format": row['report_format'],
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                    "completed_at": row['completed_at'].isoformat() if row['completed_at'] else None,
                    "progress_percent": round((row['processed_count'] / row['total_images'] * 100) if row['total_images'] > 0 else 0, 1)
                }
                for row in rows
            ]

    async def delete_batch_job(self, batch_id: str) -> bool:
        """Delete a batch job and all its results.

        Args:
            batch_id: Batch identifier

        Returns:
            True if deleted, False otherwise
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM batch_analysis_jobs WHERE batch_id = $1",
                batch_id
            )
            self._db_operations += 1
            return "DELETE 1" in result
