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
import glob
from typing import List as TList, Tuple, Optional, Callable, Dict
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from dotenv import load_dotenv
from logger import logger
import requests
from pymilvus import connections, Collection


class CustomEmbeddings:
    """Wraps qwen3 embedding model to match OpenAI format"""
    def __init__(self, model: str = "Qwen3-Embedding-4B-Q8_0.gguf", host: str = "http://qwen3-embedding:8000"):
        self.model = model
        self.url = f"{host}/v1/embeddings"

    def __call__(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            response = requests.post(
                self.url,
                json={"input": text, "model": self.model},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            embeddings.append(data["data"][0]["embedding"])
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts. Required by Milvus library."""
        return self.__call__(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text. Required by Milvus library."""
        return self.__call__([text])[0]


class VectorStore:
    """Vector store for document embedding and retrieval.

    Modifications:
      • Supports upsert-by-id (auto-delete existing rows)
      • Allows disabling chunking (one row per work order)
      • NEW: Target any Milvus collection by name (per company)
    """

    def __init__(
        self,
        embeddings=None,
        uri: str = "http://milvus:19530",
        on_source_deleted: Optional[Callable[[str], None]] = None,
    ):
        try:
            self.embeddings = embeddings or CustomEmbeddings(model="qwen3-embedding-custom")
            self.uri = uri
            self.on_source_deleted = on_source_deleted
            self._initialize_store()

            # cache per-collection langchain stores
            self._stores: Dict[str, Milvus] = {"context": self._store}

            # splitter for generic document ingestion
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )

            logger.debug({"message": "VectorStore initialized successfully"})
        except Exception as e:
            logger.error(
                {"message": "Error initializing VectorStore", "error": str(e)}, exc_info=True
            )
            raise

    # ---------------------------------------------------------------------- #
    #   Core store setup / connection
    # ---------------------------------------------------------------------- #
    def _initialize_store(self):
        # Base collection for generic RAG (app-wide, not org-scoped)
        self._store = Milvus(
            embedding_function=self.embeddings,
            collection_name="context",
            connection_args={"uri": self.uri},
            auto_id=True,
            # IMPORTANT: allow arbitrary metadata keys (no strict schema)
            enable_dynamic_field=True,
    )
        logger.debug(
            {
                "message": "Milvus vector store initialized",
                "uri": self.uri,
                "collection": "context",
            }
    )


    def default_collection_name(self) -> str:
        return "context"

    def _sanitize_collection(self, name: str) -> str:
        return "".join(c if c.isalnum() or c == "_" else "_" for c in name)

    def _get_store_for(self, collection_name: Optional[str] = None) -> Milvus:
        """
        Return a langchain Milvus wrapper for the requested collection.

        Used for per-org collections like:
        - org_83247135_6B56_4C33_8098_78634386BC3C
        and any other custom collections.
        """
        name = self.default_collection_name() if not collection_name else self._sanitize_collection(collection_name)
        if name in self._stores:
            return self._stores[name]

        store = Milvus(
            embedding_function=self.embeddings,
            collection_name=name,
            connection_args={"uri": self.uri},
            auto_id=True,
            # IMPORTANT: allow arbitrary metadata (doc_type, work_order_id, inspection_id, etc.)
            enable_dynamic_field=True,
        )
        self._stores[name] = store
        logger.debug({"message": "Milvus store ready", "collection": name})
        return store

    # ---------------------------------------------------------------------- #
    #   Internal helpers
    # ---------------------------------------------------------------------- #
    def _predelete_ids(self, ids: set[str], collection_name: Optional[str] = None):
        """Delete any existing rows for these logical ids in the target collection."""
        if not ids:
            return
        target = self.default_collection_name() if not collection_name else self._sanitize_collection(collection_name)
        try:
            connections.connect(uri=self.uri)
            col = Collection(target)
            col.load()
            for wo_id in ids:
                col.delete(expr=f'id == "{wo_id}"')
                logger.debug(f"Deleted existing rows for id={wo_id} in collection={target}")
        except Exception as e:
            logger.warning(f"Pre-delete failed in {target}: {e}")
        finally:
            try:
                connections.disconnect("default")
            except Exception:
                pass

    # ---------------------------------------------------------------------- #
    #   Indexing
    # ---------------------------------------------------------------------- #
    def index_documents(
        self,
        documents: TList[Document],
        no_split: bool = False,
        upsert_by_id: bool = True,
        echo: bool = False,
        collection_name: Optional[str] = None,   # NEW
    ):
        """Index documents into Milvus (optionally to a specific collection)."""
        try:
            target = self.default_collection_name() if not collection_name else self._sanitize_collection(collection_name)
            logger.debug(
                {"message": "Starting document indexing", "document_count": len(documents), "collection": target}
            )

            # chunking control
            splits = documents if no_split else self.text_splitter.split_documents(documents)

            # annotate chunk indices
            chunk_total = len(splits)
            for i, d in enumerate(splits):
                d.metadata = d.metadata or {}
                d.metadata["chunk_index"] = i
                d.metadata["chunk_total"] = chunk_total

            # collect logical ids for upsert
            ids = {d.metadata.get("id") for d in splits if d.metadata.get("id")}
            if upsert_by_id and ids:
                self._predelete_ids(ids, collection_name=target)

            # insert into target collection
            store = self._get_store_for(target)
            store.add_documents(splits)
            self.flush_store()

            logger.debug(
                {
                    "message": "Document indexing completed",
                    "chunks_written": len(splits),
                    "upsert_ids": list(ids),
                    "collection": target,
                }
            )

            return splits if echo else None
        except Exception as e:
            logger.error(
                {"message": "Error during document indexing", "error": str(e)}, exc_info=True
            )
            raise

    # ---------------------------------------------------------------------- #
    #   Flush / persistence
    # ---------------------------------------------------------------------- #
    def flush_store(self):
        """Flush the Milvus collection(s) to persist documents."""
        try:
            from pymilvus import utility
            connections.connect(uri=self.uri)
            utility.flush_all()
            logger.debug({"message": "Milvus store flushed"})
        except Exception as e:
            logger.error({"message": "Error flushing Milvus store", "error": str(e)}, exc_info=True)
        finally:
            try:
                connections.disconnect("default")
            except Exception:
                pass

    # ---------------------------------------------------------------------- #
    #   Retrieval
    # ---------------------------------------------------------------------- #
    def get_documents(
        self,
        query: str,
        k: int = 8,
        sources: Optional[TList[str]] = None,
        collection_name: Optional[str] = None,   # NEW
    ) -> TList[Document]:
        """Retrieve similar documents by embedding similarity from a target collection."""
        try:
            target = self.default_collection_name() if not collection_name else self._sanitize_collection(collection_name)
            search_kwargs = {"k": k}
            if sources:
                if len(sources) == 1:
                    filter_expr = f'source == "{sources[0]}"'
                else:
                    source_conditions = [f'source == "{s}"' for s in sources]
                    filter_expr = " || ".join(source_conditions)
                search_kwargs["expr"] = filter_expr

            retriever = self._get_store_for(target).as_retriever(
                search_type="similarity", search_kwargs=search_kwargs
            )
            docs = retriever.invoke(query)
            logger.debug({"message": "Retrieved documents", "query": query, "count": len(docs), "collection": target})
            return docs
        except Exception as e:
            logger.error({"message": "Error retrieving documents", "error": str(e)}, exc_info=True)
            return []

    # ---------------------------------------------------------------------- #
    #   Delete collection
    # ---------------------------------------------------------------------- #
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection from Milvus."""
        try:
            from pymilvus import utility

            connections.connect(uri=self.uri)
            if utility.has_collection(collection_name):
                Collection(name=collection_name).drop()
                if self.on_source_deleted:
                    self.on_source_deleted(collection_name)
                logger.debug({"message": "Collection deleted", "name": collection_name})
                return True
            logger.warning({"message": "Collection not found", "name": collection_name})
            return False
        except Exception as e:
            logger.error(
                {"message": "Error deleting collection", "name": collection_name, "error": str(e)},
                exc_info=True,
            )
            return False
        finally:
            try:
                connections.disconnect("default")
            except Exception:
                pass


def create_vector_store_with_config(config_manager, uri: str = "http://milvus:19530") -> VectorStore:
    """Factory function to create a VectorStore with ConfigManager integration."""
    def handle_source_deleted(source_name: str):
        config = config_manager.read_config()
        if hasattr(config, 'sources') and source_name in config.sources:
            config.sources.remove(source_name)
            config_manager.write_config(config)

    return VectorStore(
        uri=uri,
        on_source_deleted=handle_source_deleted
    )
