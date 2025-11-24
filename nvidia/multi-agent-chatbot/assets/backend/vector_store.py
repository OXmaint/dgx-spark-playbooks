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
from typing import List, Tuple, Optional, Callable, Dict
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
        self._store = Milvus(
            embedding_function=self.embeddings,
            collection_name="context",
            connection_args={"uri": self.uri},
            auto_id=True,
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
        """Return a langchain Milvus wrapper for the requested collection. Lazily creates if needed."""
        name = self.default_collection_name() if not collection_name else self._sanitize_collection(collection_name)
        if name in self._stores:
            return self._stores[name]

        store = Milvus(
            embedding_function=self.embeddings,
            collection_name=name,
            connection_args={"uri": self.uri},
            auto_id=True,
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
    def _load_documents(self, file_paths: List[str] = None, input_dir: str = None, doc_types: List[str] = None) -> List[str]:
        try:
            documents = []
            source_name = None
            
            if input_dir:
                source_name = os.path.basename(os.path.normpath(input_dir))
                logger.debug({
                    "message": "Loading files from directory",
                    "directory": input_dir,
                    "source": source_name
                })
                file_paths = glob.glob(os.path.join(input_dir, "**"), recursive=True)
                file_paths = [f for f in file_paths if os.path.isfile(f)]
            
            logger.info(f"Processing {len(file_paths)} files: {file_paths}")
            
            for i, file_path in enumerate(file_paths):
                try:
                    if not source_name:
                        source_name = os.path.basename(file_path)
                        logger.info(f"Using filename as source: {source_name}")
                    
                    logger.info(f"Loading file: {file_path}")
                    
                    file_ext = os.path.splitext(file_path)[1].lower()
                    logger.info(f"File extension: {file_ext}")
                    
                    try:
                        loader = UnstructuredLoader(file_path)
                        docs = loader.load()
                        logger.info(f"Successfully loaded {len(docs)} documents from {file_path}")
                    except Exception as pdf_error:
                        logger.error(f'error with unstructured loader, trying to load from scratch')
                        file_text = None
                        if file_ext == ".pdf":
                            logger.info("Attempting PyPDF text extraction fallback")
                            try:
                                from pypdf import PdfReader
                                reader = PdfReader(file_path)
                                extracted_pages = []
                                for page in reader.pages:
                                    try:
                                        extracted_pages.append(page.extract_text() or "")
                                    except Exception as per_page_err:
                                        logger.info(f"Warning: failed to extract a page: {per_page_err}")
                                        extracted_pages.append("")
                                file_text = "\n\n".join(extracted_pages).strip()
                            except Exception as pypdf_error:
                                logger.info(f"PyPDF fallback failed: {pypdf_error}")
                                file_text = None

                        if not file_text:
                            logger.info("Falling back to raw text read of file contents")
                            try:
                                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                    file_text = f.read()
                            except Exception as read_error:
                                logger.info(f"Fallback read failed: {read_error}")
                                file_text = ""

                        if file_text and file_text.strip():
                            docs = [Document(
                                page_content=file_text,
                                metadata={
                                    "source": source_name,
                                    "file_path": file_path,
                                    "filename": os.path.basename(file_path),
                                }
                            )]
                        else:
                            logger.info("Creating a simple document as fallback (no text extracted)")
                            docs = [Document(
                                page_content=f"Document: {os.path.basename(file_path)}",
                                metadata={
                                    "source": source_name,
                                    "file_path": file_path,
                                    "filename": os.path.basename(file_path),
                                }
                            )]
                    
                    for doc in docs:
                        if not doc.metadata:
                            doc.metadata = {}
                        
                        cleaned_metadata = {}
                        cleaned_metadata["source"] = source_name
                        cleaned_metadata["file_path"] = file_path
                        cleaned_metadata["filename"] = os.path.basename(file_path)
                        cleaned_metadata["type"] = doc_types[i]
                        
                        for key, value in doc.metadata.items():
                            if key not in ["source", "file_path"]:
                                if isinstance(value, (list, dict, set)):
                                    cleaned_metadata[key] = str(value)
                                elif value is not None:
                                    cleaned_metadata[key] = str(value)
                        
                        doc.metadata = cleaned_metadata
                    documents.extend(docs)
                    logger.debug({
                        "message": "Loaded documents from file",
                        "file_path": file_path,
                        "document_count": len(docs)
                    })
                except Exception as e:
                    logger.error({
                        "message": "Error loading file",
                        "file_path": file_path,
                        "error": str(e)
                    }, exc_info=True)
                    continue

            logger.info(f"Total documents loaded: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error({
                "message": "Error loading documents",
                "error": str(e)
            }, exc_info=True)
            raise
    
    def _get_required_schema_fields(self, collection_name: str) -> dict:
        """Get required fields from collection schema and their default values.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary mapping field names to their default values
        """
        try:
            from pymilvus import utility, DataType

            connections.connect(uri=self.uri)

            # Check if collection exists
            if not utility.has_collection(collection_name):
                # Collection doesn't exist yet, return empty dict
                return {}

            collection = Collection(name=collection_name)
            required_fields = {}

            # Iterate through schema fields
            for field in collection.schema.fields:
                # Skip primary key, vector fields, and fields that are auto-generated
                if field.is_primary or field.dtype == DataType.FLOAT_VECTOR:
                    continue

                # Check if field is required (not nullable and no default value)
                # For LangChain Milvus, common metadata fields that might be required
                field_name = field.name

                # Add common fields that UnstructuredLoader provides for PDFs
                # but might be missing for other file types like CSV
                if field_name == "page_number":
                    required_fields[field_name] = "0"
                elif field_name == "coordinates":
                    required_fields[field_name] = ""
                elif field_name in ["text", "source", "file_path", "filename", "type"]:
                    # These are handled elsewhere, skip
                    continue
                else:
                    # For any other field, try to infer a default based on type
                    if field.dtype in [DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64]:
                        required_fields[field_name] = 0
                    elif field.dtype in [DataType.FLOAT, DataType.DOUBLE]:
                        required_fields[field_name] = 0.0
                    elif field.dtype in [DataType.VARCHAR, DataType.STRING]:
                        required_fields[field_name] = ""
                    elif field.dtype == DataType.BOOL:
                        required_fields[field_name] = False

            logger.debug({
                "message": "Identified required schema fields",
                "collection": collection_name,
                "required_fields": required_fields
            })

            return required_fields

        except Exception as e:
            logger.warning({
                "message": "Error getting schema fields, using default required fields",
                "collection": collection_name,
                "error": str(e)
            })
            # Return common required fields as fallback
            return {
                "page_number": "0",
                "coordinates": ""
            }
        finally:
            try:
                connections.disconnect("default")
            except Exception:
                pass

    def index_documents(
        self,
        documents: List[Document],
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

            # Get required fields for this collection
            required_fields = self._get_required_schema_fields(target)

            # chunking control
            splits = documents if no_split else self.text_splitter.split_documents(documents)

            # annotate chunk indices and add required fields
            chunk_total = len(splits)
            for i, d in enumerate(splits):
                d.metadata = d.metadata or {}
                d.metadata["chunk_index"] = i
                d.metadata["chunk_total"] = chunk_total

                # Add any missing required fields with their default values
                for field_name, default_value in required_fields.items():
                    if field_name not in d.metadata:
                        d.metadata[field_name] = default_value

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
    def view_documents(
        self,
        collection_name: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, any]:
        """
        View documents and their metadata from the specified collection with pagination.
        If collection_name is not provided, uses the default collection.

        Args:
            collection_name: Optional collection name
            limit: Maximum number of documents to return (default 100)
            offset: Number of documents to skip (default 0)

        Returns:
            Dict with 'documents', 'total_count', 'limit', 'offset', and 'collection' keys.
        """
        try:
            target = self.default_collection_name() if not collection_name else self._sanitize_collection(collection_name)

            # Connect and query Milvus directly for proper pagination support
            connections.connect(uri=self.uri)
            milvus_collection = Collection(name=target)
            milvus_collection.load()

            # Get total count
            total_count = milvus_collection.num_entities

            # Query with pagination - Milvus query uses expr="" for all docs
            results = milvus_collection.query(
                expr="",
                output_fields=["text", "source", "file_path", "filename", "type", "chunk_index", "chunk_total"],
                limit=limit,
                offset=offset
            )

            documents = []
            for row in results:
                # Extract content (Milvus stores it as 'text' field)
                content = row.get("text", "")

                # Build metadata from other fields
                metadata = {
                    "source": row.get("source", ""),
                    "file_path": row.get("file_path", ""),
                    "filename": row.get("filename", ""),
                    "type": row.get("type", ""),
                    "chunk_index": row.get("chunk_index", 0),
                    "chunk_total": row.get("chunk_total", 0),
                }

                documents.append({
                    "content": content,
                    "metadata": metadata,
                    "collection": target,
                })

            logger.debug({
                "message": "Viewed documents with pagination",
                "collection": target,
                "document_count": len(documents),
                "total_count": total_count,
                "limit": limit,
                "offset": offset
            })

            return {
                "documents": documents,
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "collection": target,
                "has_more": (offset + len(documents)) < total_count
            }

        except Exception as e:
            logger.error({
                "message": "Error viewing documents",
                "error": str(e),
                "collection": collection_name
            }, exc_info=True)
            return {
                "documents": [],
                "total_count": 0,
                "limit": limit,
                "offset": offset,
                "collection": collection_name or self.default_collection_name(),
                "has_more": False,
                "error": str(e)
            }
        finally:
            try:
                connections.disconnect("default")
            except Exception:
                pass
    
    
    def get_documents(
        self,
        query: str,
        k: int = 8,
        sources: Optional[List[str]] = None,
        collection_name: Optional[str] = None, 
        doc_type: Optional[str] = 'document', # NEW
    ) -> List[Document]:
        """Retrieve similar documents by embedding similarity from a target collection."""
        try:
            target = self.default_collection_name() if not collection_name else self._sanitize_collection(collection_name)
            search_kwargs = {"k": k}
            filter_expr = ""
            if sources:
                if len(sources) == 1:
                    filter_expr = f'source == "{sources[0]}"'
                else:
                    source_conditions = [f'source == "{s}"' for s in sources]
                    filter_expr = " || ".join(source_conditions)
            
            if doc_type:
                if filter_expr:
                    filter_expr += " && "
                filter_expr += f'type == "{doc_type}"'
            
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
    #   List collections
    # ---------------------------------------------------------------------- #
    def list_collections(self) -> List[Dict[str, any]]:
        """List all collections and their document counts.

        Returns:
            List of dicts with 'name' and 'count' keys for each collection
        """
        try:
            from pymilvus import utility

            connections.connect(uri=self.uri)
            collection_names = utility.list_collections()

            collections_info = []
            for name in collection_names:
                try:
                    col = Collection(name=name)
                    col.load()
                    count = col.num_entities
                    collections_info.append({
                        "name": name,
                        "document_count": count
                    })
                except Exception as e:
                    logger.warning(f"Error getting count for collection {name}: {e}")
                    collections_info.append({
                        "name": name,
                        "document_count": 0,
                        "error": str(e)
                    })

            logger.debug({
                "message": "Listed collections",
                "collection_count": len(collections_info)
            })
            return collections_info

        except Exception as e:
            logger.error({
                "message": "Error listing collections",
                "error": str(e)
            }, exc_info=True)
            return []
        finally:
            try:
                connections.disconnect("default")
            except Exception:
                pass

    # ---------------------------------------------------------------------- #
    #   Delete / Purge collection
    # ---------------------------------------------------------------------- #
    def purge_collection(self, collection_name: str) -> Dict[str, any]:
        """Delete all documents from a collection without dropping the collection.

        Args:
            collection_name: Name of the collection to purge

        Returns:
            Dict with status and count of deleted documents
        """
        try:
            from pymilvus import utility

            target = self._sanitize_collection(collection_name)
            connections.connect(uri=self.uri)

            if not utility.has_collection(target):
                logger.warning({"message": "Collection not found", "name": target})
                return {
                    "success": False,
                    "message": f"Collection '{target}' not found",
                    "deleted_count": 0
                }

            collection = Collection(name=target)
            collection.load()

            # Get count before deletion
            count_before = collection.num_entities

            if count_before == 0:
                logger.info({"message": "Collection already empty", "name": target})
                return {
                    "success": True,
                    "message": f"Collection '{target}' is already empty",
                    "deleted_count": 0,
                    "remaining_count": 0
                }

            # Get the primary key field name from schema
            primary_field = None
            for field in collection.schema.fields:
                if field.is_primary:
                    primary_field = field.name
                    break

            if not primary_field:
                raise ValueError(f"No primary key field found in collection '{target}'")

            logger.debug({
                "message": "Found primary key field",
                "collection": target,
                "primary_field": primary_field
            })

            # Query all primary keys in batches and delete
            batch_size = 1000
            total_deleted = 0
            offset = 0

            while True:
                # Query a batch of primary keys
                results = collection.query(
                    expr="",
                    output_fields=[primary_field],
                    limit=batch_size,
                    offset=offset
                )

                if not results:
                    break

                # Extract IDs
                ids = [str(row[primary_field]) for row in results]

                if not ids:
                    break

                # Build delete expression for this batch
                # Use IN operator for batch deletion
                ids_str = ", ".join([f'"{id}"' if isinstance(row[primary_field], str) else str(id) for row, id in zip(results, ids)])
                delete_expr = f'{primary_field} in [{ids_str}]'

                # Delete this batch
                collection.delete(expr=delete_expr)
                total_deleted += len(ids)

                logger.debug({
                    "message": "Deleted batch",
                    "collection": target,
                    "batch_size": len(ids),
                    "total_deleted": total_deleted
                })

                # If we got fewer results than batch_size, we're done
                if len(results) < batch_size:
                    break

            # Flush to ensure deletion is persisted
            collection.flush()

            # Get count after deletion to verify
            collection.load()
            count_after = collection.num_entities

            logger.info({
                "message": "Collection purged",
                "name": target,
                "count_before": count_before,
                "deleted_count": total_deleted,
                "remaining_count": count_after
            })

            # Remove from store cache to force reload
            if target in self._stores:
                del self._stores[target]

            return {
                "success": True,
                "message": f"Purged {total_deleted} documents from collection '{target}'",
                "deleted_count": total_deleted,
                "remaining_count": count_after
            }

        except Exception as e:
            logger.error({
                "message": "Error purging collection",
                "name": collection_name,
                "error": str(e)
            }, exc_info=True)
            return {
                "success": False,
                "message": f"Error purging collection: {str(e)}",
                "deleted_count": 0
            }
        finally:
            try:
                connections.disconnect("default")
            except Exception:
                pass

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