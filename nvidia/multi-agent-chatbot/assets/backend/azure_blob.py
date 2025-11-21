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

"""Azure Blob Storage integration for file operations.

This module provides functionality to connect to Azure Blob Storage and perform
push/pull operations on containers.
"""

import os
from pathlib import Path
from typing import List, Optional, Union
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError, AzureError
import logging

logger = logging.getLogger(__name__)


class AzureBlobStorage:
    """Azure Blob Storage client for container operations."""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        container_name: Optional[str] = None
    ):
        """Initialize Azure Blob Storage client.

        Args:
            connection_string: Azure Storage connection string (preferred method)
            account_name: Storage account name (alternative to connection_string)
            account_key: Storage account key (required if using account_name)
            container_name: Default container name for operations

        Environment variables (if parameters not provided):
            - AZURE_STORAGE_CONNECTION_STRING: Full connection string
            - AZURE_STORAGE_ACCOUNT_NAME: Storage account name
            - AZURE_STORAGE_ACCOUNT_KEY: Storage account key
            - AZURE_STORAGE_CONTAINER_NAME: Default container name
        """
        # Get connection details from parameters or environment
        self.connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.account_name = account_name or os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self.account_key = account_key or os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        self.container_name = container_name or os.getenv("AZURE_STORAGE_CONTAINER_NAME")

        # Initialize BlobServiceClient
        if self.connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.connection_string
            )
            logger.info("Initialized Azure Blob Storage client with connection string")
        elif self.account_name and self.account_key:
            account_url = f"https://{self.account_name}.blob.core.windows.net"
            self.blob_service_client = BlobServiceClient(
                account_url=account_url,
                credential=self.account_key
            )
            logger.info(f"Initialized Azure Blob Storage client for account: {self.account_name}")
        else:
            raise ValueError(
                "Either connection_string or both account_name and account_key must be provided. "
                "Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME + "
                "AZURE_STORAGE_ACCOUNT_KEY environment variables."
            )

    def get_container_client(self, container_name: Optional[str] = None) -> ContainerClient:
        """Get a container client for the specified container.

        Args:
            container_name: Container name (uses default if not provided)

        Returns:
            ContainerClient instance
        """
        target_container = container_name or self.container_name
        if not target_container:
            raise ValueError("Container name must be provided or set as default")

        return self.blob_service_client.get_container_client(target_container)

    def pull_from_container(
        self,
        blob_name: Optional[str] = None,
        local_path: Optional[Union[str, Path]] = None,
        container_name: Optional[str] = None,
        prefix: Optional[str] = None,
        download_all: bool = False
    ) -> Union[bytes, List[str]]:
        """Pull blob(s) from Azure container.

        Args:
            blob_name: Name of the blob to download (ignored if download_all=True)
            local_path: Local file path to save the blob. If None, returns blob data as bytes
            container_name: Container name (uses default if not provided)
            prefix: Blob name prefix for filtering (used with download_all)
            download_all: If True, downloads all blobs (optionally filtered by prefix)

        Returns:
            - If download_all=True: List of downloaded blob names
            - If local_path is None: Blob data as bytes
            - Otherwise: Path to downloaded file

        Example:
            # Download single blob to file
            storage.pull_from_container("data/file.txt", "./downloads/file.txt")

            # Download single blob and get bytes
            data = storage.pull_from_container("data/file.txt")

            # Download all blobs with prefix
            files = storage.pull_from_container(
                prefix="images/",
                local_path="./downloads",
                download_all=True
            )
        """
        container_client = self.get_container_client(container_name)

        if download_all:
            # Download multiple blobs
            downloaded_files = []
            blobs = container_client.list_blobs(name_starts_with=prefix)

            for blob in blobs:
                try:
                    blob_client = container_client.get_blob_client(blob.name)

                    # Determine local file path
                    if local_path:
                        local_file_path = Path(local_path) / blob.name
                        local_file_path.parent.mkdir(parents=True, exist_ok=True)

                        with open(local_file_path, "wb") as file:
                            download_stream = blob_client.download_blob()
                            file.write(download_stream.readall())

                        downloaded_files.append(str(local_file_path))
                        logger.info(f"Downloaded: {blob.name} -> {local_file_path}")
                    else:
                        logger.warning(f"Skipping {blob.name}: local_path required for download_all")

                except Exception as e:
                    logger.error(f"Error downloading blob {blob.name}: {e}")

            return downloaded_files

        else:
            # Download single blob
            if not blob_name:
                raise ValueError("blob_name is required when download_all=False")

            try:
                blob_client = container_client.get_blob_client(blob_name)
                download_stream = blob_client.download_blob()
                blob_data = download_stream.readall()

                if local_path:
                    # Save to file
                    local_file_path = Path(local_path)
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(local_file_path, "wb") as file:
                        file.write(blob_data)

                    logger.info(f"Downloaded: {blob_name} -> {local_file_path}")
                    return str(local_file_path)
                else:
                    # Return bytes
                    logger.info(f"Downloaded: {blob_name} (returned as bytes)")
                    return blob_data

            except ResourceNotFoundError:
                logger.error(f"Blob not found: {blob_name}")
                raise
            except AzureError as e:
                logger.error(f"Azure error downloading blob {blob_name}: {e}")
                raise

    def push_to_container(
        self,
        local_path: Union[str, Path],
        blob_name: Optional[str] = None,
        container_name: Optional[str] = None,
        overwrite: bool = True,
        upload_directory: bool = False
    ) -> Union[str, List[str]]:
        """Push file(s) to Azure container.

        Args:
            local_path: Local file or directory path to upload
            blob_name: Blob name in container (uses filename if not provided)
            container_name: Container name (uses default if not provided)
            overwrite: Whether to overwrite existing blobs
            upload_directory: If True, uploads entire directory recursively

        Returns:
            - If upload_directory=True: List of uploaded blob names
            - Otherwise: Name of uploaded blob

        Example:
            # Upload single file
            storage.push_to_container("./data/file.txt", "uploads/file.txt")

            # Upload file with auto-naming
            storage.push_to_container("./data/file.txt")

            # Upload entire directory
            storage.push_to_container(
                "./data",
                blob_name="uploads/",
                upload_directory=True
            )
        """
        container_client = self.get_container_client(container_name)
        local_path = Path(local_path)

        if upload_directory:
            # Upload multiple files from directory
            if not local_path.is_dir():
                raise ValueError(f"Path is not a directory: {local_path}")

            uploaded_blobs = []
            base_blob_name = blob_name or ""

            # Walk through directory
            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    # Calculate relative path for blob name
                    relative_path = file_path.relative_to(local_path)
                    target_blob_name = f"{base_blob_name}{relative_path}".replace("\\", "/")

                    try:
                        blob_client = container_client.get_blob_client(target_blob_name)

                        with open(file_path, "rb") as data:
                            blob_client.upload_blob(data, overwrite=overwrite)

                        uploaded_blobs.append(target_blob_name)
                        logger.info(f"Uploaded: {file_path} -> {target_blob_name}")

                    except Exception as e:
                        logger.error(f"Error uploading {file_path}: {e}")

            return uploaded_blobs

        else:
            # Upload single file
            if not local_path.is_file():
                raise ValueError(f"Path is not a file: {local_path}")

            target_blob_name = blob_name or local_path.name

            try:
                blob_client = container_client.get_blob_client(target_blob_name)

                with open(local_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=overwrite)

                logger.info(f"Uploaded: {local_path} -> {target_blob_name}")
                return target_blob_name

            except AzureError as e:
                logger.error(f"Azure error uploading {local_path}: {e}")
                raise

    def list_blobs(
        self,
        container_name: Optional[str] = None,
        prefix: Optional[str] = None
    ) -> List[str]:
        """List all blobs in a container.

        Args:
            container_name: Container name (uses default if not provided)
            prefix: Blob name prefix for filtering

        Returns:
            List of blob names
        """
        container_client = self.get_container_client(container_name)

        try:
            blobs = container_client.list_blobs(name_starts_with=prefix)
            blob_names = [blob.name for blob in blobs]
            logger.info(f"Found {len(blob_names)} blobs in container")
            return blob_names
        except AzureError as e:
            logger.error(f"Error listing blobs: {e}")
            raise

    def delete_blob(
        self,
        blob_name: str,
        container_name: Optional[str] = None
    ) -> bool:
        """Delete a blob from container.

        Args:
            blob_name: Name of the blob to delete
            container_name: Container name (uses default if not provided)

        Returns:
            True if deletion was successful
        """
        container_client = self.get_container_client(container_name)

        try:
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
            logger.info(f"Deleted blob: {blob_name}")
            return True
        except ResourceNotFoundError:
            logger.warning(f"Blob not found: {blob_name}")
            return False
        except AzureError as e:
            logger.error(f"Error deleting blob {blob_name}: {e}")
            raise


# Convenience functions for quick operations
def create_storage_client(**kwargs) -> AzureBlobStorage:
    """Create and return an AzureBlobStorage client instance.

    Args:
        **kwargs: Parameters to pass to AzureBlobStorage constructor

    Returns:
        AzureBlobStorage instance
    """
    return AzureBlobStorage(**kwargs)
