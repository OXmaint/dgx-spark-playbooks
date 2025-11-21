#!/usr/bin/env python3
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

"""Cron job script for automated batch image processing.

This script:
1. Calls an external API to get image URLs and descriptions
2. Downloads images from provided URLs
3. Uploads images to the local backend
4. Triggers batch processing
5. Waits for completion and downloads HTML report
6. Uploads the HTML report back to external API
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/batch_processor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BatchProcessorCron:
    """Automated batch processor for image analysis via cron job."""

    def __init__(
        self,
        external_api_get_url: str,
        external_api_post_url: str,
        external_api_key: str,
        backend_api_url: str,
        organization_id: str
    ):
        """Initialize the batch processor.

        Args:
            external_api_get_url: URL of the external API to fetch job details (with organization_id param)
            external_api_post_url: URL of the external API to upload results (with organization_id param)
            external_api_key: API key for the external API
            backend_api_url: URL of your local backend API
            organization_id: Organization ID for API requests
        """
        self.external_api_get_url = external_api_get_url
        self.external_api_post_url = external_api_post_url
        self.external_api_key = external_api_key
        self.backend_api_url = backend_api_url.rstrip('/')
        self.organization_id = organization_id

        # Create temp directory for downloads
        self.temp_dir = Path('/tmp/batch_processor')
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def fetch_jobs_from_external_api(self) -> List[Dict]:
        """Fetch all unprocessed job details from external API.

        The API returns a list of vision analysis objects. Each object contains:
        - vision_analysis_id
        - vision_analysis_run_code
        - user_query
        - organization_name
        - is_processed
        - documents (array with image URLs and optional JSON description file URL)

        Returns:
            List of dictionaries with parsed job details, each including:
            - vision_analysis_id: ID of the vision analysis
            - vision_analysis_run_code: Run code for the job
            - image_urls: List of image URLs from documents
            - descriptions_url: URL to JSON description file (if exists)
            - analysis_prompt: The user query
            - organization: Organization name
        """
        try:
            logger.info(f"Fetching jobs from external API: {self.external_api_get_url}")

            headers = {
                'Authorization': f'Bearer {self.external_api_key}',
                'Content-Type': 'application/json'
            }

            # Query for all unprocessed jobs (no page_size limit)
            params = {
                'is_processed': 'false'
            }

            response = requests.get(
                self.external_api_get_url,
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()

            api_response = response.json()

            # Check if response has the expected structure
            if api_response.get('code') != 200 or api_response.get('status') != 'success':
                logger.error(f"API returned non-success status: {api_response}")
                return []

            data = api_response.get('data', {})
            vision_analyses = data.get('vision_analysis', [])

            if not vision_analyses:
                logger.warning("No unprocessed vision analysis jobs found")
                return []

            logger.info(f"Found {len(vision_analyses)} unprocessed jobs")

            # Parse all jobs
            parsed_jobs = []
            for job in vision_analyses:
                # Parse the job data
                vision_analysis_id = job.get('vision_analysis_id')
                run_code = job.get('vision_analysis_run_code')
                user_query = job.get('user_query')
                organization_name = job.get('organization_name')
                documents = job.get('documents', [])

                # Extract image URLs and description file URL from documents
                image_urls = []
                descriptions_url = None

                for doc in documents:
                    # Only process input documents
                    if doc.get('is_input', False):
                        doc_type = doc.get('document_type', '').lower()
                        doc_path = doc.get('document_path')

                        if doc_type == 'image':
                            image_urls.append(doc_path)
                        elif doc_type == 'json':
                            descriptions_url = doc_path

                logger.info(f"Parsed job: {run_code} with {len(image_urls)} images")
                if descriptions_url:
                    logger.info(f"Description file found: {descriptions_url}")

                parsed_jobs.append({
                    'vision_analysis_id': vision_analysis_id,
                    'vision_analysis_run_code': run_code,
                    'image_urls': image_urls,
                    'descriptions_url': descriptions_url,
                    'analysis_prompt': user_query,
                    'organization': organization_name
                })

            return parsed_jobs

        except requests.RequestException as e:
            logger.error(f"Error fetching jobs from external API: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching jobs: {e}", exc_info=True)
            return []

    def download_images(self, image_urls: List[str]) -> List[Tuple[str, bytes, str]]:
        """Download images from URLs.

        Args:
            image_urls: List of image URLs

        Returns:
            List of tuples (filename, image_data, content_type)
        """
        images = []

        for i, url in enumerate(image_urls):
            try:
                logger.info(f"Downloading image {i+1}/{len(image_urls)}: {url}")

                # Download image
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                image_data = response.content

                # Extract filename from URL
                filename = url.split('/')[-1].split('?')[0]  # Remove query params if any
                if not filename or '.' not in filename:
                    filename = f"image_{i+1}.jpg"

                # Determine content type from extension or response headers
                content_type = response.headers.get('Content-Type', 'image/jpeg')

                images.append((filename, image_data, content_type))

            except Exception as e:
                logger.error(f"Error downloading image {url}: {e}")
                continue

        logger.info(f"Successfully downloaded {len(images)} images")
        return images

    def download_descriptions(self, descriptions_url: str) -> Optional[Dict[str, str]]:
        """Download and parse descriptions JSON from URL.

        Args:
            descriptions_url: URL to descriptions JSON file

        Returns:
            Dictionary mapping filename to description
        """
        try:
            logger.info(f"Downloading descriptions file: {descriptions_url}")

            # Download descriptions JSON
            response = requests.get(descriptions_url, timeout=30)
            response.raise_for_status()

            # Parse JSON
            descriptions = response.json()
            logger.info(f"Loaded descriptions for {len(descriptions)} images")

            return descriptions

        except Exception as e:
            logger.error(f"Error downloading descriptions: {e}")
            return None

    def upload_images_to_backend(
        self,
        images: List[Tuple[str, bytes, str]]
    ) -> List[str]:
        """Upload images to the backend and get image IDs.

        Args:
            images: List of tuples (filename, image_data, content_type)

        Returns:
            List of image IDs
        """
        image_ids = []

        for filename, image_data, content_type in images:
            try:
                logger.info(f"Uploading image to backend: {filename}")

                files = {
                    'image': (filename, image_data, content_type)
                }
                data = {
                    'chat_id': 'cron_batch_job'
                }

                response = requests.post(
                    f"{self.backend_api_url}/upload-image",
                    files=files,
                    data=data,
                    timeout=60
                )
                response.raise_for_status()

                result = response.json()
                image_ids.append(result['image_id'])
                logger.info(f"Uploaded {filename} with ID: {result['image_id']}")

            except Exception as e:
                logger.error(f"Error uploading image {filename}: {e}")
                continue

        logger.info(f"Successfully uploaded {len(image_ids)} images")
        return image_ids

    def trigger_batch_analysis(
        self,
        image_ids: List[str],
        analysis_prompt: str,
        descriptions_map: Optional[Dict[str, str]] = None,
        organization: Optional[str] = None
    ) -> Optional[str]:
        """Trigger batch analysis on the backend.

        Args:
            image_ids: List of image IDs to process
            analysis_prompt: The analysis prompt
            descriptions_map: Optional descriptions mapping
            organization: Optional organization filter

        Returns:
            Batch ID if successful, None otherwise
        """
        try:
            logger.info(f"Triggering batch analysis for {len(image_ids)} images")

            payload = {
                'image_ids': image_ids,
                'analysis_prompt': analysis_prompt,
                'report_format': 'html',
                'organization': organization,
                'descriptions_map': descriptions_map
            }

            response = requests.post(
                f"{self.backend_api_url}/batch-analyze",
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            batch_id = result['batch_id']
            logger.info(f"Batch analysis started with ID: {batch_id}")

            return batch_id

        except Exception as e:
            logger.error(f"Error triggering batch analysis: {e}")
            return None

    def wait_for_completion(
        self,
        batch_id: str,
        timeout: int = 3600,
        poll_interval: int = 10
    ) -> bool:
        """Wait for batch analysis to complete.

        Args:
            batch_id: Batch job ID
            timeout: Maximum time to wait in seconds
            poll_interval: Polling interval in seconds

        Returns:
            True if completed successfully, False otherwise
        """
        logger.info(f"Waiting for batch {batch_id} to complete...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.backend_api_url}/batch-analyze/{batch_id}/status",
                    timeout=10
                )
                response.raise_for_status()

                status_data = response.json()
                status = status_data.get('status')

                logger.info(f"Batch status: {status} ({status_data.get('processed_count', 0)}/{status_data.get('total_images', 0)} images)")

                if status == 'completed':
                    logger.info(f"Batch {batch_id} completed successfully")
                    return True
                elif status == 'failed':
                    logger.error(f"Batch {batch_id} failed")
                    return False

                time.sleep(poll_interval)

            except Exception as e:
                logger.error(f"Error checking batch status: {e}")
                time.sleep(poll_interval)

        logger.error(f"Batch {batch_id} timed out after {timeout} seconds")
        return False

    def download_report(self, batch_id: str) -> Optional[str]:
        """Download the HTML report from the backend.

        Args:
            batch_id: Batch job ID

        Returns:
            HTML report content as string, or None if failed
        """
        try:
            logger.info(f"Downloading HTML report for batch {batch_id}")

            response = requests.get(
                f"{self.backend_api_url}/batch-analyze/{batch_id}/report",
                params={'format': 'html', 'include_images': True},
                timeout=60
            )
            response.raise_for_status()

            html_content = response.text
            logger.info(f"Downloaded HTML report ({len(html_content)} bytes)")

            return html_content

        except Exception as e:
            logger.error(f"Error downloading report: {e}")
            return None

    def upload_report_to_external_api(
        self,
        html_content: str,
        batch_id: str,
        vision_analysis_id: str
    ) -> bool:
        """Upload HTML report to external API.

        Args:
            html_content: HTML report content
            batch_id: Batch job ID
            vision_analysis_id: Vision analysis ID from the job

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Uploading report to external API for vision_analysis_id: {vision_analysis_id}")

            headers = {
                'Authorization': f'Bearer {self.external_api_key}'
            }

            # Save to temp file
            temp_file = self.temp_dir / f"report_{batch_id}.html"
            temp_file.write_text(html_content, encoding='utf-8')

            # Upload as multipart form data
            with open(temp_file, 'rb') as f:
                files = {
                    'file': (f'report_{batch_id}.html', f, 'text/html')
                }
                data = {
                    'organization_id': self.organization_id,
                    'vision_analysis_id': vision_analysis_id
                }

                response = requests.post(
                    self.external_api_post_url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=60
                )
                response.raise_for_status()

            logger.info(f"Successfully uploaded report to external API")

            # Clean up temp file
            temp_file.unlink()

            return True

        except Exception as e:
            logger.error(f"Error uploading report to external API: {e}")
            return False

    def process_single_job(self, job_data: Dict) -> bool:
        """Process a single vision analysis job.

        Args:
            job_data: Dictionary with job details

        Returns:
            True if successful, False otherwise
        """
        vision_analysis_id = job_data.get('vision_analysis_id')
        run_code = job_data.get('vision_analysis_run_code')
        image_urls = job_data.get('image_urls', [])
        descriptions_url = job_data.get('descriptions_url')
        analysis_prompt = job_data.get('analysis_prompt', 'Analyze this image for defects and anomalies')
        organization = job_data.get('organization')

        logger.info(f"Processing job: {run_code} (ID: {vision_analysis_id})")

        try:
            if not image_urls:
                logger.warning(f"No images to process for job {run_code}")
                return False

            # Step 1: Download images
            images = self.download_images(image_urls)
            if not images:
                logger.error(f"Failed to download any images for job {run_code}")
                return False

            # Step 2: Download descriptions if provided
            descriptions_map = None
            if descriptions_url:
                descriptions_map = self.download_descriptions(descriptions_url)
                if descriptions_map:
                    logger.info(f"Loaded descriptions for {len(descriptions_map)} images")

            # Step 3: Upload images to backend
            image_ids = self.upload_images_to_backend(images)
            if not image_ids:
                logger.error(f"Failed to upload any images to backend for job {run_code}")
                return False

            # Step 4: Trigger batch analysis
            batch_id = self.trigger_batch_analysis(
                image_ids,
                analysis_prompt,
                descriptions_map,
                organization
            )
            if not batch_id:
                logger.error(f"Failed to trigger batch analysis for job {run_code}")
                return False

            # Step 5: Wait for completion
            if not self.wait_for_completion(batch_id, timeout=3600):
                logger.error(f"Batch processing did not complete successfully for job {run_code}")
                return False

            # Step 6: Download HTML report
            html_report = self.download_report(batch_id)
            if not html_report:
                logger.error(f"Failed to download HTML report for job {run_code}")
                return False

            # Step 7: Upload report to external API
            if not self.upload_report_to_external_api(html_report, batch_id, vision_analysis_id):
                logger.error(f"Failed to upload report to external API for job {run_code}")
                return False

            logger.info(f"Successfully completed job: {run_code} (ID: {vision_analysis_id})")
            return True

        except Exception as e:
            logger.error(f"Error processing job {run_code}: {e}", exc_info=True)
            return False

    def run(self) -> bool:
        """Run the complete batch processing workflow for all unprocessed jobs.

        Returns:
            True if at least one job was processed successfully, False otherwise
        """
        logger.info("=" * 80)
        logger.info("Starting batch processing cron job")
        logger.info("=" * 80)

        try:
            # Step 1: Fetch all unprocessed jobs from external API
            jobs = self.fetch_jobs_from_external_api()
            if not jobs:
                logger.warning("No unprocessed jobs found")
                return False

            logger.info(f"Processing {len(jobs)} unprocessed jobs")

            # Track results
            successful_jobs = 0
            failed_jobs = 0

            # Step 2: Process each job
            for i, job_data in enumerate(jobs, 1):
                logger.info("=" * 80)
                logger.info(f"Processing job {i}/{len(jobs)}")
                logger.info("=" * 80)

                if self.process_single_job(job_data):
                    successful_jobs += 1
                else:
                    failed_jobs += 1

            # Summary
            logger.info("=" * 80)
            logger.info(f"Batch processing completed!")
            logger.info(f"Total jobs: {len(jobs)}")
            logger.info(f"Successful: {successful_jobs}")
            logger.info(f"Failed: {failed_jobs}")
            logger.info("=" * 80)

            # Return True if at least one job succeeded
            return successful_jobs > 0

        except Exception as e:
            logger.error(f"Unexpected error in batch processing: {e}", exc_info=True)
            return False


def main():
    """Main entry point for the cron job."""
    parser = argparse.ArgumentParser(description='Automated batch image processing cron job')
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON)',
                       default='/etc/batch_processor/config.json')
    parser.add_argument('--external-api-get-url', type=str, help='External API GET URL for fetching jobs')
    parser.add_argument('--external-api-post-url', type=str, help='External API POST URL for uploading results')
    parser.add_argument('--external-api-key', type=str, help='External API key')
    parser.add_argument('--backend-api-url', type=str, help='Backend API URL')
    parser.add_argument('--organization-id', type=str, help='Organization ID')

    args = parser.parse_args()

    # Load configuration
    config = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")

    # Override with command line args and environment variables
    external_api_get_url = (args.external_api_get_url or
                           config.get('external_api_get_url') or
                           os.getenv('EXTERNAL_API_GET_URL'))
    external_api_post_url = (args.external_api_post_url or
                            config.get('external_api_post_url') or
                            os.getenv('EXTERNAL_API_POST_URL'))
    external_api_key = (args.external_api_key or
                       config.get('external_api_key') or
                       os.getenv('EXTERNAL_API_KEY'))
    backend_api_url = (args.backend_api_url or
                      config.get('backend_api_url') or
                      os.getenv('BACKEND_API_URL', 'http://localhost:8000'))
    organization_id = (args.organization_id or
                      config.get('organization_id') or
                      os.getenv('ORGANIZATION_ID'))

    # Validate required parameters
    if not all([external_api_get_url, external_api_post_url, external_api_key, organization_id]):
        logger.error("Missing required configuration. Please provide:")
        logger.error("- external_api_get_url")
        logger.error("- external_api_post_url")
        logger.error("- external_api_key")
        logger.error("- organization_id")
        sys.exit(1)

    # Run batch processor
    processor = BatchProcessorCron(
        external_api_get_url=external_api_get_url,
        external_api_post_url=external_api_post_url,
        external_api_key=external_api_key,
        backend_api_url=backend_api_url,
        organization_id=organization_id
    )

    success = processor.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
