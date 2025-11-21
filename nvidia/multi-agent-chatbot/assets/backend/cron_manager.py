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

"""Cron Job Manager for scheduling and managing batch processing jobs.

This module provides functionality to:
- Create and schedule cron jobs
- Activate/deactivate cron jobs
- List active cron jobs
- Execute cron jobs via system crontab or APScheduler
"""

import json
import os
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.base import JobLookupError

from logger import logger


class CronJobManager:
    """Manager for cron jobs using APScheduler for in-process scheduling."""

    def __init__(self, postgres_storage=None):
        """Initialize the cron job manager.

        Args:
            postgres_storage: PostgreSQL storage instance for persistence
        """
        self.storage = postgres_storage
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        logger.info("CronJobManager initialized with APScheduler")

    async def create_cron_job(
        self,
        name: str,
        schedule: str,
        external_api_get_url: str,
        external_api_post_url: str,
        external_api_key: str,
        backend_api_url: str,
        organization_id: str,
        enabled: bool = True
    ) -> str:
        """Create a new cron job.

        Args:
            name: Job name/identifier
            schedule: Cron schedule expression (e.g., "0 * * * *" for hourly)
            external_api_get_url: URL of external API for fetching jobs
            external_api_post_url: URL of external API for uploading results
            external_api_key: API key for external API
            backend_api_url: Backend API URL
            organization_id: Organization ID
            enabled: Whether job is enabled

        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())

        # Store job configuration in database
        await self.storage.create_cron_job(
            job_id=job_id,
            name=name,
            schedule=schedule,
            config={
                'external_api_get_url': external_api_get_url,
                'external_api_post_url': external_api_post_url,
                'external_api_key': external_api_key,
                'backend_api_url': backend_api_url,
                'organization_id': organization_id
            },
            enabled=enabled
        )

        # Schedule job if enabled
        if enabled:
            await self._schedule_job(job_id, name, schedule, {
                'external_api_get_url': external_api_get_url,
                'external_api_post_url': external_api_post_url,
                'external_api_key': external_api_key,
                'backend_api_url': backend_api_url,
                'organization_id': organization_id
            })

        logger.info(f"Created cron job: {name} ({job_id}) - {'enabled' if enabled else 'disabled'}")
        return job_id

    async def _schedule_job(self, job_id: str, name: str, schedule: str, config: Dict):
        """Schedule a job with APScheduler.

        Args:
            job_id: Job ID
            name: Job name
            schedule: Cron schedule expression
            config: Job configuration
        """
        try:
            # Parse cron schedule (e.g., "0 * * * *" -> minute=0, hour=*)
            parts = schedule.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid cron schedule: {schedule}")

            minute, hour, day, month, day_of_week = parts

            # Create cron trigger
            trigger = CronTrigger(
                minute=minute,
                hour=hour,
                day=day,
                month=month,
                day_of_week=day_of_week
            )

            # Schedule job
            self.scheduler.add_job(
                func=self._execute_batch_job,
                trigger=trigger,
                id=job_id,
                name=name,
                kwargs={'job_id': job_id, 'config': config},
                replace_existing=True
            )

            logger.info(f"Scheduled job: {name} ({job_id}) with schedule: {schedule}")

        except Exception as e:
            logger.error(f"Error scheduling job {job_id}: {e}")
            raise

    def _execute_batch_job(self, job_id: str, config: Dict):
        """Execute a batch processing job.

        Args:
            job_id: Job ID
            config: Job configuration
        """
        logger.info(f"Executing cron job: {job_id}")

        try:
            # Build command to execute cron_batch_processor.py
            script_path = Path(__file__).parent / 'cron_batch_processor.py'

            # Create temp config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config, f)
                temp_config = f.name

            # Execute script
            cmd = [
                'python3',
                str(script_path),
                '--config', temp_config
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hours max
            )

            # Clean up temp file
            os.unlink(temp_config)

            # Log execution
            if result.returncode == 0:
                logger.info(f"Cron job {job_id} completed successfully")
                self._record_execution(job_id, 'success', result.stdout)
            else:
                logger.error(f"Cron job {job_id} failed: {result.stderr}")
                self._record_execution(job_id, 'failed', result.stderr)

        except subprocess.TimeoutExpired:
            logger.error(f"Cron job {job_id} timed out")
            self._record_execution(job_id, 'timeout', 'Job exceeded 2 hour timeout')
        except Exception as e:
            logger.error(f"Error executing cron job {job_id}: {e}")
            self._record_execution(job_id, 'error', str(e))

    def _record_execution(self, job_id: str, status: str, output: str):
        """Record job execution in database (async wrapper needed).

        Args:
            job_id: Job ID
            status: Execution status
            output: Execution output/error
        """
        # Note: This needs to be called from async context
        # For now, just log it
        logger.info(f"Job {job_id} execution: {status}")

    async def activate_cron_job(self, job_id: str) -> bool:
        """Activate (enable) a cron job.

        Args:
            job_id: Job ID

        Returns:
            True if successful
        """
        try:
            # Get job from database
            job = await self.storage.get_cron_job(job_id)
            if not job:
                logger.error(f"Job not found: {job_id}")
                return False

            # Update database
            await self.storage.update_cron_job_status(job_id, enabled=True)

            # Schedule with APScheduler
            await self._schedule_job(
                job_id,
                job['name'],
                job['schedule'],
                job['config']
            )

            logger.info(f"Activated cron job: {job_id}")
            return True

        except Exception as e:
            logger.error(f"Error activating cron job {job_id}: {e}")
            return False

    async def deactivate_cron_job(self, job_id: str) -> bool:
        """Deactivate (disable) a cron job.

        Args:
            job_id: Job ID

        Returns:
            True if successful
        """
        try:
            # Update database
            await self.storage.update_cron_job_status(job_id, enabled=False)

            # Remove from scheduler
            try:
                self.scheduler.remove_job(job_id)
                logger.info(f"Removed job {job_id} from scheduler")
            except JobLookupError:
                logger.warning(f"Job {job_id} not found in scheduler")

            logger.info(f"Deactivated cron job: {job_id}")
            return True

        except Exception as e:
            logger.error(f"Error deactivating cron job {job_id}: {e}")
            return False

    async def list_cron_jobs(self, enabled_only: bool = False) -> List[Dict]:
        """List all cron jobs.

        Args:
            enabled_only: Only return enabled jobs

        Returns:
            List of job dictionaries
        """
        try:
            jobs = await self.storage.list_cron_jobs(enabled_only=enabled_only)

            # Add scheduler status
            scheduled_jobs = {job.id: job for job in self.scheduler.get_jobs()}

            for job in jobs:
                job_id = job['job_id']
                job['scheduled'] = job_id in scheduled_jobs

                if job['scheduled']:
                    scheduler_job = scheduled_jobs[job_id]
                    job['next_run'] = scheduler_job.next_run_time.isoformat() if scheduler_job.next_run_time else None

            return jobs

        except Exception as e:
            logger.error(f"Error listing cron jobs: {e}")
            return []

    async def get_cron_job(self, job_id: str) -> Optional[Dict]:
        """Get a specific cron job.

        Args:
            job_id: Job ID

        Returns:
            Job dictionary or None
        """
        try:
            job = await self.storage.get_cron_job(job_id)

            if job:
                # Add scheduler info
                try:
                    scheduler_job = self.scheduler.get_job(job_id)
                    job['scheduled'] = True
                    job['next_run'] = scheduler_job.next_run_time.isoformat() if scheduler_job.next_run_time else None
                except JobLookupError:
                    job['scheduled'] = False
                    job['next_run'] = None

            return job

        except Exception as e:
            logger.error(f"Error getting cron job {job_id}: {e}")
            return None

    async def delete_cron_job(self, job_id: str) -> bool:
        """Delete a cron job.

        Args:
            job_id: Job ID

        Returns:
            True if successful
        """
        try:
            # Remove from scheduler
            try:
                self.scheduler.remove_job(job_id)
            except JobLookupError:
                pass

            # Delete from database
            success = await self.storage.delete_cron_job(job_id)

            if success:
                logger.info(f"Deleted cron job: {job_id}")

            return success

        except Exception as e:
            logger.error(f"Error deleting cron job {job_id}: {e}")
            return False

    async def get_job_executions(self, job_id: str, limit: int = 50) -> List[Dict]:
        """Get execution history for a job.

        Args:
            job_id: Job ID
            limit: Maximum number of executions to return

        Returns:
            List of execution records
        """
        try:
            return await self.storage.get_cron_job_executions(job_id, limit=limit)
        except Exception as e:
            logger.error(f"Error getting executions for job {job_id}: {e}")
            return []

    async def restore_jobs_from_database(self):
        """Restore all enabled jobs from database on startup."""
        try:
            jobs = await self.storage.list_cron_jobs(enabled_only=True)

            for job in jobs:
                try:
                    await self._schedule_job(
                        job['job_id'],
                        job['name'],
                        job['schedule'],
                        job['config']
                    )
                    logger.info(f"Restored cron job: {job['name']} ({job['job_id']})")
                except Exception as e:
                    logger.error(f"Error restoring job {job['job_id']}: {e}")

            logger.info(f"Restored {len(jobs)} cron jobs from database")

        except Exception as e:
            logger.error(f"Error restoring jobs from database: {e}")

    def shutdown(self):
        """Shutdown the scheduler."""
        self.scheduler.shutdown()
        logger.info("CronJobManager shut down")
