# Cron Job Management API

This document describes the API endpoints for managing automated batch processing cron jobs.

## Overview

The Cron Job Management API allows you to:
- Create scheduled jobs that automatically fetch unprocessed vision analysis jobs from an external API
- Download images and descriptions from provided URLs
- Process them through batch analysis
- Upload HTML reports back to the external API
- Activate/deactivate jobs dynamically
- Monitor job execution history

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Create Cron Job

Create a new scheduled batch processing job.

**Endpoint:** `POST /cron-jobs`

**Request Body:**
```json
{
  "name": "Vision Analysis Job",
  "schedule": "0 2 * * *",
  "external_api_get_url": "https://api.example.com/api/v1/documents/vision_analysis",
  "external_api_post_url": "https://api.example.com/api/v1/documents/vision_analysis/upload",
  "external_api_key": "your_jwt_token_here",
  "backend_api_url": "http://localhost:8000",
  "organization_id": "38162954-469A-40BC-B619-3E9F71DE6DB9",
  "enabled": true
}
```

**Fields:**
- `name` (required): Descriptive name for the job
- `schedule` (required): Cron expression (see schedule format below)
- `external_api_get_url` (required): URL to fetch unprocessed vision analysis jobs
- `external_api_post_url` (required): URL to upload analysis reports
- `external_api_key` (required): JWT token for API authentication
- `backend_api_url` (optional): Local backend URL (default: `http://localhost:8000`)
- `organization_id` (required): Organization identifier for filtering
- `enabled` (optional): Whether job is active (default: `true`)

**Schedule Format:**
Cron expression: `minute hour day month day_of_week`
- `0 * * * *` - Every hour
- `0 2 * * *` - Daily at 2 AM
- `*/30 * * * *` - Every 30 minutes
- `0 0,6,12,18 * * *` - At 12 AM, 6 AM, 12 PM, 6 PM
- `0 9 * * 1` - Every Monday at 9 AM

**Response:**
```json
{
  "status": "success",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Vision Analysis Job",
  "schedule": "0 2 * * *",
  "enabled": true
}
```

---

### 2. List All Cron Jobs

Get a list of all cron jobs.

**Endpoint:** `GET /cron-jobs?enabled_only=false`

**Query Parameters:**
- `enabled_only` (optional, boolean): Only return enabled jobs (default: false)

**Response:**
```json
{
  "cron_jobs": [
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Vision Analysis Job",
      "schedule": "0 2 * * *",
      "config": {
        "external_api_get_url": "https://api.example.com/api/v1/documents/vision_analysis",
        "external_api_post_url": "https://api.example.com/api/v1/documents/vision_analysis/upload",
        "external_api_key": "***",
        "backend_api_url": "http://localhost:8000",
        "organization_id": "38162954-469A-40BC-B619-3E9F71DE6DB9"
      },
      "enabled": true,
      "scheduled": true,
      "next_run": "2025-01-21T02:00:00",
      "created_at": "2025-01-20T10:30:00",
      "updated_at": "2025-01-20T10:30:00"
    }
  ]
}
```

---

### 3. Get Cron Job Details

Get details of a specific cron job.

**Endpoint:** `GET /cron-jobs/{job_id}`

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Vision Analysis Job",
  "schedule": "0 2 * * *",
  "config": {
    "external_api_get_url": "https://api.example.com/api/v1/documents/vision_analysis",
    "external_api_post_url": "https://api.example.com/api/v1/documents/vision_analysis/upload",
    "external_api_key": "***",
    "backend_api_url": "http://localhost:8000",
    "organization_id": "38162954-469A-40BC-B619-3E9F71DE6DB9"
  },
  "enabled": true,
  "scheduled": true,
  "next_run": "2025-01-21T02:00:00",
  "created_at": "2025-01-20T10:30:00",
  "updated_at": "2025-01-20T10:30:00"
}
```

---

### 4. Activate Cron Job

Enable/activate a cron job to start running on schedule.

**Endpoint:** `POST /cron-jobs/{job_id}/activate`

**Response:**
```json
{
  "status": "success",
  "message": "Cron job 550e8400-e29b-41d4-a716-446655440000 activated"
}
```

---

### 5. Deactivate Cron Job

Disable/deactivate a cron job to stop it from running.

**Endpoint:** `POST /cron-jobs/{job_id}/deactivate`

**Response:**
```json
{
  "status": "success",
  "message": "Cron job 550e8400-e29b-41d4-a716-446655440000 deactivated"
}
```

---

### 6. Delete Cron Job

Permanently delete a cron job.

**Endpoint:** `DELETE /cron-jobs/{job_id}`

**Response:**
```json
{
  "status": "success",
  "message": "Cron job 550e8400-e29b-41d4-a716-446655440000 deleted"
}
```

---

### 7. Get Job Execution History

Get execution history for a cron job.

**Endpoint:** `GET /cron-jobs/{job_id}/executions?limit=50`

**Query Parameters:**
- `limit` (optional, integer): Maximum number of executions to return (default: 50)

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "executions": [
    {
      "execution_id": "exec-123",
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "success",
      "started_at": "2025-01-20T02:00:00",
      "completed_at": "2025-01-20T02:15:30",
      "output": "Processed 15 images successfully",
      "batch_id": "batch-456"
    },
    {
      "execution_id": "exec-122",
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "failed",
      "started_at": "2025-01-19T02:00:00",
      "completed_at": "2025-01-19T02:02:15",
      "output": "Error: External API returned 500",
      "batch_id": null
    }
  ]
}
```

## Usage Examples

### cURL Examples

#### Create a Cron Job
```bash
curl -X POST http://localhost:8000/cron-jobs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Daily Vision Analysis",
    "schedule": "0 2 * * *",
    "external_api_get_url": "https://api.example.com/api/v1/documents/vision_analysis",
    "external_api_post_url": "https://api.example.com/api/v1/documents/vision_analysis/upload",
    "external_api_key": "your_jwt_token",
    "backend_api_url": "http://localhost:8000",
    "organization_id": "38162954-469A-40BC-B619-3E9F71DE6DB9",
    "enabled": true
  }'
```

#### List All Active Jobs
```bash
curl http://localhost:8000/cron-jobs?enabled_only=true
```

#### Activate a Job
```bash
curl -X POST http://localhost:8000/cron-jobs/550e8400-e29b-41d4-a716-446655440000/activate
```

#### Deactivate a Job
```bash
curl -X POST http://localhost:8000/cron-jobs/550e8400-e29b-41d4-a716-446655440000/deactivate
```

#### Get Execution History
```bash
curl http://localhost:8000/cron-jobs/550e8400-e29b-41d4-a716-446655440000/executions?limit=20
```

### Python Examples

```python
import requests

BASE_URL = "http://localhost:8000"

# Create a cron job
response = requests.post(f"{BASE_URL}/cron-jobs", json={
    "name": "Hourly Image Analysis",
    "schedule": "0 * * * *",
    "external_api_get_url": "https://api.example.com/api/v1/documents/vision_analysis",
    "external_api_post_url": "https://api.example.com/api/v1/documents/vision_analysis/upload",
    "external_api_key": "your_jwt_token",
    "backend_api_url": BASE_URL,
    "organization_id": "38162954-469A-40BC-B619-3E9F71DE6DB9",
    "enabled": True
})
job_id = response.json()["job_id"]
print(f"Created job: {job_id}")

# List all jobs
jobs = requests.get(f"{BASE_URL}/cron-jobs").json()
print(f"Total jobs: {len(jobs['cron_jobs'])}")

# Get job details
job = requests.get(f"{BASE_URL}/cron-jobs/{job_id}").json()
print(f"Job schedule: {job['schedule']}")
print(f"Next run: {job.get('next_run', 'Not scheduled')}")

# Deactivate job
requests.post(f"{BASE_URL}/cron-jobs/{job_id}/deactivate")
print(f"Job {job_id} deactivated")

# Re-activate job
requests.post(f"{BASE_URL}/cron-jobs/{job_id}/activate")
print(f"Job {job_id} activated")

# Get execution history
executions = requests.get(
    f"{BASE_URL}/cron-jobs/{job_id}/executions",
    params={"limit": 10}
).json()
print(f"Recent executions: {len(executions['executions'])}")
for exec in executions['executions']:
    print(f"  - {exec['started_at']}: {exec['status']}")
```

## How It Works

### Job Execution Flow

1. **Scheduled Trigger**: APScheduler triggers the job based on the cron schedule
2. **External API Call**: Job calls the external GET API to fetch all unprocessed vision analysis jobs:
   - URL: `{external_api_get_url}?is_processed=false`
   - Returns all unprocessed jobs with:
     - `vision_analysis_id`
     - `user_query` (analysis prompt)
     - `documents` array (images and optional description JSON)
     - `organization_name`
3. **Process Each Job**: For each unprocessed job in the list:
   - **Download Images**: Downloads all images from URLs in documents array
   - **Download Descriptions**: If a JSON document is present, downloads and parses it
   - **Upload to Backend**: Uploads images to the local backend API
   - **Batch Processing**: Triggers batch analysis with descriptions and organization filter
   - **Wait for Completion**: Polls the batch status until complete (max 1 hour per job)
   - **Download Report**: Retrieves the HTML report from backend
   - **Upload to External API**: Posts report to external API:
     - URL: `{external_api_post_url}`
     - Form data: `file`, `organization_id`, `vision_analysis_id`
4. **Summary & Record**: Logs execution summary (total, successful, failed) in the database

### Job States

- **Enabled**: Job is active and will run on schedule
- **Disabled**: Job is paused and won't run
- **Scheduled**: Job is loaded in the scheduler (runtime state)

### Execution Statuses

- `success`: Job completed successfully
- `failed`: Job failed during execution
- `timeout`: Job exceeded the 1-hour timeout
- `error`: Unexpected error occurred

## Integration with External API

### GET Endpoint Requirements

Your external API should return JSON in this format:

```json
{
  "code": 200,
  "status": "success",
  "data": {
    "page": 0,
    "page_size": 0,
    "total": 1,
    "vision_analysis": [
      {
        "vision_analysis_id": "6004729D-E2C2-43D3-833C-79CE18E00646",
        "vision_analysis_run_code": "VISION_RUN_001",
        "is_processed": false,
        "organization_id": "38162954-469A-40BC-B619-3E9F71DE6DB9",
        "organization_name": "LexCorp",
        "user_query": "Analyze these images for defects and anomalies",
        "created_date": "2025-11-21T18:55:19Z",
        "documents": [
          {
            "vision_analysis_documents_id": "D1356F80-84AB-4920-BCDF-1F3D29EBA300",
            "vision_analysis_id": "6004729D-E2C2-43D3-833C-79CE18E00646",
            "document_id": "A85B12FE-C172-4F59-A70D-9F4F10F89DA2",
            "document_path": "https://storage.blob.core.windows.net/container/image1.jpg",
            "document_name": "image1.jpg",
            "is_input": true,
            "document_type": "image"
          },
          {
            "vision_analysis_documents_id": "08D16084-4247-4712-B417-FBCD9EF45896",
            "vision_analysis_id": "6004729D-E2C2-43D3-833C-79CE18E00646",
            "document_id": "D3D95A0E-6B4F-4713-84EF-07F90B615550",
            "document_path": "https://storage.blob.core.windows.net/container/descriptions.json",
            "document_name": "descriptions.json",
            "is_input": true,
            "document_type": "json"
          }
        ]
      }
    ]
  }
}
```

**Key Fields:**
- `code`: Must be `200` for success
- `status`: Must be `"success"`
- `vision_analysis[].is_processed`: Must be `false` for unprocessed jobs
- `documents[].document_type`: Either `"image"` or `"json"` (for descriptions)
- `documents[].is_input`: Must be `true` for input documents

**Description File Format** (`document_type: "json"`):
```json
{
  "image1.jpg": "Port side hull section near waterline",
  "image2.jpg": "Starboard deck area with recent repairs"
}
```

### POST Endpoint Requirements

The cron job will POST the analysis report to your external API with:

**URL:** `{external_api_post_url}`

**Headers:**
```
Authorization: Bearer {external_api_key}
```

**Form Data:**
- `file`: HTML report file (Content-Type: text/html)
- `organization_id`: Organization identifier
- `vision_analysis_id`: Vision analysis job ID

## Monitoring and Management

### View All Active Jobs
```bash
curl http://localhost:8000/cron-jobs?enabled_only=true
```

### Check Job Status
```bash
curl http://localhost:8000/cron-jobs/{job_id}
```

### Monitor Executions
```bash
curl http://localhost:8000/cron-jobs/{job_id}/executions
```

### Temporarily Pause a Job
```bash
curl -X POST http://localhost:8000/cron-jobs/{job_id}/deactivate
```

### Resume a Paused Job
```bash
curl -X POST http://localhost:8000/cron-jobs/{job_id}/activate
```

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK`: Success
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error response format:
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Best Practices

1. **Schedule Wisely**: Don't schedule jobs too frequently. Consider processing time and external API limits.

2. **Monitor Executions**: Regularly check execution history to catch failures early.

3. **Use Organization Filters**: Use the `organization_id` parameter to filter jobs and results appropriately.

4. **Secure API Keys**: Never expose JWT tokens in logs or error messages. The API automatically redacts sensitive config data.

5. **Test Before Enabling**: Create jobs in disabled state first (`enabled: false`), verify configuration, then activate.

6. **Clean Up Old Jobs**: Delete jobs that are no longer needed to keep the database clean.

7. **Handle Timeouts**: Jobs timeout after 1 hour of processing. Adjust batch sizes if consistently hitting timeouts.

## Troubleshooting

### Job Not Running

1. Check if job is enabled:
   ```bash
   curl http://localhost:8000/cron-jobs/{job_id}
   ```

2. Check execution history for errors:
   ```bash
   curl http://localhost:8000/cron-jobs/{job_id}/executions
   ```

3. Verify the schedule expression is valid

4. Check backend logs for scheduler errors:
   ```bash
   tail -f /var/log/batch_processor.log
   ```

### External API Errors

Check execution output for HTTP errors:
```bash
curl http://localhost:8000/cron-jobs/{job_id}/executions
```

Common issues:
- Invalid or expired JWT token
- API rate limiting
- Network connectivity
- Invalid response format
- No unprocessed jobs available

### Image Download Failures

Common issues:
- Images not accessible (403/404 errors)
- Expired SAS tokens (if using Azure Blob Storage)
- Network connectivity issues
- Invalid URLs in `document_path`

## Future Enhancements

Planned features:
- Webhook notifications on job completion/failure
- Email alerts for failed jobs
- Job statistics dashboard
- Retry logic for failed executions
- Job priority and queuing
- Concurrent job limits
- Custom timeout configuration per job
