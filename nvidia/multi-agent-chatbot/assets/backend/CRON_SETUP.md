# Cron Batch Processor Setup Guide

This guide explains how to set up the automated batch processing cron job that:
1. Calls an external Vision Analysis API to get unprocessed jobs
2. Downloads images and descriptions from provided URLs
3. Processes them through your batch analysis backend
4. Uploads the HTML report back to the external API

## Prerequisites

1. Python 3.10+
2. External Vision Analysis API endpoint with JWT authentication
3. Backend API running and accessible
4. Access to image URLs (can be from Azure Blob Storage or any HTTP-accessible location)

## Installation

### 1. Install Dependencies

```bash
cd /path/to/backend
pip install -e .
```

This will install all required dependencies including:
- `requests` for API calls
- `apscheduler` for cron job management

### 2. Create Configuration File

Copy the example configuration:

```bash
sudo mkdir -p /etc/batch_processor
sudo cp cron_config.example.json /etc/batch_processor/config.json
sudo chmod 600 /etc/batch_processor/config.json
```

Edit the configuration file:

```bash
sudo nano /etc/batch_processor/config.json
```

Update with your actual values:

```json
{
  "external_api_get_url": "https://your-api.com/api/v1/documents/vision_analysis",
  "external_api_post_url": "https://your-api.com/api/v1/documents/vision_analysis/upload",
  "external_api_key": "your_jwt_token_here",
  "backend_api_url": "http://localhost:8000",
  "organization_id": "your_organization_id"
}
```

### 3. Make Script Executable

```bash
chmod +x cron_batch_processor.py
```

### 4. Create Log Directory

```bash
sudo mkdir -p /var/log
sudo touch /var/log/batch_processor.log
sudo chmod 666 /var/log/batch_processor.log
```

## External API Requirements

### GET Endpoint (Fetch Unprocessed Jobs)

**Endpoint:** `GET /api/v1/documents/vision_analysis`

**Query Parameters:**
- `is_processed=false` - Filter for unprocessed jobs
- `page=1` - Page number
- `page_size=1` - Return only the first unprocessed job

**Headers:**
```
Authorization: Bearer {jwt_token}
Content-Type: application/json
```

**Expected Response:**
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
            "document_path": "https://storage.blob.core.windows.net/container/oceanix_01.jpg",
            "document_name": "oceanix_01.jpg",
            "is_input": true,
            "document_type": "image"
          },
          {
            "vision_analysis_documents_id": "0B1DA115-449C-4E11-83F5-BDB6D240BE1D",
            "vision_analysis_id": "6004729D-E2C2-43D3-833C-79CE18E00646",
            "document_id": "5F5A099B-9825-4402-9104-E810193E460C",
            "document_path": "https://storage.blob.core.windows.net/container/oceanix_02.jpg",
            "document_name": "oceanix_02.jpg",
            "is_input": true,
            "document_type": "image"
          },
          {
            "vision_analysis_documents_id": "08D16084-4247-4712-B417-FBCD9EF45896",
            "vision_analysis_id": "6004729D-E2C2-43D3-833C-79CE18E00646",
            "document_id": "D3D95A0E-6B4F-4713-84EF-07F90B615550",
            "document_path": "https://storage.blob.core.windows.net/container/descriptions_VISION_RUN_001.json",
            "document_name": "descriptions_VISION_RUN_001.json",
            "is_input": true,
            "document_type": "json"
          }
        ]
      }
    ]
  }
}
```

**Description File Format** (optional JSON document):
```json
{
  "oceanix_01.jpg": "Port side hull section near waterline",
  "oceanix_02.jpg": "Starboard deck area with recent repairs"
}
```

### POST Endpoint (Upload Analysis Report)

**Endpoint:** `POST /api/v1/documents/vision_analysis/upload`

**Headers:**
```
Authorization: Bearer {jwt_token}
Content-Type: multipart/form-data
```

**Form Data:**
- `file` - HTML report file (e.g., `report_{batch_id}.html`)
- `organization_id` - Organization ID
- `vision_analysis_id` - Vision analysis ID from the GET response

## Testing the Script

### Manual Testing (Recommended Before Scheduling)

Test the script manually before setting up the cron job:

```bash
python3 cron_batch_processor.py --config /etc/batch_processor/config.json
```

Or with command-line arguments:

```bash
python3 cron_batch_processor.py \
  --external-api-get-url "https://api.example.com/api/v1/documents/vision_analysis" \
  --external-api-post-url "https://api.example.com/api/v1/documents/vision_analysis/upload" \
  --external-api-key "your_jwt_token" \
  --backend-api-url "http://localhost:8000" \
  --organization-id "your_org_id"
```

Check the logs:

```bash
tail -f /var/log/batch_processor.log
```

### Using the REST API (Alternative Testing Method)

You can also create and trigger a one-time cron job using the REST API:

**1. Create a cron job:**
```bash
curl -X POST http://localhost:8000/cron-jobs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Vision Analysis Job",
    "schedule": "0 * * * *",
    "external_api_get_url": "https://api.example.com/api/v1/documents/vision_analysis",
    "external_api_post_url": "https://api.example.com/api/v1/documents/vision_analysis/upload",
    "external_api_key": "your_jwt_token",
    "backend_api_url": "http://localhost:8000",
    "organization_id": "your_org_id",
    "enabled": false
  }'
```

**Response:**
```json
{
  "status": "success",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Test Vision Analysis Job",
  "schedule": "0 * * * *",
  "enabled": false
}
```

**2. Check the job details:**
```bash
curl http://localhost:8000/cron-jobs/550e8400-e29b-41d4-a716-446655440000
```

**3. For testing, the cron job will run automatically based on the schedule. To enable it:**
```bash
curl -X POST http://localhost:8000/cron-jobs/550e8400-e29b-41d4-a716-446655440000/activate
```

**4. Monitor execution history:**
```bash
curl http://localhost:8000/cron-jobs/550e8400-e29b-41d4-a716-446655440000/executions
```

## Setting Up Scheduled Cron Jobs via API

The recommended approach is to use the REST API to manage cron jobs:

### Create a Scheduled Job

```bash
curl -X POST http://localhost:8000/cron-jobs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Hourly Vision Analysis",
    "schedule": "0 * * * *",
    "external_api_get_url": "https://api.example.com/api/v1/documents/vision_analysis",
    "external_api_post_url": "https://api.example.com/api/v1/documents/vision_analysis/upload",
    "external_api_key": "your_jwt_token",
    "backend_api_url": "http://localhost:8000",
    "organization_id": "your_org_id",
    "enabled": true
  }'
```

### Schedule Examples

**Every Hour:**
```json
{"schedule": "0 * * * *"}
```

**Every Day at 2 AM:**
```json
{"schedule": "0 2 * * *"}
```

**Every 30 Minutes:**
```json
{"schedule": "*/30 * * * *"}
```

**Multiple Times Per Day (6 AM, 12 PM, 6 PM, 12 AM):**
```json
{"schedule": "0 0,6,12,18 * * *"}
```

**Every Monday at 9 AM:**
```json
{"schedule": "0 9 * * 1"}
```

### Manage Existing Jobs

**List all jobs:**
```bash
curl http://localhost:8000/cron-jobs
```

**Get specific job:**
```bash
curl http://localhost:8000/cron-jobs/{job_id}
```

**Activate job:**
```bash
curl -X POST http://localhost:8000/cron-jobs/{job_id}/activate
```

**Deactivate job:**
```bash
curl -X POST http://localhost:8000/cron-jobs/{job_id}/deactivate
```

**Delete job:**
```bash
curl -X DELETE http://localhost:8000/cron-jobs/{job_id}
```

**View execution history:**
```bash
curl http://localhost:8000/cron-jobs/{job_id}/executions?limit=20
```

## Environment Variables (Alternative to Config File)

You can also use environment variables instead of a config file:

```bash
export EXTERNAL_API_GET_URL="https://api.example.com/api/v1/documents/vision_analysis"
export EXTERNAL_API_POST_URL="https://api.example.com/api/v1/documents/vision_analysis/upload"
export EXTERNAL_API_KEY="your_jwt_token"
export BACKEND_API_URL="http://localhost:8000"
export ORGANIZATION_ID="your_org_id"
```

## Workflow

The cron job performs the following steps:

1. **Fetch All Jobs** - Calls external API with `is_processed=false` filter to get all unprocessed jobs
2. **Process Each Job** - For each unprocessed job:
   - Download all images from provided URLs
   - Download descriptions (if a JSON description file is included)
   - Upload images to your backend API
   - Trigger batch analysis with optional descriptions and organization filter
   - Wait for completion (max 1 hour timeout per job)
   - Download the HTML report from backend
   - Upload report to external API with vision_analysis_id and organization_id
3. **Summary** - Logs total jobs processed, successful, and failed counts

## Monitoring

### View Logs

```bash
# Main application logs
tail -f /var/log/batch_processor.log

# Follow logs in real-time
tail -f /var/log/batch_processor.log | grep -E "(INFO|ERROR|WARNING)"
```

### Check Job Status via API

```bash
# List all active jobs
curl http://localhost:8000/cron-jobs?enabled_only=true

# Get execution history
curl http://localhost:8000/cron-jobs/{job_id}/executions
```

### Manual Trigger

You can manually trigger the script at any time for testing:

```bash
python3 cron_batch_processor.py --config /etc/batch_processor/config.json
```

## Troubleshooting

### Script Not Finding Jobs

1. Check if there are unprocessed jobs:
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" \
     "https://api.example.com/api/v1/documents/vision_analysis?is_processed=false"
   ```

2. Verify API credentials in config file

3. Check logs for API errors:
   ```bash
   grep "API returned" /var/log/batch_processor.log
   ```

### Permission Issues

```bash
# Fix log file permissions
sudo chown $USER:$USER /var/log/batch_processor.log

# Fix config permissions
sudo chmod 600 /etc/batch_processor/config.json
```

### Backend API Issues

Test backend connectivity:

```bash
curl -X GET http://localhost:8000/available_models
```

### Image Download Failures

Check if images are accessible:

```bash
curl -I "https://storage.blob.core.windows.net/container/image.jpg"
```

If using Azure Blob Storage with SAS tokens, ensure:
- Tokens are not expired
- URLs include proper SAS parameters
- Container has appropriate read permissions

## Output Files

- **Logs**: `/var/log/batch_processor.log`
- **Temp Files**: `/tmp/batch_processor/` (auto-cleaned after upload)
- **Reports**: Uploaded to external API with vision_analysis_id

## Security Notes

1. **API Keys**: Never commit JWT tokens or API keys to version control
2. **Config File**: Set restrictive permissions (600) on config file
3. **HTTPS**: Always use HTTPS for external API endpoints
4. **Logs**: Ensure logs don't contain sensitive information (API keys are automatically redacted)

## Advanced Configuration

### Custom Timeout

Modify the script to change timeout:

```python
# In cron_batch_processor.py, line ~531
if not self.wait_for_completion(batch_id, timeout=7200):  # 2 hours
    ...
```

### Multiple Organizations

Create separate cron jobs for different organizations:

```bash
# Organization 1
curl -X POST http://localhost:8000/cron-jobs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Org1 Vision Analysis",
    "schedule": "0 */2 * * *",
    "organization_id": "org1-id",
    ...
  }'

# Organization 2
curl -X POST http://localhost:8000/cron-jobs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Org2 Vision Analysis",
    "schedule": "30 */2 * * *",
    "organization_id": "org2-id",
    ...
  }'
```

## Support

For issues or questions:
- Check the logs first: `/var/log/batch_processor.log`
- Verify API connectivity and authentication
- Test each component individually (fetch, download, upload)
- Review execution history via REST API
- See [CRON_API.md](CRON_API.md) for complete API documentation
