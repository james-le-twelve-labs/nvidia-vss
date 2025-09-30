# Twelve Labs VSS Integration

## Overview

This document describes the integration between NVIDIA's Video Search and Summarization (VSS) Blueprint and Twelve Labs' video understanding platform. The integration enables VSS deployments to use Twelve Labs' cloud-based video analysis capabilities.

### What is NVIDIA VSS Blueprint?

NVIDIA VSS Blueprint is an AI-powered video analytics platform that processes video content using Vision Language Models (VLMs). VSS typically:
- Breaks videos into chunks and extracts frames for analysis
- Uses local VLMs (like VILA, NVILA) to generate video captions and embeddings
- Enables natural language search and summarization across video content
- Provides REST APIs for video upload, search, and summarization

### What is Twelve Labs?

Twelve Labs provides cloud-based video understanding APIs with two primary engines:
- **Marengo**: Video search and embedding model that understands visual and audio content for precise clip retrieval
- **Pegasus**: Video summarization model that generates comprehensive video summaries and answers questions

### Integration Summary

This integration adds a `twelve-labs` model to VSS that:
- **Automatically uploads** videos to Twelve Labs when added to VSS
- **Provides video search** via a new `/search` endpoint using Twelve Labs' Marengo engine
- **Enables multi-video summarization** using Twelve Labs' Pegasus engine
- **Maintains VSS compatibility** with minimal changes to core VSS architecture

## How It Works

### File Upload
1. Client uploads video to VSS `/files` endpoint
2. VSS stores video locally and automatically uploads to Twelve Labs
3. Twelve Labs indexes video in both Marengo (search) and Pegasus (summarization) engines
4. VSS maintains mapping between VSS file IDs and Twelve Labs video IDs

### Video Search  
1. Client searches via new VSS `/search` endpoint
2. VSS queries Twelve Labs Marengo index with natural language
3. Twelve Labs returns matching video clips with timestamps
4. VSS returns results using standard VSS response format

### Video Summarization
1. Client requests summary via VSS `/summarize` endpoint (supports multiple videos)
2. VSS sends request to Twelve Labs Pegasus engine
3. Twelve Labs processes entire videos (no chunking required)
4. VSS streams summarization results back to client

## API Endpoints

### Search Endpoint
```http
POST /search
Content-Type: application/json

{
  "query": "Find scenes with people running",
  "model": "twelve-labs",
  "id": ["uuid1", "uuid2"],          // Optional: search specific files
  "max_clips": 10,                   // Optional: number of clips to return (1-100)
  "threshold": "medium",             // Optional: low/medium/high
  "analyze": false,                  // Optional: include analysis
  "stream": false                    // Optional: stream response
}
```

Returns: Standard VSS `CompletionResponse` with search results

### Video Summarization
```http  
POST /summarize
Content-Type: application/json

{
  "model": "twelve-labs",
  "prompt": "What happens in these videos?",
  "stream": true,
  "id": ["uuid1", "uuid2", "uuid3"]  // Use string for single video, array for multiple
}
```

Returns: Standard VSS `CompletionResponse` with video summary

## Configuration

### Prerequisites

1. **Twelve Labs API Key**: Sign up at [Twelve Labs](https://twelvelabs.io/) to get your API key
2. **Docker & Docker Compose**: Required for deployment
3. **NVIDIA GPU**: Optional, needed only if using CV pipeline

### Environment Variables (.env)

```bash
# Required: Twelve Labs API Configuration
TWELVE_LABS_API_KEY="your_api_key_here"

# Optional: Twelve Labs Configuration  
TWELVE_LABS_MARENGO_INDEX_NAME="vss-marengo-search"      # Search index name
TWELVE_LABS_PEGASUS_INDEX_NAME="vss-pegasus-summarization" # Summarization index name

# Required: VSS Configuration
VLM_MODEL_TO_USE="twelve-labs"
DISABLE_CA_RAG=true
BACKEND_PORT=8080
FRONTEND_PORT=8090
```

**Note**: Change index names to create separate video collections for different environments.

### VSS Config (config.yaml)
```yaml
# VSS Configuration with Twelve Labs Integration
vlm:
  model_type: "twelve-labs"  # Unified model using Marengo/Pegasus
  batch_size: 1

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Deployment

### Quick Start
```bash
# 1. Navigate to Twelve Labs deployment
cd deploy/docker/twelve_labs_deployment

# 2. Edit .env file to set your API key
# TWELVE_LABS_API_KEY="your_actual_api_key_here"

# 3. Start services
docker compose up

# 4. Verify deployment
curl http://localhost:8080/health/ready
```

### File Structure
```
deploy/docker/twelve_labs_deployment/
├── compose.yaml              # Docker Compose configuration
├── config.yaml              # VSS configuration  
├── .env                     # Environment variables
├── assets/                   # Asset storage
├── logs/                     # Log storage
└── guardrails/              # NeMo Guardrails config

src/vss-engine/src/models/twelve_labs/
├── twelve_labs_model.py           # Main model implementation
├── twelve_labs_common.py          # Utilities and config
├── twelve_labs_context.py         # Context integration
└── twelve_labs_upload_handler.py  # Upload integration

src/vss-engine/src/
├── search_handler.py              # Search operations (new)
└── upload_handlers.py             # Upload callback registry (new)
```

## Key Integration Features

### Automatic Video Upload via Callback System
- **Upload Registry**: VSS loads upload handlers from configuration automatically
- **Callback Trigger**: Every video upload triggers the unified callback system
- **Twelve Labs Handler**: Automatically uploads videos to Twelve Labs cloud via callback
- **Index Management**: Twelve Labs indexes videos for search (Marengo) and summarization (Pegasus)  
- **ID Mapping**: VSS stores the mapping between VSS video IDs and Twelve Labs video IDs

### Multi-Video Processing
- **VSS Challenge**: VSS normally breaks videos into chunks and frames for processing
- **Twelve Labs Solution**: Processes entire videos directly without chunking
- **Integration Approach**: VSS detects twelve-labs model and skips frame extraction for multi-video requests
- **Video ID Extraction**: Supports multiple detection methods (config, semicolon-separated paths, legacy kwargs)
- **Pre-flight Validation**: Ensures all videos are uploaded and indexed before processing
- **Result**: Multiple videos can be processed together with robust error handling and validation

### Twelve Labs Index Management
- **Index Names**: Configure `TWELVE_LABS_MARENGO_INDEX_NAME` and `TWELVE_LABS_PEGASUS_INDEX_NAME` to create separate video collections
- **Automatic Creation**: VSS checks if indexes exist, creates them automatically if missing
- **Index Isolation**: Different index names create separate video collections (useful for dev/staging/prod environments)
- **Fresh Start**: Change index names to start with clean video collections
- **VSS Integration**: VSS manages video ID mappings between VSS and Twelve Labs indexes

### Error Handling
- **VSS Side**: Validates video files and IDs before sending to Twelve Labs
- **Twelve Labs Side**: Handles upload failures, processing errors, and API timeouts
- **Multi-Video**: If some videos fail, successful ones still return results

## Limitations

### Video Management
- **No Delete Functionality**: VSS cannot delete videos from Twelve Labs indexes once uploaded
- **Persistent Storage**: Videos remain in Twelve Labs indexes after VSS restart or container recreation
- **Manual Cleanup**: Video removal requires manual deletion through Twelve Labs API or dashboard

### Video Requirements
Videos must meet Twelve Labs requirements (most restrictive apply when both Marengo and Pegasus enabled):
- **Resolution**: Minimum 360x360, maximum 3840x2160
- **Aspect Ratio**: Must be 1:1, 4:3, 4:5, 5:4, 16:9, 9:16, or 17:9
- **Duration**: 4 seconds to 60 minutes (Pegasus limit), 4 seconds to 2 hours (Marengo limit)
- **File Size**: Maximum 2 GB
- **Formats**: Must be supported by FFmpeg (see FFmpeg Formats Documentation)

### Integration Constraints
- **Index Persistence**: Changing index names creates new indexes but doesn't remove old ones

---

## Technical Implementation Details

### VSS Code Changes Summary

**Modified VSS Core Files (4)**:
- `via_server.py` - Added `/search` endpoint and upload callback integration
- `via_stream_handler.py` - Refactored multi-video processing logic  
- `vlm_pipeline.py` - Added `twelve-labs` as new VLM model type
- `asset_manager.py` - Added upload callback system infrastructure

**New VSS Core Files (2)**:
- `upload_handlers.py` - Generic upload callback registry system
- `search_handler.py` - Dedicated video search processing

### Integration Architecture

#### Upload Callback System
VSS now supports upload callbacks that execute automatically when videos are added:

```python
# VSS initializes callback system on startup
from upload_handlers import upload_registry
upload_registry.load_handlers_from_config()
upload_callback = upload_registry.get_upload_callback()

# AssetManager calls callback after saving each video
self._asset_manager = AssetManager(
    asset_dir=asset_dir,
    asset_upload_callback=upload_callback
)
```

#### Multi-Video Processing
VSS detects `twelve-labs` model requests with multiple videos via several methods:

1. **Video ID Detection**: From `generation_config.video_ids` parameter
2. **Path Analysis**: From semicolon-separated file paths in chunk info
3. **Legacy Support**: From `video_ids` in kwargs

```python
# VSS detects multi-video requests and extracts video IDs
if isinstance(generation_config, dict) and "video_ids" in generation_config:
    video_ids_from_config = generation_config["video_ids"]
    
# Or extracts from semicolon-separated paths: "./assets/{uuid1}/file1.mp4;./assets/{uuid2}/file2.mp4"
if ';' in chunks[0].file:
    import re
    paths = chunks[0].file.split(';')
    video_ids = []
    for path in paths:
        match = re.search(r'/([0-9a-f-]{36})/', path)
        if match:
            video_ids.append(match.group(1))
```

#### Search Integration
New `/search` endpoint routes directly to Twelve Labs without VSS's normal video processing:

```python
@app.post("/search")
async def search(query: SearchQuery) -> CompletionResponse:
    if query.model != "twelve-labs":
        raise ViaException("Search only supported with twelve-labs model")
    return await self._search_handler.handle_search(query)
```

### Twelve Labs Components

- **TwelveLabsModel**: Main bridge implementing VSS's `CustomModelBase` interface
- **SearchHandler**: Processes search requests via Marengo API
- **twelve_labs_common.py**: Video ID mapping and index management utilities
- **twelve_labs_upload_handler.py**: Automatic upload callback implementation

This integration maintains clean separation between VSS infrastructure and Twelve Labs functionality while enabling powerful cross-video search and multi-video summarization capabilities.

## Usage Examples

### Upload Video
```python
import requests
import os

VSS_BASE_URL = "http://localhost:8080"

def upload_video(video_path):
    """Upload video to VSS (automatically syncs to Twelve Labs)."""
    with open(video_path, 'rb') as f:
        files = {'file': (os.path.basename(video_path), f, 'video/mp4')}
        data = {'purpose': 'vision', 'media_type': 'video'}
        response = requests.post(f"{VSS_BASE_URL}/files", files=files, data=data)
    
    if response.status_code in [200, 201]:
        file_id = response.json()['id']
        print(f"Uploaded video with ID: {file_id}")
        return file_id
    else:
        print(f"Upload failed: {response.status_code} - {response.text}")
        return None
```

### List Files
```python
def list_files():
    """List all uploaded files."""
    response = requests.get(f"{VSS_BASE_URL}/files?purpose=vision")
    if response.status_code == 200:
        files = response.json()['data']
        for f in files:
            print(f"{f['id']}: {f.get('filename', 'Unknown')}")
        return files
    return []
```

### Search Videos
```python
def search_videos(query, max_clips=10):
    """Search across all videos using natural language."""
    payload = {
        "query": query,
        "model": "twelve-labs",
        "max_clips": max_clips,
        "threshold": "medium",
        "stream": True
    }
    
    response = requests.post(f"{VSS_BASE_URL}/search", json=payload, stream=True)
    
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    data_content = decoded_line[12:]
                    try:
                        chunk = json.loads(data_content)
                        if 'choices' in chunk and chunk['choices']:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                print(delta['content'], end='', flush=True)
                        return chunk
                    except json.JSONDecodeError:
                        pass
```

### Summarize Videos
```python
def summarize_videos(prompt, video_ids, stream=True):
    """Summarize one or more videos."""
    # Ensure video_ids is a list
    if isinstance(video_ids, str):
        video_ids = [video_ids]
    
    payload = {
        "model": "twelve-labs",
        "prompt": prompt,
        "stream": stream,
        "id": video_ids[0] if len(video_ids) == 1 else video_ids
    }
    
    response = requests.post(f"{VSS_BASE_URL}/summarize", json=payload, stream=stream)
    
    if response.status_code == 200:
        if stream:
            # Handle streaming response
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        data_content = decoded_line[6:]
                        if data_content == '[DONE]':
                            break
                        try:
                            import json
                            chunk = json.loads(data_content)
                            if 'choices' in chunk and chunk['choices']:
                                choice = chunk['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    print(choice['delta']['content'], end='', flush=True)
                        except json.JSONDecodeError:
                            pass
        else:
            # Handle non-streaming response
            for line in response.text.splitlines():
                if line.startswith('data: '):
                    data_content = line[6:]
                    data = json.loads(data_content)
                    content = data['choices'][0]['message']['content']
                    return content

# Example usage
if __name__ == "__main__":
    # Upload a video (Note: This video_id is not your TwelveLabs video ID, but the internal NVIDIA VSS video ID mapping.)
    video_id = upload_video("sample_video.mp4")
    
    # Search for content
    search_videos("people talking", max_clips=5)
    
    # Summarize single video
    summarize_videos("What happens in this video?", video_id)
    
    # Summarize multiple videos
    video_ids = ["id1", "id2", "id3"]
    summarize_videos("Compare these cooking videos", video_ids)
```