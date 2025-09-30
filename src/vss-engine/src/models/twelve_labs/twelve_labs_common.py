######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
######################################################################################################

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

from via_logger import logger

ASSET_STORAGE_DIR = os.environ.get("ASSET_STORAGE_DIR")


class TwelveLabsConfig:
    """Configuration class for Twelve Labs integration."""
    
    def __init__(self):
        # API Configuration
        self.api_key = self._get_env_var("TWELVE_LABS_API_KEY")
        self.api_url = self._get_env_var("TWELVE_LABS_API_URL", "https://api.twelvelabs.io")
        
        # Index Configuration
        self.marengo_index_name = self._get_env_var("TWELVE_LABS_MARENGO_INDEX_NAME", "vss-marengo-index")
        self.pegasus_index_name = self._get_env_var("TWELVE_LABS_PEGASUS_INDEX_NAME", "vss-pegasus-index")
        self.marengo_model = self._get_env_var("TWELVE_LABS_MARENGO_MODEL", "marengo-2.7")
        self.pegasus_model = self._get_env_var("TWELVE_LABS_PEGASUS_MODEL", "pegasus-1")
        
        # Search Options
        self.marengo_options = self._get_env_list("TWELVE_LABS_MARENGO_OPTIONS", ["visual", "audio"])
        self.pegasus_options = self._get_env_list("TWELVE_LABS_PEGASUS_OPTIONS", ["visual", "audio"])
        self.search_options = self._get_env_list("TWELVE_LABS_SEARCH_OPTIONS", ["visual", "audio"])
        
        # Search Parameters
        self.max_clips = self._get_env_int("TWELVE_LABS_MAX_CLIPS", 10)
        self.max_clips_for_analysis = self._get_env_int("TWELVE_LABS_MAX_CLIPS_FOR_ANALYSIS", 3)
        self.search_threshold = self._get_env_var("TWELVE_LABS_SEARCH_THRESHOLD", "medium")
        
        # Analysis Parameters
        self.analysis_temperature = self._get_env_float("TWELVE_LABS_ANALYSIS_TEMPERATURE", 0.7)
        
        # Upload Configuration
        self.upload_timeout = self._get_env_int("TWELVE_LABS_UPLOAD_TIMEOUT", 600)
        self.auto_upload_enabled = self._get_env_bool("VSS_ENABLE_TWELVE_LABS_AUTO_UPLOAD", False)
        
        # Retry Configuration
        self.max_retries = self._get_env_int("TWELVE_LABS_MAX_RETRIES", 3)
        self.retry_delay_base = self._get_env_int("TWELVE_LABS_RETRY_DELAY_BASE", 2)
        
    def _get_env_var(self, key: str, default: str = None) -> str:
        value = os.environ.get(key, default)
        if value is None:
            logger.warning(f"Environment variable {key} not set")
        return value
    
    def _get_env_int(self, key: str, default: int) -> int:
        try:
            return int(os.environ.get(key, str(default)))
        except ValueError:
            logger.warning(f"Invalid integer value for {key}, using default: {default}")
            return default
    
    def _get_env_float(self, key: str, default: float) -> float:
        try:
            return float(os.environ.get(key, str(default)))
        except ValueError:
            logger.warning(f"Invalid float value for {key}, using default: {default}")
            return default
    
    def _get_env_bool(self, key: str, default: bool) -> bool:
        value = os.environ.get(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def _get_env_list(self, key: str, default: list) -> list:
        value = os.environ.get(key)
        if value:
            return [item.strip() for item in value.split(',')]
        return default
    
    def validate(self) -> bool:
        """Validate the configuration."""
        if not self.api_key:
            logger.error("TWELVE_LABS_API_KEY is required")
            return False
        
        if not ASSET_STORAGE_DIR:
            logger.error("ASSET_STORAGE_DIR is required")
            return False
            
        return True


class VideoIDMapper:
    # Cache for video ID mappings to avoid repeated file I/O
    _mapping_cache = {}
    _cache_max_size = 1000
    
    @staticmethod
    def save_mapping(vss_file_id: str, marengo_video_id: str = None, 
                     pegasus_video_id: str = None, marengo_index_id: str = None,
                     pegasus_index_id: str = None) -> None:
        asset_dir = Path(ASSET_STORAGE_DIR) / vss_file_id
        if not asset_dir.exists():
            asset_dir = Path(ASSET_STORAGE_DIR) / vss_file_id.replace("-", "")
        
        mapping_file = asset_dir / "twelve_labs_mapping.json"
        
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
        else:
            mapping = {"vss_file_id": vss_file_id}
        
        if marengo_video_id:
            mapping["marengo_video_id"] = marengo_video_id
        if pegasus_video_id:
            mapping["pegasus_video_id"] = pegasus_video_id
        if marengo_index_id:
            mapping["marengo_index_id"] = marengo_index_id
        if pegasus_index_id:
            mapping["pegasus_index_id"] = pegasus_index_id
        
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f)
        
        # Update cache
        VideoIDMapper._update_cache(vss_file_id, mapping)
    
    @staticmethod
    def get_mapping(vss_file_id: str) -> Optional[Dict]:
        # Check cache first
        if vss_file_id in VideoIDMapper._mapping_cache:
            return VideoIDMapper._mapping_cache[vss_file_id]
        
        # Check alternative file ID format in cache
        alt_file_id = vss_file_id.replace("-", "")
        if alt_file_id in VideoIDMapper._mapping_cache:
            return VideoIDMapper._mapping_cache[alt_file_id]
        
        # Load from file system
        for file_id in [vss_file_id, alt_file_id]:
            mapping_file = Path(ASSET_STORAGE_DIR) / file_id / "twelve_labs_mapping.json"
            if mapping_file.exists():
                with open(mapping_file, 'r') as f:
                    mapping = json.load(f)
                    VideoIDMapper._update_cache(file_id, mapping)
                    return mapping
        return None
    
    @staticmethod
    def _update_cache(vss_file_id: str, mapping: Dict) -> None:
        """Update the mapping cache, implementing LRU eviction."""
        if len(VideoIDMapper._mapping_cache) >= VideoIDMapper._cache_max_size:
            # Remove oldest item (simple FIFO for now)
            oldest_key = next(iter(VideoIDMapper._mapping_cache))
            del VideoIDMapper._mapping_cache[oldest_key]
        
        VideoIDMapper._mapping_cache[vss_file_id] = mapping
    
    @staticmethod
    def get_pegasus_for_marengo(marengo_video_id: str) -> Optional[str]:
        """Optimized lookup to get Pegasus video ID for a given Marengo video ID."""
        # First check cache for reverse mapping
        for cached_mapping in VideoIDMapper._mapping_cache.values():
            if cached_mapping.get("marengo_video_id") == marengo_video_id:
                pegasus_id = cached_mapping.get("pegasus_video_id")
                if pegasus_id:
                    logger.debug(f"Found Pegasus video ID {pegasus_id} in cache for Marengo video ID {marengo_video_id}")
                    return pegasus_id
        
        # If not in cache, do file system scan (but cache results)
        if not ASSET_STORAGE_DIR:
            return None
            
        for asset_dir in Path(ASSET_STORAGE_DIR).iterdir():
            if asset_dir.is_dir():
                mapping_file = asset_dir / "twelve_labs_mapping.json"
                if mapping_file.exists():
                    # Load and cache the mapping
                    mapping = VideoIDMapper.get_mapping(asset_dir.name)
                    if mapping and mapping.get("marengo_video_id") == marengo_video_id:
                        pegasus_video_id = mapping.get("pegasus_video_id")
                        logger.debug(f"Found Pegasus video ID {pegasus_video_id} for Marengo video ID {marengo_video_id}")
                        return pegasus_video_id
        
        logger.warning(f"No Pegasus video ID found for Marengo video {marengo_video_id}")
        return None
    
    @staticmethod
    def clear_cache():
        """Clear the mapping cache. Useful for testing or memory management."""
        VideoIDMapper._mapping_cache.clear()
    
    @staticmethod
    def get_vss_id_for_marengo(marengo_video_id: str) -> Optional[str]:
        """Get VSS file ID for a given Marengo video ID."""
        logger.info(f"Looking up VSS file ID for Marengo video ID: {marengo_video_id}")
        logger.info(f"ASSET_STORAGE_DIR: {ASSET_STORAGE_DIR}")
        logger.info(f"Cache size: {len(VideoIDMapper._mapping_cache)}")
        
        # First check cache for reverse mapping
        for vss_id, cached_mapping in VideoIDMapper._mapping_cache.items():
            logger.debug(f"Checking cache entry {vss_id}: {cached_mapping}")
            if cached_mapping.get("marengo_video_id") == marengo_video_id:
                logger.info(f"Found VSS file ID {vss_id} in cache for Marengo video ID {marengo_video_id}")
                return vss_id
        
        # If not in cache, do file system scan
        if not ASSET_STORAGE_DIR:
            logger.error("ASSET_STORAGE_DIR is not set")
            return None
            
        logger.info(f"Scanning asset directories in {ASSET_STORAGE_DIR}")
        for asset_dir in Path(ASSET_STORAGE_DIR).iterdir():
            if asset_dir.is_dir():
                logger.debug(f"Checking directory: {asset_dir.name}")
                mapping_file = asset_dir / "twelve_labs_mapping.json"
                if mapping_file.exists():
                    # Load and cache the mapping
                    mapping = VideoIDMapper.get_mapping(asset_dir.name)
                    logger.debug(f"Loaded mapping for {asset_dir.name}: {mapping}")
                    if mapping and mapping.get("marengo_video_id") == marengo_video_id:
                        vss_file_id = mapping.get("vss_file_id", asset_dir.name)
                        logger.info(f"Found VSS file ID {vss_file_id} for Marengo video ID {marengo_video_id}")
                        return vss_file_id
                else:
                    logger.debug(f"No mapping file found in {asset_dir}")
        
        logger.warning(f"No VSS file ID found for Marengo video {marengo_video_id}")
        logger.info("Available mappings:")
        for vss_id, mapping in VideoIDMapper._mapping_cache.items():
            logger.info(f"  {vss_id}: marengo_video_id={mapping.get('marengo_video_id')}")
        return None
    
    @staticmethod
    def get_video_path(vss_file_id: str) -> Optional[Path]:
        for file_id in [vss_file_id, vss_file_id.replace("-", "")]:
            asset_dir = Path(ASSET_STORAGE_DIR) / file_id
            if asset_dir.exists():
                for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                    video_files = list(asset_dir.glob(f"*{ext}"))
                    if video_files:
                        return video_files[0]
        return None


def wait_for_task_completion(client, task_id: str, timeout_seconds: int = None) -> Dict:
    if timeout_seconds is None:
        timeout_seconds = int(os.environ.get("TWELVE_LABS_UPLOAD_TIMEOUT", "600"))
    
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        try:
            task = client.task.retrieve(task_id)
            
            if task.status == "ready":
                return {"status": "ready", "video_id": task.video_id}
            elif task.status == "failed":
                return {"status": "failed", "error": "Task failed"}
            
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error checking task: {e}")
            time.sleep(5)
    
    return {"status": "timeout", "error": "Upload timeout"}


def retry_with_exponential_backoff(func, max_retries: int = 3, base_delay: int = 2):
    """Retry a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            delay = base_delay ** attempt
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
    
    return None


def ensure_index_exists(client, index_name: str, engine_name: str, engine_options: list) -> str:
    """Ensure that a Twelve Labs index exists, creating it if necessary."""
    try:
        indexes = list(client.indexes.list())
        
        for index in indexes:
            if index.index_name == index_name:
                logger.info(f"Using existing index: {index_name}")
                return index.id
        
        logger.info(f"Creating index: {index_name}")
        index = client.indexes.create(
            index_name=index_name,
            models=[{"model_name": engine_name, "model_options": engine_options}]
        )
        return index.id
        
    except Exception as e:
        if "already_exists" in str(e):
            indexes = list(client.indexes.list())
            for index in indexes:
                if index.index_name == index_name:
                    return index.id
        raise e