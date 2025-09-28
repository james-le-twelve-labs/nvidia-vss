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

import os
import time
import json
from typing import List, Optional, Dict, Generator, Any

import torch

from base_class import CustomModelBase, EmbeddingGeneratorBase, VlmGenerationConfig
from chunk_info import ChunkInfo
from via_logger import TimeMeasure, logger
from .twelve_labs_common import VideoIDMapper, ensure_index_exists, TwelveLabsConfig, retry_with_exponential_backoff

try:
    from twelvelabs import TwelveLabs
except ImportError:
    TwelveLabs = None
    logger.warning("TwelveLabs SDK not installed. Please install: pip install twelvelabs")


class TwelveLabsModel(CustomModelBase):
    def __init__(self, async_output: bool = True):
        self._client = None
        self._config = TwelveLabsConfig()
        self._initialize_client()
        
    def _initialize_client(self):
        if TwelveLabs is None:
            raise ImportError("TwelveLabs SDK not installed")
        
        if not self._config.validate():
            logger.error("Invalid Twelve Labs configuration")
            return
            
        try:
            self._client = TwelveLabs(api_key=self._config.api_key)
            indexes = list(self._client.indexes.list())
            logger.info(f"Connected to Twelve Labs. Found {len(indexes)} indexes")
        except Exception as e:
            logger.error(f"Failed to initialize client: {e}")
            self._client = None
    
    def generate(self, prompt: str, input_tensors: List[torch.Tensor], 
                 video_frames_times: List[List], generation_config: VlmGenerationConfig) -> List:
        """Generate video summarization using Twelve Labs Pegasus model."""
        logger.info(f"TwelveLabsModel.generate() called with prompt: {prompt[:50]}...")
        
        if self._client is None:
            return ["Client not available"]
        
        if not prompt:
            return ["No prompt provided"]
            
        # Note: TwelveLabs operates differently from other models as it works with 
        # uploaded videos rather than processing tensors directly. The actual video
        # processing is handled through the TwelveLabsContext which provides the 
        # necessary chunk information via the legacy generate_chunks() method.
        
        # This method maintains interface compatibility but delegates to chunk-based processing
        logger.warning("TwelveLabsModel.generate() called via standard interface - this should normally go through TwelveLabsContext")
        
        # Return a basic response indicating this should be called through the context
        return [f"TwelveLabs model requires video chunk information for summarization of: {prompt}"]
    
    def generate_chunks(self, chunks: List[ChunkInfo], frames: torch.Tensor, frame_times: List[float], 
                       generation_config=None, **kwargs) -> dict:
        """Generate video summarization using Twelve Labs Pegasus model (legacy interface)."""
        logger.info(f"TwelveLabsModel.generate_chunks() called with {len(chunks) if chunks else 0} chunks")
        
        if self._client is None:
            return {"text": "Client not available"}
        
        if isinstance(generation_config, dict):
            prompt = generation_config.get("prompt")
            stream = generation_config.get("stream", False)
        else:
            prompt = getattr(generation_config, "prompt", None) if generation_config else None
            stream = getattr(generation_config, "stream", False) if generation_config else False
        
        if not prompt:
            return {"text": "No prompt provided"}
        
        # Handle video summarization with chunks
        if not chunks or len(chunks) == 0:
            logger.warning("No chunks provided for summarization")
            return {"text": f"No video content available for summarization of: {prompt}"}
        
        logger.info(f"🔥 GENERATE_CHUNKS: received {len(chunks)} chunks")
        if len(chunks) > 0 and chunks[0].file:
            logger.info(f"🔥 GENERATE_CHUNKS: First chunk file path: {chunks[0].file}")
            # Check for semicolon-separated paths
            if ';' in chunks[0].file:
                logger.info(f"🔥 GENERATE_CHUNKS: DETECTED SEMICOLON-SEPARATED PATH: {chunks[0].file}")
                semicolon_paths = chunks[0].file.split(';')
                logger.info(f"🔥 GENERATE_CHUNKS: Split into {len(semicolon_paths)} paths: {semicolon_paths}")
        
        # Check if this is a multi-video request via video_ids in generation_config
        logger.info(f"🔥 GENERATE_CHUNKS: generation_config type: {type(generation_config)}, content: {generation_config}")
        video_ids_from_config = None
        if isinstance(generation_config, dict) and "video_ids" in generation_config:
            video_ids_from_config = generation_config["video_ids"]
            logger.info(f"🔥 GENERATE_CHUNKS: Found video_ids in dict generation_config: {video_ids_from_config}")
        elif generation_config and hasattr(generation_config, "video_ids"):
            video_ids_from_config = generation_config.video_ids
            logger.info(f"🔥 GENERATE_CHUNKS: Found video_ids in object generation_config: {video_ids_from_config}")
        
        logger.info(f"🔥 GENERATE_CHUNKS: Final extracted video_ids from config: {video_ids_from_config}")
        
        # If no video_ids in generation_config, try to extract from semicolon-separated paths
        if not video_ids_from_config and len(chunks) > 0 and chunks[0].file and ';' in chunks[0].file:
            logger.info(f"🔥 GENERATE_CHUNKS: No video_ids in config, attempting to extract from semicolon-separated paths")
            import re
            semicolon_paths = chunks[0].file.split(';')
            extracted_video_ids = []
            
            for path in semicolon_paths:
                path_parts = str(path).split('/')
                for part in path_parts:
                    if re.match(r'^[0-9a-f-]{36}$', part):  # UUID format
                        extracted_video_ids.append(part)
                        logger.info(f"🔥 GENERATE_CHUNKS: Extracted video ID from path '{path}': {part}")
                        break
            
            if len(extracted_video_ids) > 1:
                video_ids_from_config = extracted_video_ids
                logger.info(f"🔥 GENERATE_CHUNKS: Successfully extracted {len(extracted_video_ids)} video IDs: {video_ids_from_config}")
            else:
                logger.warning(f"🔥 GENERATE_CHUNKS: Could only extract {len(extracted_video_ids)} video IDs from paths, need at least 2 for multi-video")
        
        logger.info(f"🔥 GENERATE_CHUNKS: Final video_ids (after path extraction): {video_ids_from_config}")
        
        if video_ids_from_config and len(video_ids_from_config) > 1:
            logger.info(f"🔥 GENERATE_CHUNKS: MULTI-VIDEO DETECTED from generation_config: {len(video_ids_from_config)} videos - IDs: {video_ids_from_config}")
            if stream:
                logger.info(f"🔥 GENERATE_CHUNKS: CALLING _summarize_multiple_videos_stream with prompt='{prompt}', video_ids={video_ids_from_config}")
                return self._summarize_multiple_videos_stream(prompt, video_ids_from_config)
            else:
                logger.info(f"🔥 GENERATE_CHUNKS: CALLING _summarize_multiple_videos_concatenated with prompt='{prompt}', video_ids={video_ids_from_config}")
                result = self._summarize_multiple_videos_concatenated(prompt, video_ids_from_config)
                if result.get("error"):
                    logger.error(f"Multi-video concatenation failed: {result.get('text')}")
                    return {"text": result.get("text", "Multi-video summarization failed")}
                return {"text": result.get("text", "No summary generated")}
        
        # Check if this is a multi-video request via video_ids in kwargs (legacy support)
        video_ids = kwargs.get('video_ids', None)
        logger.info(f"🔥 GENERATE_CHUNKS: Checking legacy video_ids in kwargs: {video_ids}")
        if video_ids and isinstance(video_ids, list) and len(video_ids) > 1:
            logger.info(f"🔥 GENERATE_CHUNKS: MULTI-VIDEO DETECTED from kwargs: {len(video_ids)} videos - IDs: {video_ids}")
            if stream:
                logger.info(f"🔥 GENERATE_CHUNKS: CALLING _summarize_multiple_videos_stream (legacy) with prompt='{prompt}', video_ids={video_ids}")
                return self._summarize_multiple_videos_stream(prompt, video_ids)
            else:
                logger.info(f"🔥 GENERATE_CHUNKS: CALLING _summarize_multiple_videos_concatenated (legacy) with prompt='{prompt}', video_ids={video_ids}")
                result = self._summarize_multiple_videos_concatenated(prompt, video_ids)
                if result.get("error"):
                    logger.error(f"Multi-video concatenation failed: {result.get('text')}")
                    return {"text": result.get("text", "Multi-video summarization failed")}
                return {"text": result.get("text", "No summary generated")}
        
        logger.info(f"🔥 GENERATE_CHUNKS: NO MULTI-VIDEO DETECTED, starting Pegasus single video summarization for {len(chunks)} chunks")
        if stream:
            logger.info(f"🔥 GENERATE_CHUNKS: CALLING _summarize_video_stream with prompt='{prompt}', chunks={len(chunks)}")
            return self._summarize_video_stream(prompt, chunks)
        else:
            logger.info(f"🔥 GENERATE_CHUNKS: CALLING _summarize_video with prompt='{prompt}', chunks={len(chunks)}")
            result = self._summarize_video(prompt, chunks)
            if result.get("error"):
                logger.error(f"Single video summarization failed: {result.get('text')}")
                return {"text": result.get("text", "Video summarization failed")}
            return {"text": result.get("text", "No summary generated")}
    
    def _summarize_video(self, prompt: str, chunks: List[ChunkInfo]) -> dict:
        """Execute video summarization using Twelve Labs Pegasus model."""
        try:
            logger.info(f"Starting video summarization for {len(chunks)} chunks")
            
            # Get the video file path from the first chunk
            if not chunks or len(chunks) == 0:
                return {"text": "No video chunks provided for summarization", "error": True}
            
            # Get the VSS file ID from the chunk info
            chunk = chunks[0]
            video_file_path = chunk.file if hasattr(chunk, 'file') else None
            
            if not video_file_path:
                return {"text": "Unable to determine video file path from chunks", "error": True}
            
            # Extract VSS file ID from the path (typically in format: ./assets/{vss_id}/video.mp4)
            import re
            path_parts = str(video_file_path).split('/')
            vss_file_id = None
            for part in path_parts:
                if re.match(r'^[0-9a-f-]{36}$', part):  # UUID format
                    vss_file_id = part
                    break
            
            if not vss_file_id:
                return {"text": f"Unable to extract VSS file ID from path: {video_file_path}", "error": True}
            
            logger.info(f"Summarizing video with VSS ID: {vss_file_id}")
            
            # Ensure the video is uploaded to Twelve Labs
            upload_result = self.ensure_video_uploaded(vss_file_id)
            if upload_result.get("error"):
                return upload_result
            
            pegasus_video_id = upload_result.get("pegasus_video_id")
            if not pegasus_video_id:
                return {"text": "Failed to get Pegasus video ID for summarization", "error": True}
            
            # Use Twelve Labs Pegasus for actual video summarization
            pegasus_index_id = self._get_pegasus_index_id()
            
            logger.info(f"Using Pegasus to summarize video {pegasus_video_id}")
            response = self._client.summarize(
                video_id=pegasus_video_id,
                prompt=prompt,
                temperature=self._config.analysis_temperature,
                type="summary"
            )
            
            logger.info(f"Result ID: {response.id}")
            
            if response.summary is not None:
                summary_text = response.summary
                logger.info(f"Video summarization completed: {len(summary_text)} chars")
            else:
                logger.warning(f"No summary in response: {type(response)}")
                summary_text = f"Unable to generate summary for video {vss_file_id}"
            
            return {
                "text": summary_text,
                "vss_file_id": vss_file_id,
                "pegasus_video_id": pegasus_video_id,
                "workflow": "pegasus_summarization"
            }
            
        except Exception as e:
            logger.error(f"Video summarization error: {e}")
            return {"text": f"Summarization error: {str(e)}", "error": True}
    
    def _summarize_video_stream(self, prompt: str, chunks: List[ChunkInfo]) -> dict:
        """Execute streaming video summarization using Twelve Labs Pegasus model."""
        try:
            logger.info(f"Starting streaming video summarization for {len(chunks)} chunks")
            
            # Get the video file info same as non-streaming
            if not chunks or len(chunks) == 0:
                yield {"choices": [{"message": {"content": "No video chunks provided for summarization"}}]}
                return
            
            chunk = chunks[0]
            video_file_path = chunk.file if hasattr(chunk, 'file') else None
            
            if not video_file_path:
                yield {"choices": [{"message": {"content": "Unable to determine video file path from chunks"}}]}
                return
            
            # Extract VSS file ID from path
            import re
            path_parts = str(video_file_path).split('/')
            vss_file_id = None
            for part in path_parts:
                if re.match(r'^[0-9a-f-]{36}$', part):  # UUID format
                    vss_file_id = part
                    break
            
            if not vss_file_id:
                yield {"choices": [{"message": {"content": f"Unable to extract VSS file ID from path: {video_file_path}"}}]}
                return
            
            logger.info(f"Streaming summarization for video with VSS ID: {vss_file_id}")
            
            # Ensure video is uploaded
            upload_result = self.ensure_video_uploaded(vss_file_id)
            if upload_result.get("error"):
                yield {"choices": [{"message": {"content": upload_result["text"]}}]}
                return
            
            pegasus_video_id = upload_result.get("pegasus_video_id")
            if not pegasus_video_id:
                yield {"choices": [{"message": {"content": "Failed to get Pegasus video ID for summarization"}}]}
                return
            
            # Use Pegasus summarization (Twelve Labs may not have streaming summarize)
            logger.info(f"Streaming Pegasus summarization for video {pegasus_video_id}")
            try:
                # Try streaming summarize if available
                if hasattr(self._client, 'summarize_stream'):
                    for chunk in self._client.summarize_stream(
                        video_id=pegasus_video_id,
                        prompt=prompt,
                        temperature=self._config.analysis_temperature,
                        type="summary"
                    ):
                        if hasattr(chunk, 'summary') and chunk.summary:
                            yield {"choices": [{"delta": {"content": chunk.summary}}]}
                        elif hasattr(chunk, 'text') and chunk.text:
                            yield {"choices": [{"delta": {"content": chunk.text}}]}
                else:
                    # Fallback to non-streaming summarize
                    response = self._client.summarize(
                        video_id=pegasus_video_id,
                        prompt=prompt,
                        temperature=self._config.analysis_temperature,
                        type="summary"
                    )
                    
                    if response.summary is not None:
                        yield {"choices": [{"message": {"content": response.summary}}]}
                    else:
                        yield {"choices": [{"message": {"content": f"Unable to generate summary for video {vss_file_id}"}}]}
                
                # Send completion marker
                yield {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]}
                
            except Exception as stream_error:
                logger.warning(f"Streaming failed, falling back to regular summarize: {stream_error}")
                # Fallback to non-streaming
                response = self._client.summarize(
                    video_id=pegasus_video_id,
                    prompt=prompt,
                    temperature=self._config.analysis_temperature,
                    type="summary"
                )
                
                if response.summary is not None:
                    yield {"choices": [{"message": {"content": response.summary}}]}
                else:
                    yield {"choices": [{"message": {"content": f"Unable to generate summary for video {vss_file_id}"}}]}
                    
        except Exception as e:
            logger.error(f"Streaming video summarization error: {e}")
            yield {"choices": [{"message": {"content": f"Summarization error: {str(e)}"}}]}
    
    def _check_all_videos_ready(self, video_ids: List[str]) -> Dict:
        """Pre-check to ensure all videos are uploaded and ready before summarization."""
        logger.info(f"Pre-checking readiness of {len(video_ids)} videos: {video_ids}")
        
        failed_videos = []
        ready_videos = []
        
        for i, vss_file_id in enumerate(video_ids):
            logger.info(f"Checking video {i+1}/{len(video_ids)}: {vss_file_id}")
            
            # Check if video is uploaded and ready
            upload_result = self.ensure_video_uploaded(vss_file_id)
            if upload_result.get("error"):
                failed_videos.append({
                    "vss_file_id": vss_file_id,
                    "position": i + 1,
                    "error": upload_result.get("text")
                })
                logger.error(f"Video {i+1} ({vss_file_id}) is not ready: {upload_result.get('text')}")
            else:
                ready_videos.append(vss_file_id)
                logger.info(f"Video {i+1} ({vss_file_id}) is ready")
        
        if failed_videos:
            error_details = []
            for failed in failed_videos:
                error_details.append(f"Video {failed['position']} ({failed['vss_file_id']}): {failed['error']}")
            
            error_message = f"Cannot proceed with multi-video summarization. {len(failed_videos)}/{len(video_ids)} videos are not ready:\n" + "\n".join(error_details)
            logger.error(error_message)
            return {"error": True, "text": error_message, "failed_videos": failed_videos}
        
        logger.info(f"All {len(video_ids)} videos are ready for summarization")
        return {"error": False, "ready_videos": ready_videos}

    def _summarize_multiple_videos_concatenated(self, prompt: str, video_ids: List[str]) -> dict:
        """Execute video summarization for multiple videos and concatenate results."""
        try:
            logger.info(f"Starting concatenated multi-video summarization for {len(video_ids)} videos")
            
            # Pre-check all videos are ready
            readiness_check = self._check_all_videos_ready(video_ids)
            if readiness_check.get("error"):
                return readiness_check
            
            individual_summaries = []
            successful_count = 0
            failed_count = 0
            
            for i, vss_file_id in enumerate(video_ids):
                logger.info(f"Processing video {i+1}/{len(video_ids)}: {vss_file_id}")
                
                try:
                    # Ensure the video is uploaded to Twelve Labs
                    upload_result = self.ensure_video_uploaded(vss_file_id)
                    if upload_result.get("error"):
                        logger.warning(f"Failed to upload video {vss_file_id}: {upload_result.get('text')}")
                        individual_summaries.append({
                            "video_number": i + 1,
                            "vss_file_id": vss_file_id,
                            "status": "error",
                            "content": f"❌ Upload failed: {upload_result.get('text', 'Unknown error')}"
                        })
                        failed_count += 1
                        continue
                    
                    pegasus_video_id = upload_result.get("pegasus_video_id")
                    if not pegasus_video_id:
                        error_msg = f"Failed to get Pegasus video ID for {vss_file_id}"
                        logger.warning(error_msg)
                        individual_summaries.append({
                            "video_number": i + 1,
                            "vss_file_id": vss_file_id,
                            "status": "error",
                            "content": f"❌ {error_msg}"
                        })
                        failed_count += 1
                        continue
                    
                    # Perform summarization for this video
                    logger.info(f"Summarizing video {pegasus_video_id} (VSS ID: {vss_file_id})")
                    response = self._client.summarize(
                        video_id=pegasus_video_id,
                        prompt=prompt,
                        temperature=self._config.analysis_temperature,
                        type="summary"
                    )
                    
                    if response.summary is not None:
                        individual_summaries.append({
                            "video_number": i + 1,
                            "vss_file_id": vss_file_id,
                            "status": "success",
                            "content": response.summary
                        })
                        successful_count += 1
                        logger.info(f"Successfully summarized video {i+1}/{len(video_ids)}")
                    else:
                        error_msg = f"No summary returned for video {vss_file_id}"
                        logger.warning(error_msg)
                        individual_summaries.append({
                            "video_number": i + 1,
                            "vss_file_id": vss_file_id,
                            "status": "error",
                            "content": f"❌ {error_msg}"
                        })
                        failed_count += 1
                
                except Exception as video_error:
                    logger.error(f"Error summarizing video {vss_file_id}: {video_error}")
                    individual_summaries.append({
                        "video_number": i + 1,
                        "vss_file_id": vss_file_id,
                        "status": "error",
                        "content": f"❌ Processing error: {str(video_error)}"
                    })
                    failed_count += 1
            
            # Format the concatenated results
            formatted_result = self._format_concatenated_summaries(prompt, individual_summaries, successful_count, failed_count)
            
            return {
                "text": formatted_result,
                "successful_count": successful_count,
                "failed_count": failed_count,
                "total_count": len(video_ids),
                "workflow": "concatenated_multi_video_summarization"
            }
            
        except Exception as e:
            logger.error(f"Multi-video summarization error: {e}")
            return {"text": f"Multi-video summarization error: {str(e)}", "error": True}
    
    def _summarize_multiple_videos_stream(self, prompt: str, video_ids: List[str]):
        """Execute streaming video summarization for multiple videos with concatenation."""
        try:
            logger.info(f"🔥 STARTING MULTI-VIDEO STREAMING: {len(video_ids)} videos with prompt: '{prompt}'")
            logger.info(f"🔥 MULTI-VIDEO VIDEO IDS: {video_ids}")
            
            # Pre-check all videos are ready
            logger.info(f"🔥 MULTI-VIDEO: Running readiness pre-check...")
            readiness_check = self._check_all_videos_ready(video_ids)
            logger.info(f"🔥 MULTI-VIDEO: Readiness check result: {readiness_check}")
            
            if readiness_check.get("error"):
                error_msg = f"❌ Multi-video readiness check failed:\n{readiness_check.get('text')}\n"
                logger.error(f"🔥 MULTI-VIDEO: Readiness check failed, yielding error: {error_msg}")
                yield {"choices": [{"delta": {"content": error_msg}}]}
                return
            
            logger.info(f"🔥 MULTI-VIDEO: All videos ready, starting streaming...")
            
            # Start with overview
            overview_msg = f"Multi-Video Summary\n"
            logger.info(f"🔥 MULTI-VIDEO: Yielding overview: {overview_msg}")
            yield {"choices": [{"delta": {"content": overview_msg}}]}
            
            processing_msg = f"Processing {len(video_ids)} videos...\n\n"
            logger.info(f"🔥 MULTI-VIDEO: Yielding processing message: {processing_msg}")
            yield {"choices": [{"delta": {"content": processing_msg}}]}
            
            successful_count = 0
            failed_count = 0
            
            for i, vss_file_id in enumerate(video_ids):
                logger.info(f"🔥 MULTI-VIDEO: Processing video {i+1}/{len(video_ids)}: {vss_file_id}")
                
                video_header = f"Video {i+1}:\n"
                logger.info(f"🔥 MULTI-VIDEO: Yielding video header: {video_header}")
                yield {"choices": [{"delta": {"content": video_header}}]}
                
                try:
                    # Ensure the video is uploaded to Twelve Labs
                    logger.info(f"🔥 MULTI-VIDEO: Checking upload status for {vss_file_id}...")
                    upload_result = self.ensure_video_uploaded(vss_file_id)
                    logger.info(f"🔥 MULTI-VIDEO: Upload result for {vss_file_id}: {upload_result}")
                    
                    if upload_result.get("error"):
                        error_msg = f"❌ Upload failed: {upload_result.get('text', 'Unknown error')}\n\n"
                        logger.error(f"🔥 MULTI-VIDEO: Upload failed for {vss_file_id}, yielding error: {error_msg}")
                        yield {"choices": [{"delta": {"content": error_msg}}]}
                        failed_count += 1
                        continue
                    
                    pegasus_video_id = upload_result.get("pegasus_video_id")
                    logger.info(f"🔥 MULTI-VIDEO: Got Pegasus video ID for {vss_file_id}: {pegasus_video_id}")
                    
                    if not pegasus_video_id:
                        error_msg = f"❌ Failed to get Pegasus video ID\n\n"
                        logger.error(f"🔥 MULTI-VIDEO: No Pegasus video ID for {vss_file_id}, yielding error: {error_msg}")
                        yield {"choices": [{"delta": {"content": error_msg}}]}
                        failed_count += 1
                        continue
                    
                    # Perform summarization for this video
                    logger.info(f"🔥 MULTI-VIDEO: Starting Twelve Labs API call for video {pegasus_video_id}...")
                    logger.info(f"🔥 MULTI-VIDEO: API params - video_id: {pegasus_video_id}, prompt: '{prompt}', temp: {self._config.analysis_temperature}")
                    
                    response = self._client.summarize(
                        video_id=pegasus_video_id,
                        prompt=prompt,
                        temperature=self._config.analysis_temperature,
                        type="summary"
                    )
                    
                    logger.info(f"🔥 MULTI-VIDEO: API response for {pegasus_video_id}: type={type(response)}, summary_length={len(response.summary) if response.summary else 0}")
                    
                    if response.summary is not None:
                        logger.info(f"🔥 MULTI-VIDEO: Got valid summary for {pegasus_video_id}, length: {len(response.summary)}")
                        logger.info(f"🔥 MULTI-VIDEO: Yielding summary content...")
                        yield {"choices": [{"delta": {"content": response.summary}}]}
                        yield {"choices": [{"delta": {"content": "\n\n"}}]}
                        successful_count += 1
                        logger.info(f"🔥 MULTI-VIDEO: Successfully processed video {i+1}, total successful: {successful_count}")
                    else:
                        logger.error(f"🔥 MULTI-VIDEO: No summary in response for {pegasus_video_id}")
                        yield {"choices": [{"delta": {"content": f"❌ No summary generated\n\n"}}]}
                        failed_count += 1
                
                except Exception as video_error:
                    logger.error(f"🔥 MULTI-VIDEO: Exception processing video {vss_file_id}: {video_error}")
                    error_msg = f"❌ Processing error: {str(video_error)}\n\n"
                    logger.error(f"🔥 MULTI-VIDEO: Yielding exception error: {error_msg}")
                    yield {"choices": [{"delta": {"content": error_msg}}]}
                    failed_count += 1
            
            logger.info(f"🔥 MULTI-VIDEO: Finished processing all videos - successful: {successful_count}, failed: {failed_count}")
            
            # Summary footer
            summary_footer = f"---\nProcessed {successful_count + failed_count} videos"
            if successful_count > 0:
                summary_footer += f" ({successful_count} successful"
                if failed_count > 0:
                    summary_footer += f", {failed_count} failed"
                summary_footer += ")"
            elif failed_count > 0:
                summary_footer += f" (all {failed_count} failed)"
            summary_footer += "\n"
            
            logger.info(f"🔥 MULTI-VIDEO: Yielding summary footer: {summary_footer}")
            yield {"choices": [{"delta": {"content": summary_footer}}]}
            
            logger.info(f"🔥 MULTI-VIDEO: Yielding final stop token")
            yield {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]}
            
            logger.info(f"🔥 MULTI-VIDEO: COMPLETED STREAMING - successful: {successful_count}, failed: {failed_count}")
            
        except Exception as e:
            logger.error(f"Streaming multi-video summarization error: {e}")
            yield {"choices": [{"message": {"content": f"Multi-video summarization error: {str(e)}"}}]}
    
    def _format_concatenated_summaries(self, prompt: str, summaries: List[dict], successful_count: int, failed_count: int) -> str:
        """Format individual video summaries into a concatenated response."""
        result_lines = []
        result_lines.append("Multi-Video Summary")
        result_lines.append("=" * 50)
        result_lines.append(f"Query: {prompt}")
        result_lines.append(f"Processed: {successful_count + failed_count} videos ({successful_count} successful, {failed_count} failed)")
        result_lines.append("")
        
        # Add individual video summaries
        for summary_data in summaries:
            video_num = summary_data["video_number"]
            content = summary_data["content"]
            
            result_lines.append(f"Video {video_num}:")
            result_lines.append("-" * 20)
            result_lines.append(content)
            result_lines.append("")
        
        if successful_count > 1:
            result_lines.append("Combined Analysis:")
            result_lines.append("-" * 20)
            result_lines.append(f"Successfully processed {successful_count} videos. Each video was individually summarized based on the query: {prompt}")
            result_lines.append("")
        
        result_lines.append("=" * 50)
        
        return "\n".join(result_lines)
    
    def search(self, query: str, analyze: bool = False, stream: bool = False, max_clips: int = None, threshold: str = None):
        """Execute search across all videos using Twelve Labs Marengo model."""
        if self._client is None:
            return {"text": "Client not available"}
        
        logger.info(f"Starting Twelve Labs search: query='{query}', analyze={analyze}, stream={stream}")
        
        # Set search parameters
        if max_clips is not None:
            self._config.max_clips = max_clips
        if threshold is not None:
            self._config.search_threshold = threshold
        
        if stream:
            return self._search_stream(query)
        else:
            result = self._search_only(query)
            return {"text": result["text"]}
    
    def _search_stream(self, query: str):
        """Execute streaming search across all videos."""
        # Get the search results
        result = self._search_only(query)
        
        # Convert to streaming format and yield
        text = result.get("text", "")
        if text:
            yield {
                "choices": [{
                    "delta": {"content": text},
                    "index": 0
                }]
            }
        
        # Send completion marker
        yield {
            "choices": [{
                "delta": {"content": ""}, 
                "finish_reason": "stop"
            }]
        }
    
    def _search_only(self, prompt: str) -> dict:
        """Execute search-only workflow: return clips without analysis."""
        try:
            logger.info(f"Starting search-only workflow")
            
            marengo_index_id = self._get_marengo_index_id()
            
            logger.info("Searching across all previously uploaded videos")
            relevant_clips = self._marengo_search_all(prompt, marengo_index_id)
            if not relevant_clips or len(relevant_clips) == 0:
                logger.warning("No relevant clips found")
                return {
                    "text": f"No relevant clips found for query: {prompt}",
                    "workflow": "marengo_search_no_results"
                }
            
            # Format search results without analysis
            search_results = f"Found {len(relevant_clips)} relevant clips for query: {prompt}\n\n"
            
            for i, clip in enumerate(relevant_clips[:10]):  # Show top 10 clips
                video_title = clip.get('video_title', f"Video {clip['video_id'][:8]}")
                vss_file_id = clip.get('vss_file_id', 'Unknown')
                twelve_labs_video_id = clip.get('video_id', 'Unknown')
                search_results += f"{i+1}. {video_title}\n"
                search_results += f"   VSS Video ID: {vss_file_id}\n"
                search_results += f"   Twelve Labs Video ID: {twelve_labs_video_id}\n"
                search_results += f"   Time: {clip['start']:.1f}s - {clip['end']:.1f}s\n"
                search_results += f"   Score: {clip['score']:.2f}\n"
                search_results += f"   Confidence: {clip['confidence']}\n\n"
            
            return {
                "text": search_results,
                "clips_found": len(relevant_clips),
                "workflow": "marengo_search_only"
            }
            
        except Exception as e:
            logger.error(f"Twelve Labs search error: {e}")
            return {"text": f"Search error: {str(e)}", "error": True}
    
    
    
    def _get_marengo_index_id(self):
        def create_index():
            return ensure_index_exists(
                self._client,
                self._config.marengo_index_name,
                self._config.marengo_model,
                self._config.marengo_options
            )
        
        marengo_index_id = retry_with_exponential_backoff(
            create_index,
            self._config.max_retries,
            self._config.retry_delay_base
        )
        logger.info(f"Marengo index ready: {marengo_index_id}")
        return marengo_index_id
    
    def _get_pegasus_index_id(self):
        def create_index():
            return ensure_index_exists(
                self._client,
                self._config.pegasus_index_name,
                self._config.pegasus_model,
                self._config.pegasus_options
            )
        
        pegasus_index_id = retry_with_exponential_backoff(
            create_index,
            self._config.max_retries,
            self._config.retry_delay_base
        )
        logger.info(f"Pegasus index ready: {pegasus_index_id}")
        return pegasus_index_id
    
    def ensure_video_uploaded(self, vss_file_id: str) -> Dict:
        logger.info(f"Ensuring video uploaded for VSS ID: {vss_file_id}")
        
        try:
            marengo_index_id = self._get_marengo_index_id()
            pegasus_index_id = self._get_pegasus_index_id()
        except Exception as e:
            logger.error(f"Failed to get indexes for VSS ID {vss_file_id}: {e}")
            return {"text": f"Failed to initialize indexes: {str(e)}", "error": True}
        
        mapping = VideoIDMapper.get_mapping(vss_file_id)
        marengo_video_id = mapping.get("marengo_video_id") if mapping else None
        pegasus_video_id = mapping.get("pegasus_video_id") if mapping else None
        
        logger.info(f"VSS ID {vss_file_id}: existing mapping - Marengo: {marengo_video_id}, Pegasus: {pegasus_video_id}")
        
        video_path = VideoIDMapper.get_video_path(vss_file_id)
        if not video_path:
            logger.error(f"Video file not found for VSS ID: {vss_file_id}")
            return {"text": f"Video file not found for ID: {vss_file_id}", "error": True}
        
        logger.info(f"VSS ID {vss_file_id}: video path found at {video_path}")
        
        if not marengo_video_id:
            logger.info(f"VSS ID {vss_file_id}: uploading to Marengo index {marengo_index_id}")
            marengo_result = self._upload_video_to_index(video_path, marengo_index_id, "marengo")
            if marengo_result.get("error"):
                logger.error(f"VSS ID {vss_file_id}: Marengo upload failed - {marengo_result.get('text')}")
                return marengo_result
            marengo_video_id = marengo_result.get("video_id")
            VideoIDMapper.save_mapping(vss_file_id, marengo_video_id=marengo_video_id, marengo_index_id=marengo_index_id)
            logger.info(f"VSS ID {vss_file_id}: Marengo upload successful, video ID: {marengo_video_id}")
        else:
            logger.info(f"VSS ID {vss_file_id}: Marengo video already exists with ID: {marengo_video_id}")
        
        if not pegasus_video_id:
            logger.info(f"VSS ID {vss_file_id}: uploading to Pegasus index {pegasus_index_id}")
            pegasus_result = self._upload_video_to_index(video_path, pegasus_index_id, "pegasus")
            if pegasus_result.get("error"):
                logger.error(f"VSS ID {vss_file_id}: Pegasus upload failed - {pegasus_result.get('text')}")
                return pegasus_result
            pegasus_video_id = pegasus_result.get("video_id")
            VideoIDMapper.save_mapping(vss_file_id, pegasus_video_id=pegasus_video_id, pegasus_index_id=pegasus_index_id)
            logger.info(f"VSS ID {vss_file_id}: Pegasus upload successful, video ID: {pegasus_video_id}")
        else:
            logger.info(f"VSS ID {vss_file_id}: Pegasus video already exists with ID: {pegasus_video_id}")
        
        logger.info(f"VSS ID {vss_file_id}: upload check completed successfully - Marengo: {marengo_video_id}, Pegasus: {pegasus_video_id}")
        
        return {
            "marengo_video_id": marengo_video_id,
            "pegasus_video_id": pegasus_video_id
        }
    
    def _upload_video_to_index(self, video_path, index_id: str, model_name: str) -> Dict:
        try:
            logger.info(f"Uploading to {model_name}: video_path={video_path}, index_id={index_id}")
            logger.info(f"File exists: {video_path.exists()}, File size: {video_path.stat().st_size if video_path.exists() else 'N/A'}")
            logger.info(f"Client initialized: {self._client is not None}")
            
            if not video_path.exists():
                return {"text": f"Video file not found: {video_path}", "error": True}
            
            if not index_id:
                return {"text": f"No index ID provided for {model_name}", "error": True}
            
            if not self._client:
                return {"text": f"Twelve Labs client not initialized", "error": True}
            
            logger.info(f"Creating upload task for {model_name} with index {index_id}")
            with open(video_path, 'rb') as f:
                logger.info(f"Opened video file {video_path} for reading")
                task = self._client.tasks.create(
                    index_id=index_id,
                    video_file=f,
                )
            
            logger.info(f"Waiting for {model_name} upload task {task.id} to complete...")
            
            # Use TwelveLabs' built-in method that waits for full indexing completion
            def on_task_update(task_obj):
                logger.info(f"  {model_name} task status: {task_obj.status}")
            
            try:
                completed_task = self._client.tasks.wait_for_done(task_id=task.id, callback=on_task_update)
                logger.info(f"{model_name} indexing completed with status: {completed_task.status}")
                
                if completed_task.status != "ready":
                    return {"text": f"{model_name} indexing failed with status: {completed_task.status}", "error": True}
                
                video_id = completed_task.video_id
            except Exception as e:
                logger.error(f"{model_name} wait_for_done failed: {e}")
                return {"text": f"{model_name} indexing wait failed: {str(e)}", "error": True}
            logger.info(f"{model_name} upload successful, video ID: {video_id}")
            return {"video_id": video_id}
            
        except Exception as e:
            logger.error(f"Error uploading to {model_name}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return {"text": f"Video upload failed: {str(e)}", "error": True}
    
    def _marengo_search_all(self, query: str, marengo_index_id: str) -> List[Dict]:
        try:
            logger.info(f"Marengo search across all videos: '{query}'")
            
            def perform_search():
                return self._client.search.query(
                    index_id=marengo_index_id,
                    search_options=self._config.search_options,
                    query_text=query,
                    group_by="clip",  # Get individual clips
                    threshold="none",
                    page_limit=self._config.max_clips,
                    sort_option="score"
                )
            
            search_results = retry_with_exponential_backoff(
                perform_search,
                self._config.max_retries,
                self._config.retry_delay_base
            )
            
            clips = []

            for clip in search_results:
                clips.append({
                    "start": clip.start,
                    "end": clip.end,
                    "score": clip.score,
                    "confidence": clip.confidence,
                    "video_id": clip.video_id,
                    "video_thumbnail": clip.thumbnail_url,
                })
                        
            
            logger.info(f"Marengo found {len(clips)} relevant clips across all videos")
            return clips
            
        except Exception as e:
            logger.error(f"Marengo search error: {e}")
            return []
    
    def _pegasus_analyze_clips(self, prompt: str, clips: List[Dict], pegasus_index_id: str) -> str:
        try:
            logger.info(f"Pegasus analysis: {len(clips)} clips")
            
            # Use multiple clips for richer analysis
            top_clips = clips[:min(len(clips), self._config.max_clips_for_analysis)]
            
            # Build context from multiple clips
            clips_context = []
            pegasus_video_ids = set()
            
            for clip in top_clips:
                video_title = clip.get('video_title', f"Video {clip['video_id'][:8]}")
                pegasus_video_id = self._get_pegasus_video_id_for_marengo(clip['video_id'])
                if pegasus_video_id:
                    pegasus_video_ids.add(pegasus_video_id)
                    clips_context.append({
                        'video_id': pegasus_video_id,
                        'title': video_title,
                        'start': clip['start'],
                        'end': clip['end'],
                        'score': clip['score']
                    })
                else:
                    logger.warning(f"Could not find Pegasus video ID for Marengo video ID {clip['video_id']}")
            
            if not clips_context:
                error_msg = f"Could not generate analysis - no video mappings found for {len(top_clips)} clips"
                logger.error(error_msg)
                return error_msg
            
            # Use the best clip's video for analysis, but include context from others
            primary_video_id = clips_context[0]['video_id']
            
            analysis_prompt = f"""Question: {prompt}

Relevant video segments found:
"""
            for i, clip in enumerate(clips_context[:3]):  # Show top 3 clips
                analysis_prompt += f"{i+1}. {clip['title']} ({clip['start']:.1f}s - {clip['end']:.1f}s) - Score: {clip['score']:.2f}\n"
            
            analysis_prompt += f"\nAnalyze the content and provide a comprehensive answer to the question."
            
            # Use analyze instead of summarize for custom prompts
            response = self._client.analyze(
                video_id=primary_video_id,
                prompt=analysis_prompt,
                temperature=self._config.analysis_temperature
            )
            
            if hasattr(response, 'text') and response.text:
                analysis_text = response.text
                logger.info(f"Analysis generated: {len(analysis_text)} chars")
            else:
                logger.warning(f"No analysis in response: {type(response)}")
                video_count = len(set(clip['video_id'] for clip in clips))
                analysis_text = f"Found {len(clips)} relevant clips from {video_count} videos for query: {prompt}"
            
            logger.info("Pegasus analysis completed")
            return analysis_text
            
        except Exception as e:
            logger.error(f"Pegasus analysis error: {e}")
            video_count = len(set(clip['video_id'] for clip in clips))
            return f"Found {len(clips)} relevant clips from {video_count} videos for query: {prompt}"
    
    def _pegasus_analyze_clips_stream(self, prompt: str, clips: List[Dict], pegasus_index_id: str) -> Generator[dict, None, None]:
        """Stream analysis results from Pegasus."""
        try:
            logger.info(f"Pegasus streaming analysis: {len(clips)} clips")
            
            # Use multiple clips for richer analysis
            top_clips = clips[:min(len(clips), self._config.max_clips_for_analysis)]
            
            # Build context from multiple clips
            clips_context = []
            for clip in top_clips:
                video_title = clip.get('video_title', f"Video {clip['video_id'][:8]}")
                pegasus_video_id = self._get_pegasus_video_id_for_marengo(clip['video_id'])
                if pegasus_video_id:
                    clips_context.append({
                        'video_id': pegasus_video_id,
                        'title': video_title,
                        'start': clip['start'],
                        'end': clip['end'],
                        'score': clip['score']
                    })
                else:
                    logger.warning(f"Could not find Pegasus video ID for Marengo video ID {clip['video_id']}")
            
            if not clips_context:
                error_msg = f"Could not generate analysis - no video mappings found for {len(top_clips)} clips"
                logger.error(error_msg)
                yield {"choices": [{"message": {"content": error_msg}}]}
                return
            
            # Use the best clip's video for analysis
            primary_video_id = clips_context[0]['video_id']
            
            analysis_prompt = f"""Question: {prompt}

Relevant video segments found:
"""
            for i, clip in enumerate(clips_context[:3]):  # Show top 3 clips
                analysis_prompt += f"{i+1}. {clip['title']} ({clip['start']:.1f}s - {clip['end']:.1f}s) - Score: {clip['score']:.2f}\n"
            
            analysis_prompt += f"\nAnalyze the content and provide a comprehensive answer to the question."
            
            # Use streaming analyze
            try:
                for chunk in self._client.analyze_stream(
                    video_id=primary_video_id,
                    prompt=analysis_prompt,
                    temperature=self._config.analysis_temperature
                ):
                    if hasattr(chunk, 'text') and chunk.text:
                        yield {"choices": [{"delta": {"content": chunk.text}}]}
                    elif hasattr(chunk, 'content') and chunk.content:
                        yield {"choices": [{"delta": {"content": chunk.content}}]}
                        
                # Send completion marker
                yield {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]}
                
            except Exception as stream_error:
                logger.warning(f"Streaming failed, falling back to regular analysis: {stream_error}")
                # Fallback to non-streaming
                response = self._client.analyze(
                    video_id=primary_video_id,
                    prompt=analysis_prompt,
                    temperature=self._config.analysis_temperature
                )
                
                if hasattr(response, 'text') and response.text:
                    yield {"choices": [{"message": {"content": response.text}}]}
                else:
                    video_count = len(set(clip['video_id'] for clip in clips))
                    yield {"choices": [{"message": {"content": f"Found {len(clips)} relevant clips from {video_count} videos for query: {prompt}"}}]}
            
        except Exception as e:
            logger.error(f"Pegasus streaming analysis error: {e}")
            video_count = len(set(clip['video_id'] for clip in clips))
            yield {"choices": [{"message": {"content": f"Found {len(clips)} relevant clips from {video_count} videos for query: {prompt}"}}]}
    
    def _get_pegasus_video_id_for_marengo(self, marengo_video_id: str) -> Optional[str]:
        """Get the corresponding Pegasus video ID for a Marengo video ID."""
        # Use optimized lookup from VideoIDMapper
        return VideoIDMapper.get_pegasus_for_marengo(marengo_video_id)
    
    def get_embedding_generator(self):
        return None
    
    @staticmethod
    def get_model_info():
        return "twelve-labs", "cloud", "twelve-labs"
    
    def warmup(self):
        pass
