# video_manager.py
import yt_dlp
import streamlit as st
import os
import cv2
import numpy as np
from pathlib import Path
import logging
import asyncio
import aiohttp


from moviepy import VideoFileClip, TextClip, CompositeVideoClip
from PIL import Image
import torch
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Tuple
import time
import json

class VideoProcessor:
    def __init__(self):
        self.download_path = Path("downloads")
        self.processed_path = Path("processed")
        self.temp_path = Path("temp")
        
        # Create necessary directories
        for path in [self.download_path, self.processed_path, self.temp_path]:
            path.mkdir(exist_ok=True)
        
        # Initialize AI models
        self.setup_ai_models()

    def setup_ai_models(self):
        """Initialize AI models for video processing."""
        try:
            # Caption generation model
            self.caption_generator = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
            
            # NSFW content detection
            self.nsfw_detector = pipeline("image-classification", model="microsoft/resnet-50")
            
            # Hashtag generation model
            self.hashtag_generator = pipeline("text2text-generation", model="facebook/bart-large-cnn")
            
        except Exception as e:
            logging.error(f"Error loading AI models: {str(e)}")
            st.error("Error initializing AI features. Some functions may be limited.")

class VideoDownloader:
    def __init__(self, processor: VideoProcessor):
        self.processor = processor
        self.supported_platforms = {
            'youtube': self.download_youtube,
            'tiktok': self.download_tiktok,
            'instagram': self.download_instagram
        }

    async def download_video(self, url: str, platform: str) -> Tuple[bool, str, Optional[Path]]:
        """Download video from supported platforms."""
        try:
            if platform not in self.supported_platforms:
                return False, "Unsupported platform", None

            download_func = self.supported_platforms[platform]
            success, message, file_path = await download_func(url)

            if success and file_path:
                # Verify video integrity
                if not self.verify_video(file_path):
                    return False, "Downloaded video is corrupted", None

                # Check for NSFW content
                if await self.check_nsfw_content(file_path):
                    os.remove(file_path)
                    return False, "NSFW content detected", None

                return True, "Download successful", file_path
            
            return False, message, None

        except Exception as e:
            logging.error(f"Download error: {str(e)}")
            return False, f"Download failed: {str(e)}", None

    async def download_youtube(self, url: str) -> Tuple[bool, str, Optional[Path]]:
        """Download video from YouTube."""
        try:
            ydl_opts = {
                'format': 'best',
                'outtmpl': str(self.processor.download_path / '%(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                file_path = Path(ydl.prepare_filename(info))
                return True, "YouTube download successful", file_path

        except Exception as e:
            logging.error(f"YouTube download error: {str(e)}")
            return False, f"YouTube download failed: {str(e)}", None

    async def download_tiktok(self, url: str) -> Tuple[bool, str, Optional[Path]]:
        """Download video from TikTok."""
        try:
            async with aiohttp.ClientSession() as session:
                # Use TikTok API endpoint (you'll need to implement proper API authentication)
                # This is a placeholder implementation
                async with session.get(url) as response:
                    if response.status == 200:
                        file_path = self.processor.download_path / f"tiktok_{int(time.time())}.mp4"
                        with open(file_path, 'wb') as f:
                            while True:
                                chunk = await response.content.read(8192)
                                if not chunk:
                                    break
                                f.write(chunk)
                        return True, "TikTok download successful", file_path
                    return False, f"TikTok download failed: Status {response.status}", None

        except Exception as e:
            logging.error(f"TikTok download error: {str(e)}")
            return False, f"TikTok download failed: {str(e)}", None

    async def download_instagram(self, url: str) -> Tuple[bool, str, Optional[Path]]:
        """Download video from Instagram."""
        try:
            # Similar to TikTok implementation
            # You'll need to implement proper Instagram API authentication
            pass

        except Exception as e:
            logging.error(f"Instagram download error: {str(e)}")
            return False, f"Instagram download failed: {str(e)}", None

    def verify_video(self, file_path: Path) -> bool:
        """Verify if the downloaded video is valid and playable."""
        try:
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                return False
            ret, frame = cap.read()
            cap.release()
            return ret is True
        except Exception as e:
            logging.error(f"Video verification error: {str(e)}")
            return False

    async def check_nsfw_content(self, file_path: Path) -> bool:
        """Check if video contains NSFW content."""
        try:
            video = VideoFileClip(str(file_path))
            frames = []
            
            # Sample frames from the video
            duration = video.duration
            sample_times = np.linspace(0, duration, num=5)
            
            for t in sample_times:
                frame = video.get_frame(t)
                frame_pil = Image.fromarray(frame)
                frames.append(frame_pil)
            
            video.close()

            # Check frames for NSFW content
            for frame in frames:
                result = self.processor.nsfw_detector(frame)
                if any(pred['label'] == 'NSFW' and pred['score'] > 0.7 for pred in result):
                    return True

            return False

        except Exception as e:
            logging.error(f"NSFW check error: {str(e)}")
            return False

class VideoEnhancer:
    def __init__(self, processor: VideoProcessor):
        self.processor = processor

    async def enhance_video(self, file_path: Path, options: Dict) -> Tuple[bool, str, Optional[Path]]:
        """Enhance video with various effects and modifications."""
        try:
            video = VideoFileClip(str(file_path))
            enhanced = video

            if options.get('add_watermark'):
                enhanced = self.add_watermark(enhanced, options['watermark_text'])

            if options.get('add_captions'):
                enhanced = await self.add_captions(enhanced)

            if options.get('resize'):
                enhanced = self.resize_video(enhanced, options['target_size'])

            # Save enhanced video
            output_path = self.processor.processed_path / f"enhanced_{file_path.name}"
            enhanced.write_videofile(str(output_path))
            
            video.close()
            enhanced.close()

            return True, "Video enhancement successful", output_path

        except Exception as e:
            logging.error(f"Video enhancement error: {str(e)}")
            return False, f"Video enhancement failed: {str(e)}", None

    def add_watermark(self, video: VideoFileClip, text: str) -> CompositeVideoClip:
        """Add watermark to video."""
        watermark = TextClip(
            text,
            fontsize=30,
            color='white',
            bg_color='rgba(0,0,0,0.5)',
            font='Arial'
        ).set_position(('right', 'bottom')).set_duration(video.duration)
        
        return CompositeVideoClip([video, watermark])

    async def add_captions(self, video: VideoFileClip) -> CompositeVideoClip:
        """Add AI-generated captions to video."""
        try:
            # Extract frame for caption generation
            frame = video.get_frame(video.duration / 2)
            frame_pil = Image.fromarray(frame)
            
            # Generate caption
            caption = self.processor.caption_generator(frame_pil)[0]['generated_text']
            
            # Create caption clip
            caption_clip = TextClip(
                caption,
                fontsize=24,
                color='white',
                bg_color='rgba(0,0,0,0.5)',
                font='Arial'
            ).set_position('bottom').set_duration(video.duration)
            
            return CompositeVideoClip([video, caption_clip])

        except Exception as e:
            logging.error(f"Caption generation error: {str(e)}")
            return video

    def resize_video(self, video: VideoFileClip, target_size: Tuple[int, int]) -> VideoFileClip:
        """Resize video to target dimensions."""
        return video.resize(target_size)

def render_video_downloader_page():
    """Render the video downloader interface in Streamlit."""
    st.title("Video Downloader")
    
    processor = VideoProcessor()
    downloader = VideoDownloader(processor)
    enhancer = VideoEnhancer(processor)

    with st.form("video_download_form"):
        url = st.text_input("Video URL")
        platform = st.selectbox("Platform", ["youtube", "tiktok", "instagram"])
        
        # Enhancement options
        st.subheader("Enhancement Options")
        add_watermark = st.checkbox("Add Watermark")
        watermark_text = st.text_input("Watermark Text") if add_watermark else ""
        
        add_captions = st.checkbox("Add AI Captions")
        resize = st.checkbox("Resize Video")
        if resize:
            width = st.number_input("Width", min_value=100, max_value=1920, value=720)
            height = st.number_input("Height", min_value=100, max_value=1080, value=1280)

        submit = st.form_submit_button("Download and Process")

        if submit:
            with st.spinner("Downloading and processing video..."):
                # Create enhancement options dictionary
                options = {
                    'add_watermark': add_watermark,
                    'watermark_text': watermark_text,
                    'add_captions': add_captions,
                    'resize': resize,
                    'target_size': (width, height) if resize else None
                }

                # Download and enhance video
                asyncio.run(process_video(url, platform, options, downloader, enhancer))

async def process_video(url: str, platform: str, options: Dict, 
                       downloader: VideoDownloader, enhancer: VideoEnhancer):
    """Process video download and enhancement."""
    success, message, file_path = await downloader.download_video(url, platform)
    
    if success and file_path:
        st.success("Download successful!")
        
        # Enhance video if options are selected
        if any(options.values()):
            success, message, enhanced_path = await enhancer.enhance_video(file_path, options)
            if success:
                st.success("Video enhancement successful!")
                st.video(str(enhanced_path))
            else:
                st.error(f"Enhancement failed: {message}")
        else:
            st.video(str(file_path))
    else:
        st.error(f"Download failed: {message}")
