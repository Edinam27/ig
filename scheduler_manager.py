# scheduler_manager.py
import streamlit as st
import pandas as pd
import asyncio
import sqlite3
import aiohttp
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
from pathlib import Path
import threading
from queue import Queue
import pytz
from instagram_private_api import Client, ClientCompatPatch
from instabot import Bot
from twitter import Twitter, OAuth
from PIL import Image
import io
import hashlib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class PostContent:
    """Data class for post content."""
    content_type: str  # 'image', 'video', 'reel', 'story'
    file_path: Path
    caption: str
    hashtags: List[str]
    platform: str
    schedule_time: datetime
    user_id: int
    status: str = 'pending'
    post_id: Optional[str] = None

class SocialMediaManager:
    def __init__(self):
        self.platforms = {
            'instagram': self.post_to_instagram,
            'twitter': self.post_to_twitter,
            'tiktok': self.post_to_tiktok
        }
        self.api_clients = {}
        self.setup_database()
        
    def setup_database(self):
        """Initialize the SQLite database with required tables."""
        try:
            self.conn = sqlite3.connect('social_media.db', check_same_thread=False)
            self.cursor = self.conn.cursor()
            
            # Create posts table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    caption TEXT,
                    hashtags TEXT,
                    platform TEXT NOT NULL,
                    schedule_time TIMESTAMP NOT NULL,
                    user_id INTEGER NOT NULL,
                    status TEXT DEFAULT 'pending',
                    post_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create post_analytics table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS post_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id INTEGER,
                    likes INTEGER DEFAULT 0,
                    comments INTEGER DEFAULT 0,
                    shares INTEGER DEFAULT 0,
                    views INTEGER DEFAULT 0,
                    platform TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (post_id) REFERENCES posts (id)
                )
            ''')

            self.conn.commit()
            logging.info("Database setup completed successfully")

        except sqlite3.Error as e:
            logging.error(f"Database setup error: {str(e)}")
            raise

    def close_connection(self):
        """Close the database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()

    def save_post(self, content: PostContent) -> int:
        """Save post details to database."""
        try:
            self.cursor.execute('''
                INSERT INTO posts (content_type, file_path, caption, hashtags, 
                                 platform, schedule_time, user_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                content.content_type,
                str(content.file_path),
                content.caption,
                ','.join(content.hashtags),
                content.platform,
                content.schedule_time.isoformat(),
                content.user_id,
                content.status
            ))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            logging.error(f"Error saving post to database: {str(e)}")
            raise

    def update_post_status(self, post_id: int, status: str, platform_post_id: Optional[str] = None):
        """Update post status and platform-specific post ID."""
        try:
            if platform_post_id:
                self.cursor.execute('''
                    UPDATE posts 
                    SET status = ?, post_id = ? 
                    WHERE id = ?
                ''', (status, platform_post_id, post_id))
            else:
                self.cursor.execute('''
                    UPDATE posts 
                    SET status = ? 
                    WHERE id = ?
                ''', (status, post_id))
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Error updating post status: {str(e)}")
            raise

    def __del__(self):
        """Destructor to ensure database connection is closed."""
        self.close_connection()

    async def post_to_instagram(self, content: PostContent) -> Tuple[bool, str]:
        """Post content to Instagram."""
        try:
            if 'instagram' not in self.api_clients:
                return False, "Instagram client not initialized"

            client = self.api_clients['instagram']
            
            # Handle different content types
            if content.content_type == 'image':
                result = await client.post_photo(
                    content.file_path,
                    caption=content.caption,
                    hashtags=' '.join(content.hashtags)
                )
            elif content.content_type == 'video':
                result = await client.post_video(
                    content.file_path,
                    caption=content.caption,
                    hashtags=' '.join(content.hashtags)
                )
            elif content.content_type == 'story':
                result = await client.post_story(
                    content.file_path,
                    caption=content.caption
                )
            elif content.content_type == 'reel':
                result = await client.post_reel(
                    content.file_path,
                    caption=content.caption,
                    hashtags=' '.join(content.hashtags)
                )
            else:
                return False, f"Unsupported content type: {content.content_type}"

            if result.get('success'):
                # Log successful post
                await self._log_post(content, result.get('post_id'))
                return True, f"Successfully posted to Instagram with ID: {result.get('post_id')}"
            else:
                return False, f"Instagram posting failed: {result.get('error')}"

        except Exception as e:
            logging.error(f"Instagram posting error: {str(e)}")
            return False, f"Instagram posting error: {str(e)}"

    async def post_to_twitter(self, content: PostContent) -> Tuple[bool, str]:
        """Post content to Twitter."""
        try:
            if 'twitter' not in self.api_clients:
                return False, "Twitter client not initialized"

            client = self.api_clients['twitter']
            
            # Prepare tweet text with hashtags
            tweet_text = f"{content.caption}\n\n{' '.join(['#' + tag for tag in content.hashtags])}"
            
            # Handle different content types
            if content.content_type in ['image', 'video']:
                # Upload media first
                media_id = await client.upload_media(content.file_path)
                result = await client.post_tweet(
                    text=tweet_text,
                    media_ids=[media_id]
                )
            else:
                # Text-only tweet
                result = await client.post_tweet(text=tweet_text)

            if result.get('success'):
                await self._log_post(content, result.get('tweet_id'))
                return True, f"Successfully posted to Twitter with ID: {result.get('tweet_id')}"
            else:
                return False, f"Twitter posting failed: {result.get('error')}"

        except Exception as e:
            logging.error(f"Twitter posting error: {str(e)}")
            return False, f"Twitter posting error: {str(e)}"

    async def post_to_tiktok(self, content: PostContent) -> Tuple[bool, str]:
        """Post content to TikTok."""
        try:
            if 'tiktok' not in self.api_clients:
                return False, "TikTok client not initialized"

            client = self.api_clients['tiktok']
            
            if content.content_type != 'video':
                return False, "TikTok only supports video content"

            # Upload video
            result = await client.upload_video(
                video_file=content.file_path,
                description=f"{content.caption}\n\n{' '.join(['#' + tag for tag in content.hashtags])}"
            )

            if result.get('success'):
                await self._log_post(content, result.get('video_id'))
                return True, f"Successfully posted to TikTok with ID: {result.get('video_id')}"
            else:
                return False, f"TikTok posting failed: {result.get('error')}"

        except Exception as e:
            logging.error(f"TikTok posting error: {str(e)}")
            return False, f"TikTok posting error: {str(e)}"

    async def _log_post(self, content: PostContent, post_id: str):
        """Log posted content to database."""
        try:
            conn = sqlite3.connect('social_automation.db')
            c = conn.cursor()
            
            c.execute('''
                INSERT INTO post_history 
                (user_id, platform, content_type, file_path, caption, 
                 hashtags, post_id, posted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                content.user_id,
                content.platform,
                content.content_type,
                str(content.file_path),
                content.caption,
                json.dumps(content.hashtags),
                post_id,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"Post logging error: {str(e)}")

class PostScheduler:
    def __init__(self, social_media_manager: SocialMediaManager):
        self.manager = social_media_manager
        self.schedule_queue = Queue()
        self.worker_thread = threading.Thread(target=self._schedule_worker, daemon=True)
        self.worker_thread.start()

    def schedule_post(self, post_content: PostContent) -> Tuple[bool, str]:
        """Schedule a post for publishing."""
        try:
            conn = sqlite3.connect('social_automation.db')
            c = conn.cursor()
            
            # Insert post into database
            c.execute('''
                INSERT INTO scheduled_posts 
                (user_id, platform, content_type, file_path, caption, hashtags, 
                 schedule_time, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                post_content.user_id,
                post_content.platform,
                post_content.content_type,
                str(post_content.file_path),
                post_content.caption,
                json.dumps(post_content.hashtags),
                post_content.schedule_time.isoformat(),
                'pending'
            ))
            
            post_id = c.lastrowid
            conn.commit()
            conn.close()

            # Add to schedule queue
            self.schedule_queue.put((post_id, post_content))
            
            return True, f"Post scheduled successfully with ID: {post_id}"

        except Exception as e:
            logging.error(f"Scheduling error: {str(e)}")
            return False, f"Failed to schedule post: {str(e)}"

    def _schedule_worker(self):
        """Background worker to process scheduled posts."""
        while True:
            try:
                # Get next scheduled post
                post_id, post_content = self.schedule_queue.get()
                
                # Calculate sleep time until post
                now = datetime.now(pytz.UTC)
                sleep_time = (post_content.schedule_time - now).total_seconds()
                
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Publish post
                success, message = asyncio.run(self.publish_post(post_id, post_content))
                
                # Update post status
                self._update_post_status(post_id, 'published' if success else 'failed', message)

            except Exception as e:
                logging.error(f"Schedule worker error: {str(e)}")
                time.sleep(60)  # Wait before retrying

    async def publish_post(self, post_id: int, post_content: PostContent) -> Tuple[bool, str]:
        """Publish a post to the specified platform."""
        try:
            platform_handler = self.manager.platforms.get(post_content.platform)
            if not platform_handler:
                return False, f"Unsupported platform: {post_content.platform}"

            success, message = await platform_handler(post_content)
            
            if success:
                # Start tracking analytics
                asyncio.create_task(self._track_post_analytics(post_id, post_content))
            
            return success, message

        except Exception as e:
            logging.error(f"Publishing error: {str(e)}")
            return False, f"Failed to publish post: {str(e)}"

    def _update_post_status(self, post_id: int, status: str, message: str):
        """Update the status of a scheduled post."""
        try:
            conn = sqlite3.connect('social_automation.db')
            c = conn.cursor()
            
            c.execute('''
                UPDATE scheduled_posts 
                SET status = ?, 
                    updated_at = CURRENT_TIMESTAMP,
                    message = ?
                WHERE id = ?
            ''', (status, message, post_id))
            
            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"Status update error: {str(e)}")

    async def _track_post_analytics(self, post_id: int, post_content: PostContent):
        """Track post analytics periodically."""
        try:
            tracking_intervals = [1, 24, 48, 72]  # Hours to track analytics
            
            for hours in tracking_intervals:
                await asyncio.sleep(hours * 3600)
                
                analytics = await self._fetch_post_analytics(post_content)
                
                if analytics:
                    await self._save_analytics(post_id, analytics)

        except Exception as e:
            logging.error(f"Analytics tracking error: {str(e)}")

    async def _fetch_post_analytics(self, post_content: PostContent) -> Optional[Dict]:
        """Fetch analytics for a post from the platform API."""
        try:
            if post_content.platform == 'instagram':
                return await self._fetch_instagram_analytics(post_content.post_id)
            # Add other platform analytics here
            return None

        except Exception as e:
            logging.error(f"Analytics fetch error: {str(e)}")
            return None

class PostOptimizer:
    def __init__(self):
        self.hashtag_model = pipeline("text2text-generation", model="facebook/bart-large-cnn")
        self.caption_model = pipeline("text2text-generation", model="facebook/bart-large-cnn")

    async def optimize_post(self, post_content: PostContent) -> PostContent:
        """Optimize post content for better engagement."""
        try:
            # Optimize caption
            if post_content.caption:
                optimized_caption = await self._optimize_caption(post_content.caption)
                post_content.caption = optimized_caption

            # Generate relevant hashtags
            if not post_content.hashtags:
                hashtags = await self._generate_hashtags(post_content.caption)
                post_content.hashtags = hashtags

            # Optimize media if needed
            if post_content.file_path.exists():
                await self._optimize_media(post_content)

            return post_content

        except Exception as e:
            logging.error(f"Post optimization error: {str(e)}")
            return post_content

    async def _optimize_caption(self, caption: str) -> str:
        """Optimize caption for better engagement."""
        try:
            # Generate engaging caption
            optimized = self.caption_model(
                f"Make this caption more engaging: {caption}",
                max_length=100,
                num_return_sequences=1
            )[0]['generated_text']

            return optimized

        except Exception as e:
            logging.error(f"Caption optimization error: {str(e)}")
            return caption

def render_scheduler_page():
    """Render the post scheduler interface in Streamlit."""
    st.title("Post Scheduler")

    # Initialize managers
    social_media_manager = SocialMediaManager()
    scheduler = PostScheduler(social_media_manager)
    optimizer = PostOptimizer()

    with st.form("schedule_post_form"):
        # Platform selection
        platform = st.selectbox("Platform", ["instagram", "twitter", "tiktok"])
        
        # Content type selection
        content_type = st.selectbox("Content Type", ["image", "video", "reel", "story"])
        
        # File upload
        uploaded_file = st.file_uploader("Upload Media", type=["jpg", "png", "mp4"])
        
        # Caption and hashtags
        caption = st.text_area("Caption")
        hashtags = st.text_input("Hashtags (comma-separated)")
        
        # Scheduling
        schedule_date = st.date_input("Schedule Date")
        schedule_time = st.time_input("Schedule Time")
        
        # Optimization options
        optimize = st.checkbox("Optimize Post for Better Engagement")
        
        submit = st.form_submit_button("Schedule Post")

        if submit and uploaded_file:
            # Save uploaded file
            file_path = Path("uploads") / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Create post content
            post_content = PostContent(
                content_type=content_type,
                file_path=file_path,
                caption=caption,
                hashtags=hashtags.split(",") if hashtags else [],
                platform=platform,
                schedule_time=datetime.combine(schedule_date, schedule_time),
                user_id=st.session_state.user_id
            )

            # Optimize if requested
            if optimize:
                with st.spinner("Optimizing post..."):
                    post_content = asyncio.run(optimizer.optimize_post(post_content))

            # Schedule post
            success, message = scheduler.schedule_post(post_content)
            
            if success:
                st.success(message)
            else:
                st.error(message)

    # Display scheduled posts
    st.subheader("Scheduled Posts")
    display_scheduled_posts()

def display_scheduled_posts():
    """Display scheduled posts in a table."""
    conn = sqlite3.connect('social_automation.db')
    posts_df = pd.read_sql_query('''
        SELECT id, platform, content_type, schedule_time, status 
        FROM scheduled_posts 
        WHERE user_id = ? 
        ORDER BY schedule_time DESC
    ''', conn, params=(st.session_state.user_id,))
    conn.close()

    if not posts_df.empty:
        st.dataframe(posts_df)
    else:
        st.info("No scheduled posts found")