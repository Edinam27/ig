import openai
import numpy as np
from PIL import Image
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, VisionEncoderDecoderModel, ViTImageProcessor
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import json
import asyncio
from datetime import datetime
import os
from pathlib import Path
import streamlit as st


@dataclass
class GeneratedContent:
    """Data class for generated content."""
    text: str
    media_paths: List[str]
    hashtags: List[str]
    suggested_schedule: datetime
    metadata: Dict

class ContentGenerator:
    def __init__(self):
        self.setup_models()
        self.load_templates()
        self.initialize_storage()
        
        # API configurations
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        openai.api_key = self.openai_api_key

    def setup_models(self):
        """Initialize AI models for content generation."""
        try:
            # Text generation model
            self.text_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
            self.text_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            
            # Image caption generation model
            self.image_model = VisionEncoderDecoderModel.from_pretrained(
                'nlpconnect/vit-gpt2-image-captioning'
            )
            self.image_processor = ViTImageProcessor.from_pretrained(
                'nlpconnect/vit-gpt2-image-captioning'
            )
            
        except Exception as e:
            logging.error(f"Model initialization error: {str(e)}")
            raise

    def load_templates(self):
        """Load content templates and prompts."""
        try:
            with open('templates/content_templates.json', 'r') as f:
                self.templates = json.load(f)
        except Exception as e:
            logging.error(f"Template loading error: {str(e)}")
            self.templates = {}

    def initialize_storage(self):
        """Initialize storage for generated content."""
        self.content_dir = Path('generated_content')
        self.content_dir.mkdir(exist_ok=True)

    async def generate_content(self, 
                             content_type: str, 
                             parameters: Dict) -> GeneratedContent:
        """Generate content based on specified type and parameters."""
        try:
            # Generate text content
            text = await self.generate_text(content_type, parameters)
            
            # Generate or process media
            media_paths = await self.generate_media(content_type, parameters)
            
            # Generate hashtags
            hashtags = await self.generate_hashtags(text, content_type)
            
            # Determine optimal posting schedule
            schedule = await self.determine_schedule(content_type, parameters)
            
            # Generate metadata
            metadata = self.generate_metadata(content_type, parameters)
            
            return GeneratedContent(
                text=text,
                media_paths=media_paths,
                hashtags=hashtags,
                suggested_schedule=schedule,
                metadata=metadata
            )

        except Exception as e:
            logging.error(f"Content generation error: {str(e)}")
            raise

    async def generate_text(self, content_type: str, parameters: Dict) -> str:
        """Generate text content using AI."""
        try:
            # Get appropriate template
            template = self.templates.get(content_type, {}).get('text_template', '')
            
            # Generate with OpenAI
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Generate engaging social media content."},
                    {"role": "user", "content": f"Generate {content_type} content using: {template}"},
                    {"role": "user", "content": f"Parameters: {json.dumps(parameters)}"}
                ]
            )
            
            return response.choices[0].message.content

        except Exception as e:
            logging.error(f"Text generation error: {str(e)}")
            return ""

    async def generate_media(self, content_type: str, parameters: Dict) -> List[str]:
        """Generate or process media content."""
        try:
            media_paths = []
            
            if parameters.get('generate_image'):
                # Generate image using DALL-E
                response = await openai.Image.acreate(
                    prompt=parameters.get('image_prompt', ''),
                    n=1,
                    size="1024x1024"
                )
                
                # Save generated image
                image_url = response['data'][0]['url']
                image_path = await self.save_media(image_url, 'image')
                media_paths.append(str(image_path))
            
            if parameters.get('process_video'):
                # Process video content
                video_path = await self.process_video(parameters)
                media_paths.append(str(video_path))
            
            return media_paths

        except Exception as e:
            logging.error(f"Media generation error: {str(e)}")
            return []

    async def generate_hashtags(self, text: str, content_type: str) -> List[str]:
        """Generate relevant hashtags for content."""
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Generate relevant hashtags for social media content."},
                    {"role": "user", "content": f"Generate 5-10 hashtags for: {text}"}
                ]
            )
            
            hashtags = response.choices[0].message.content.split()
            return [tag.strip('#') for tag in hashtags if tag.startswith('#')]

        except Exception as e:
            logging.error(f"Hashtag generation error: {str(e)}")
            return []

    async def determine_schedule(self, content_type: str, parameters: Dict) -> datetime:
        """Determine optimal posting schedule."""
        try:
            # Consider user's timezone
            user_timezone = parameters.get('timezone', 'UTC')
            
            # Consider content type specific timing
            if content_type == 'story':
                # Stories often perform better during active hours
                return self.get_next_active_hour(user_timezone)
            elif content_type == 'reel':
                # Reels might perform better during entertainment hours
                return self.get_next_entertainment_hour(user_timezone)
            
            # Default to general optimal timing
            return self.get_next_optimal_time(user_timezone)

        except Exception as e:
            logging.error(f"Schedule determination error: {str(e)}")
            return datetime.now()

    def generate_metadata(self, content_type: str, parameters: Dict) -> Dict:
        """Generate metadata for content."""
        return {
            'generated_at': datetime.now().isoformat(),
            'content_type': content_type,
            'parameters': parameters,
            'version': '1.0'
        }

    async def save_media(self, url: str, media_type: str) -> Path:
        """Save media content to local storage."""
        try:
            media_path = self.content_dir / f"{media_type}_{datetime.now().timestamp()}"
            
            # Download and save media
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    content = await response.read()
                    with open(media_path, 'wb') as f:
                        f.write(content)
            
            return media_path

        except Exception as e:
            logging.error(f"Media saving error: {str(e)}")
            raise

def render_content_generator_page():
    """Render the content generator interface in Streamlit."""
    st.title("AI Content Generator")
    
    content_generator = ContentGenerator()

    with st.form("content_generation_form"):
        content_type = st.selectbox(
            "Content Type",
            ["post", "story", "reel", "carousel"]
        )
        
        # Content parameters
        st.subheader("Content Parameters")
        
        tone = st.select_slider(
            "Content Tone",
            options=["Professional", "Casual", "Humorous", "Inspirational"]
        )
        
        target_audience = st.multiselect(
            "Target Audience",
            ["Teenagers", "Young Adults", "Professionals", "Parents"]
        )
        
        include_media = st.checkbox("Generate Media")
        if include_media:
            image_prompt = st.text_area("Image Generation Prompt")
        
        # Advanced options
        with st.expander("Advanced Options"):
            max_length = st.slider("Maximum Content Length", 50, 500, 200)
            include_emojis = st.checkbox("Include Emojis")
            hashtag_count = st.slider("Number of Hashtags", 0, 30, 5)
        
        submit = st.form_submit_button("Generate Content")
        
        if submit:
            with st.spinner("Generating content..."):
                # Prepare parameters
                parameters = {
                    'tone': tone,
                    'target_audience': target_audience,
                    'max_length': max_length,
                    'include_emojis': include_emojis,
                    'hashtag_count': hashtag_count,
                    'generate_image': include_media,
                    'image_prompt': image_prompt if include_media else None,
                    'timezone': st.session_state.get('timezone', 'UTC')
                }
                
                # Generate content
                content = asyncio.run(content_generator.generate_content(
                    content_type,
                    parameters
                ))
                
                # Display generated content
                st.subheader("Generated Content")
                st.write(content.text)
                
                if content.media_paths:
                    st.subheader("Generated Media")
                    for path in content.media_paths:
                        st.image(path)
                
                st.subheader("Suggested Hashtags")
                st.write(" ".join([f"#{tag}" for tag in content.hashtags]))
                
                st.subheader("Suggested Posting Time")
                st.write(content.suggested_schedule.strftime("%Y-%m-%d %H:%M:%S"))

# Initialize content generation features
def init_content_generator():
    if 'content_generator' not in st.session_state:
        st.session_state.content_generator = ContentGenerator()

if __name__ == "__main__":
    init_content_generator()