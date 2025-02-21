# ai_features.py
import streamlit as st
import openai
import tensorflow as tf
import subprocess
import torch
from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    ViTModel  # Changed from VisionTransformer
)
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import json
from datetime import datetime
import asyncio
import aiohttp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import spacy
from textblob import TextBlob
import os
import psutil
from pathlib import Path
import yaml
import platform



@dataclass
class ContentAnalysis:
    """Data class for content analysis results."""
    sentiment_score: float
    engagement_prediction: float
    content_quality_score: float
    suggested_improvements: List[str]
    best_posting_time: datetime
    hashtag_suggestions: List[str]
    target_audience: Dict[str, float]
    
    
class TensorFlowConfigManager:
    """Manages TensorFlow configuration and hardware detection."""
    
    def __init__(self, memory_limit: Optional[float] = None):
        """
        Initialize TensorFlow configuration manager.
        
        Args:
            memory_limit (float, optional): GPU memory limit in GB
        """
        self.logger = logging.getLogger(__name__)
        self.memory_limit = memory_limit
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configure logging for the TensorFlow manager."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def configure_tensorflow(self) -> None:
        """Configure TensorFlow based on available hardware."""
        try:
            # Check CUDA availability
            cuda_available = tf.test.is_built_with_cuda()
            gpu_available = tf.config.list_physical_devices('GPU')
            
            if not cuda_available or not gpu_available:
                self._configure_cpu_only()
                return
                
            self._configure_gpu()
            
        except Exception as e:
            self.logger.error(f"Error configuring TensorFlow: {str(e)}")
            self._configure_cpu_only()

    def _configure_gpu(self) -> None:
        """Configure GPU settings for TensorFlow."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    # Memory growth must be set before GPUs have been initialized
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                    if self.memory_limit:
                        tf.config.set_logical_device_configuration(
                            gpu,
                            [tf.config.LogicalDeviceConfiguration(
                                memory_limit=self.memory_limit * 1024
                            )]
                        )
                
                self.logger.info(f"GPU configuration successful. Found {len(gpus)} GPU(s)")
            else:
                self._configure_cpu_only()
                
        except Exception as e:
            self.logger.error(f"GPU configuration failed: {str(e)}")
            self._configure_cpu_only()

    def _configure_cpu_only(self) -> None:
        """Configure TensorFlow to use CPU only."""
        try:
            tf.config.set_visible_devices([], 'GPU')
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.logger.info("TensorFlow configured to use CPU only")
        except Exception as e:
            self.logger.error(f"CPU configuration failed: {str(e)}")

    def get_system_info(self) -> Dict:
        """Get system information relevant to TensorFlow."""
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'tensorflow_version': tf.__version__,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024 ** 3),  # GB
            'cuda_available': tf.test.is_built_with_cuda(),
            'gpu_devices': [str(gpu) for gpu in tf.config.list_physical_devices('GPU')]
        }
        return info

    def verify_installation(self) -> Tuple[bool, str]:
        """Verify TensorFlow installation and configuration."""
        try:
            # Basic TensorFlow operation test
            test_tensor = tf.random.uniform((3, 3))
            tf.matmul(test_tensor, test_tensor)
            
            return True, "TensorFlow verification successful"
        except Exception as e:
            return False, f"TensorFlow verification failed: {str(e)}"

    def get_recommended_settings(self) -> Dict:
        """Get recommended TensorFlow settings based on system configuration."""
        system_memory = psutil.virtual_memory().total / (1024 ** 3)  # GB
        
        settings = {
            'memory_limit': min(system_memory * 0.7, 4),  # 70% of system memory or 4GB
            'allow_growth': True,
            'mixed_precision': system_memory >= 16,
            'num_parallel_calls': min(psutil.cpu_count(), 8)
        }
        
        return settings

class AIFeatureManager:
    """Manages AI-powered features with content guidelines."""
    
    def __init__(self, config_path: str = 'config/ai_config.yaml'):
        """
        Initialize AI Feature Manager with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self._setup_paths()
        self._initialize_models()
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging with proper formatting."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file not found at {config_path}. Using defaults.")
            return self._create_default_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        return {
            'models': {
                'sentiment': 'distilbert-base-uncased-finetuned-sst-2-english',
                'quality': 'bert-base-uncased',
                'engagement': {
                    'path': 'models/engagement_predictor.h5',
                    'input_shape': (9,)
                }
            },
            'paths': {
                'models': 'models',
                'data': 'data',
                'cache': 'cache'
            },
            'nlp': {
                'spacy_model': 'en_core_web_sm',
                'max_length': 512
            }
        }
        
    def _setup_paths(self) -> None:
        """Create necessary directories."""
        for path in self.config['paths'].values():
            Path(path).mkdir(parents=True, exist_ok=True)
            
    def setup_models(self):
        """Initialize AI models and pipelines."""
        try:
            # Initialize engagement model manager
            self.engagement_manager = EngagementModelManager()
            self.engagement_model = self.engagement_manager.get_model()
            
            # Rest of your model initialization code...
            
        except Exception as e:
            logging.error(f"Model initialization error: {str(e)}")
            raise


    def _initialize_models(self) -> None:
        """Initialize all AI models."""
        try:
            self._init_sentiment_analyzer()
            self._init_quality_model()
            self._init_engagement_model()
            self._init_nlp()
        except Exception as e:
            self.logger.error(f"Model initialization error: {str(e)}")
            raise
    def _init_sentiment_analyzer(self) -> None:
        """Initialize sentiment analysis pipeline."""
        try:
            self.sentiment_analyzer = pipeline(
                'sentiment-analysis',
                model=self.config['models']['sentiment']
            )
        except Exception as e:
            self.logger.error(f"Sentiment analyzer initialization error: {str(e)}")
            raise
    
    def _init_quality_model(self) -> None:
        """Initialize content quality model."""
        try:
            model_name = self.config['models']['quality']
            self.quality_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.quality_tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            self.logger.error(f"Quality model initialization error: {str(e)}")
            raise
    
    def _init_engagement_model(self) -> None:
        """Initialize engagement prediction model."""
        try:
            model_path = self.config['models']['engagement']['path']
            if os.path.exists(model_path):
                self.engagement_model = tf.keras.models.load_model(model_path)
            else:
                self._create_engagement_model()
        except Exception as e:
            self.logger.error(f"Engagement model initialization error: {str(e)}")
            raise        
        
    def _create_engagement_model(self) -> None:
        """Create and save new engagement model."""
        input_shape = self.config['models']['engagement']['input_shape']
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self.engagement_model = model
        model.save(self.config['models']['engagement']['path'])
    
    def _init_nlp(self) -> None:
        """Initialize NLP pipeline."""
        try:
            self.nlp = spacy.load(self.config['nlp']['spacy_model'])
        except OSError:
            self.logger.info(f"Downloading spacy model {self.config['nlp']['spacy_model']}")
            spacy.cli.download(self.config['nlp']['spacy_model'])
            self.nlp = spacy.load(self.config['nlp']['spacy_model'])
    
    async def analyze_content(self, content: Dict[str, Any]) -> ContentAnalysis:
        """
        Analyze content and provide comprehensive analysis.
        
        Args:
            content: Dictionary containing content data
        
        Returns:
            ContentAnalysis object with analysis results
        """
        try:
            text = content.get('text', '')
            if not text:
                raise ValueError("Content must include text")

            sentiment_score = await self.analyze_sentiment(text)
            engagement_pred = await self.predict_engagement(content)
            quality_score = await self.assess_content_quality(content)
            hashtags = await self.generate_hashtags(content)
            posting_time = await self.determine_posting_time(content)
            target_audience = await self.analyze_target_audience(content)

            improvements = self.generate_improvements([
                sentiment_score,
                engagement_pred,
                quality_score
            ])

            return ContentAnalysis(
                sentiment_score=sentiment_score,
                engagement_prediction=engagement_pred,
                content_quality_score=quality_score,
                suggested_improvements=improvements,
                best_posting_time=posting_time,
                hashtag_suggestions=hashtags,
                target_audience=target_audience
            )
        except Exception as e:
            self.logger.error(f"Content analysis error: {str(e)}")
            raise
    def load_content_guidelines(self) -> None:
        """Load content guidelines from configuration file."""
        try:
            guideline_path = Path('config/content_guidelines.yaml')
            
            # Create default guidelines if file doesn't exist
            if not guideline_path.exists():
                self._create_default_guidelines()
            
            with open(guideline_path, 'r') as f:
                guidelines_data = yaml.safe_load(f)
                
            self.guidelines = ContentGuidelines(
                min_length=guidelines_data.get('min_length', 50),
                max_length=guidelines_data.get('max_length', 2000),
                required_elements=guidelines_data.get('required_elements', []),
                forbidden_words=guidelines_data.get('forbidden_words', []),
                tone_preferences=guidelines_data.get('tone_preferences', []),
                category_rules=guidelines_data.get('category_rules', {}),
                platform_specific=guidelines_data.get('platform_specific', {})
            )
            
            self.logger.info("Content guidelines loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading content guidelines: {str(e)}")
            self._create_default_guidelines()
            
    def _create_default_guidelines(self) -> None:
        """Create default content guidelines configuration."""
        default_guidelines = {
            'min_length': 50,
            'max_length': 2000,
            'required_elements': ['hashtags', 'call_to_action'],
            'forbidden_words': ['spam', 'inappropriate'],
            'tone_preferences': ['professional', 'engaging', 'friendly'],
            'category_rules': {
                'business': {
                    'required_elements': ['branding', 'value_proposition'],
                    'tone': 'professional'
                },
                'personal': {
                    'required_elements': ['personal_story', 'engagement_question'],
                    'tone': 'casual'
                }
            },
            'platform_specific': {
                'instagram': {
                    'max_hashtags': 30,
                    'image_required': True
                },
                'twitter': {
                    'max_length': 280,
                    'max_hashtags': 3
                }
            }
        }
        
        with open('config/content_guidelines.yaml', 'w') as f:
            yaml.dump(default_guidelines, f, default_flow_style=False)
            
        self.guidelines = ContentGuidelines(**default_guidelines)
        self.logger.info("Created default content guidelines")
        
    def validate_content(self, content: Dict, platform: str) -> Dict[str, bool]:
        """Validate content against guidelines for specific platform."""
        validation_results = {
            'length_valid': self._validate_length(content.get('text', ''), platform),
            'elements_present': self._validate_required_elements(content),
            'no_forbidden_words': self._validate_forbidden_words(content.get('text', '')),
            'tone_appropriate': self._validate_tone(content.get('text', '')),
            'platform_compliant': self._validate_platform_rules(content, platform)
        }
        
        return validation_results
        
    def _validate_length(self, text: str, platform: str) -> bool:
        """Validate content length against platform-specific guidelines."""
        length = len(text)
        platform_rules = self.guidelines.platform_specific.get(platform, {})
        max_length = platform_rules.get('max_length', self.guidelines.max_length)
        return self.guidelines.min_length <= length <= max_length
        
    def _validate_required_elements(self, content: Dict) -> bool:
        """Validate presence of required content elements."""
        required = set(self.guidelines.required_elements)
        present = set(content.keys())
        return required.issubset(present)
        
    def _validate_forbidden_words(self, text: str) -> bool:
        """Check for presence of forbidden words."""
        text_lower = text.lower()
        return not any(word in text_lower for word in self.guidelines.forbidden_words)
        
    def _validate_tone(self, text: str) -> bool:
        """Validate content tone against preferences."""
        # Implement tone analysis logic here
        return True
        
    def _validate_platform_rules(self, content: Dict, platform: str) -> bool:
        """Validate content against platform-specific rules."""
        platform_rules = self.guidelines.platform_specific.get(platform, {})
        
        # Check platform-specific requirements
        if platform_rules.get('image_required', False):
            if 'image' not in content:
                return False
                
        if 'hashtags' in content:
            max_hashtags = platform_rules.get('max_hashtags', float('inf'))
            if len(content['hashtags']) > max_hashtags:
                return False
                
        return True
        
    def setup_engagement_model(self):
        """Setup engagement prediction model."""
        try:
            model_path = 'models/engagement_predictor.h5'
            if os.path.exists(model_path):
                self.engagement_model = tf.keras.models.load_model(model_path)
            else:
                # Create a simple model if none exists
                self.engagement_model = tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                self.engagement_model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
        except Exception as e:
            logging.error(f"Engagement model setup error: {str(e)}")
            raise

    def initialize_nlp(self) -> None:
        """
        Initialize NLP components by loading the Spacy model.
        If 'en_core_web_sm' is not available, download it automatically.
        """
        try:
            self.logger.info("Attempting to load the Spacy model 'en_core_web_sm'...")
            self.nlp = spacy.load("en_core_web_sm")
        except IOError as e:
            self.logger.info("Model 'en_core_web_sm' not found. Attempting to download it...")
            try:
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("Successfully downloaded and loaded 'en_core_web_sm'.")
            except Exception as download_error:
                self.logger.error(f"Failed to download 'en_core_web_sm': {download_error}")
                raise download_error
        except Exception as e:
            self.logger.error(f"Unexpected error during NLP initialization: {e}")
            raise

        self.logger.info("NLP initialization completed successfully.")

    async def analyze_content(self, content: Dict) -> ContentAnalysis:
        """Analyze content and provide optimization suggestions."""
        try:
            # Parallel analysis tasks
            sentiment_task = self.analyze_sentiment(content['text'])
            engagement_task = self.predict_engagement(content)
            quality_task = self.assess_content_quality(content)
            hashtag_task = self.generate_hashtags(content)
            timing_task = self.determine_posting_time(content)
            
            # Gather results
            results = await asyncio.gather(
                sentiment_task,
                engagement_task,
                quality_task,
                hashtag_task,
                timing_task
            )
            
            # Generate improvements based on analysis
            improvements = self.generate_improvements(results)
            
            # Analyze target audience
            target_audience = await self.analyze_target_audience(content)
            
            return ContentAnalysis(
                sentiment_score=results[0],
                engagement_prediction=results[1],
                content_quality_score=results[2],
                suggested_improvements=improvements,
                best_posting_time=results[4],
                hashtag_suggestions=results[3],
                target_audience=target_audience
            )

        except Exception as e:
            logging.error(f"Content analysis error: {str(e)}")
            raise

    async def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text content."""
        try:
            result = self.sentiment_analyzer(text)[0]
            score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
            
            # Additional context analysis
            blob = TextBlob(text)
            subjectivity = blob.sentiment.subjectivity
            
            # Combine scores with weights
            weighted_score = (score * 0.7) + (subjectivity * 0.3)
            return max(-1.0, min(1.0, weighted_score))

        except Exception as e:
            logging.error(f"Sentiment analysis error: {str(e)}")
            return 0.0

    async def predict_engagement(self, content: Dict) -> float:
        """Predict potential engagement rate."""
        try:
            # Extract features
            features = self.extract_engagement_features(content)
            
            # Make prediction
            prediction = self.engagement_model.predict(
                np.array([features])
            )[0]
            
            # Normalize prediction
            return float(prediction[0])

        except Exception as e:
            logging.error(f"Engagement prediction error: {str(e)}")
            return 0.0

    def extract_engagement_features(self, content: Dict) -> List[float]:
        """Extract features for engagement prediction."""
        features = []
        
        try:
            text = content.get('text', '')
            
            # Text features
            doc = self.nlp(text)
            features.extend([
                len(text),  # Content length
                len(doc.ents),  # Named entities
                sum(1 for token in doc if token.is_stop),  # Stop words
                sum(1 for token in doc if token.pos_ == 'VERB'),  # Verbs
                TextBlob(text).sentiment.subjectivity  # Subjectivity
            ])
            
            # Time features
            post_time = content.get('scheduled_time', datetime.now())
            features.extend([
                post_time.hour / 24.0,
                post_time.weekday() / 7.0
            ])
            
            # Media features
            has_image = 1.0 if content.get('image') else 0.0
            has_video = 1.0 if content.get('video') else 0.0
            features.extend([has_image, has_video])
            
            return features

        except Exception as e:
            logging.error(f"Feature extraction error: {str(e)}")
            return [0.0] * 9

    async def assess_content_quality(self, content: Dict) -> float:
        """Assess overall content quality."""
        try:
            text = content.get('text', '')
            
            # Prepare input for quality model
            inputs = self.quality_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Get quality prediction
            with torch.no_grad():
                outputs = self.quality_model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                quality_score = float(scores[0][1])  # Positive class score
            
            # Additional quality checks
            spelling_score = self.check_spelling(text)
            readability_score = self.calculate_readability(text)
            
            # Combine scores with weights
            weighted_score = (
                quality_score * 0.5 +
                spelling_score * 0.25 +
                readability_score * 0.25
            )
            
            return weighted_score

        except Exception as e:
            logging.error(f"Quality assessment error: {str(e)}")
            return 0.0

    async def generate_hashtags(self, content: Dict) -> List[str]:
        """Generate relevant hashtags for content."""
        try:
            text = content.get('text', '')
            
            # Extract keywords using NLP
            doc = self.nlp(text)
            keywords = [
                token.text.lower() for token in doc 
                if not token.is_stop and token.is_alpha
            ]
            
            # Get trending hashtags from API
            trending = await self.fetch_trending_hashtags()
            
            # Find relevant trending hashtags
            relevant_trending = self.find_relevant_hashtags(
                keywords,
                trending
            )
            
            # Generate additional hashtags using OpenAI
            generated = await self.generate_ai_hashtags(text)
            
            # Combine and filter hashtags
            all_hashtags = set(relevant_trending + generated)
            return list(all_hashtags)[:10]  # Limit to top 10

        except Exception as e:
            logging.error(f"Hashtag generation error: {str(e)}")
            return []

    async def generate_ai_hashtags(self, text: str) -> List[str]:
        """Generate hashtags using OpenAI."""
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Generate relevant hashtags for social media content."},
                    {"role": "user", "content": f"Generate 5 relevant hashtags for this content: {text}"}
                ]
            )
            
            hashtags = response.choices[0].message.content.split()
            return [tag.strip('#') for tag in hashtags if tag.startswith('#')]

        except Exception as e:
            logging.error(f"AI hashtag generation error: {str(e)}")
            return []

    async def determine_posting_time(self, content: Dict) -> datetime:
        """Determine the optimal posting time."""
        try:
            # Load historical engagement data
            engagement_data = self.load_engagement_data()
            
            # Analyze patterns
            best_times = self.analyze_engagement_patterns(engagement_data)
            
            # Consider content type and target audience
            content_type = content.get('type', 'text')
            target_audience = content.get('target_audience', {})
            
            # Adjust timing based on content type and audience
            adjusted_time = self.adjust_posting_time(
                best_times,
                content_type,
                target_audience
            )
            
            return adjusted_time

        except Exception as e:
            logging.error(f"Posting time determination error: {str(e)}")
            return datetime.now()

    async def analyze_target_audience(self, content: Dict) -> Dict[str, float]:
        """Analyze and predict target audience demographics."""
        try:
            # Extract content features
            text_features = self.extract_text_features(content.get('text', ''))
            media_features = await self.extract_media_features(content)
            
            # Combine features
            features = np.concatenate([text_features, media_features])
            
            # Predict demographics
            demographics = {}
            
            # Age groups
            age_groups = ['13-17', '18-24', '25-34', '35-44', '45-54', '55+']
            age_predictions = self.predict_demographic('age', features)
            demographics['age'] = dict(zip(age_groups, age_predictions))
            
            # Interests
            interests = self.predict_interests(features)
            demographics['interests'] = interests
            
            # Location relevance
            locations = await self.predict_location_relevance(content)
            demographics['locations'] = locations
            
            return demographics

        except Exception as e:
            logging.error(f"Target audience analysis error: {str(e)}")
            return {}
        
    async def analyze_content(self, content: Dict) -> ContentAnalysis:
        """Analyze content and provide optimization suggestions."""
        try:
            # Input validation
            if not content.get('text'):
                raise ValueError("Content must include text")

            # Parallel analysis tasks
            sentiment_task = self.analyze_sentiment(content['text'])
            engagement_task = self.predict_engagement(content)
            quality_task = self.assess_content_quality(content)
            hashtag_task = self.generate_hashtags(content)
            timing_task = self.determine_posting_time(content)
            
            # Gather results
            results = await asyncio.gather(
                sentiment_task,
                engagement_task,
                quality_task,
                hashtag_task,
                timing_task
            )
            
            # Generate improvements based on analysis
            improvements = self.generate_improvements(results)
            
            # Analyze target audience
            target_audience = await self.analyze_target_audience(content)
            
            return ContentAnalysis(
                sentiment_score=results[0],
                engagement_prediction=results[1],
                content_quality_score=results[2],
                suggested_improvements=improvements,
                best_posting_time=results[4],
                hashtag_suggestions=results[3],
                target_audience=target_audience
            )

        except Exception as e:
            logging.error(f"Content analysis error: {str(e)}")
            raise

    def generate_improvements(self, analysis_results: List) -> List[str]:
        """Generate content improvement suggestions."""
        improvements = []
        
        try:
            sentiment_score = analysis_results[0]
            engagement_pred = analysis_results[1]
            quality_score = analysis_results[2]
            
            # Sentiment-based improvements
            if sentiment_score < 0.2:
                improvements.append(
                    "Consider using more positive language to increase engagement"
                )
            
            # Quality-based improvements
            if quality_score < 0.6:
                improvements.append(
                    "Improve content quality by adding more detailed information"
                )
            
            # Engagement-based improvements
            if engagement_pred < 0.4:
                improvements.append(
                    "Add more engaging elements like questions or calls to action"
                )
            
            return improvements

        except Exception as e:
            logging.error(f"Improvement generation error: {str(e)}")
            return ["Unable to generate improvements"]

    def check_spelling(self, text: str) -> float:
        """Check spelling and grammar."""
        try:
            doc = self.nlp(text)
            total_words = len([token for token in doc if token.is_alpha])
            if total_words == 0:
                return 1.0
                
            misspelled = len([token for token in doc if token.is_alpha and not token.is_stop and not token.is_punct])
            return 1.0 - (misspelled / total_words)

        except Exception as e:
            logging.error(f"Spelling check error: {str(e)}")
            return 1.0

    def calculate_readability(self, text: str) -> float:
        """Calculate text readability score."""
        try:
            doc = self.nlp(text)
            
            # Calculate basic metrics
            words = len([token for token in doc if not token.is_punct])
            sentences = len(list(doc.sents))
            syllables = sum([self.count_syllables(token.text) for token in doc])
            
            if words == 0 or sentences == 0:
                return 0.0
            
            # Calculate Flesch Reading Ease score
            score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
            
            # Normalize score between 0 and 1
            return max(0.0, min(1.0, score / 100.0))

        except Exception as e:
            logging.error(f"Readability calculation error: {str(e)}")
            return 0.0

    def count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        try:
            count = 0
            vowels = 'aeiouy'
            word = word.lower()
            if word[0] in vowels:
                count += 1
            for index in range(1, len(word)):
                if word[index] in vowels and word[index - 1] not in vowels:
                    count += 1
            if word.endswith('e'):
                count -= 1
            if count == 0:
                count += 1
            return count
        except:
            return 1
        
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ContentGuidelines:
    """Data class for content guidelines configuration."""
    min_length: int
    max_length: int
    required_elements: List[str]
    forbidden_words: List[str]
    tone_preferences: List[str]
    category_rules: Dict[str, Dict]
    platform_specific: Dict[str, Dict]

    def validate_platform_rules(self, platform: str) -> Dict[str, any]:
        """Get platform-specific rules."""
        return self.platform_specific.get(platform, {})

    def get_category_rules(self, category: str) -> Dict[str, any]:
        """Get category-specific rules."""
        return self.category_rules.get(category, {})

    def get_max_length(self, platform: str) -> int:
        """Get maximum content length for platform."""
        platform_rules = self.validate_platform_rules(platform)
        return platform_rules.get('max_length', self.max_length)

    def get_required_elements(self, category: str = None) -> List[str]:
        """Get required elements for content."""
        elements = self.required_elements.copy()
        if category:
            category_rules = self.get_category_rules(category)
            elements.extend(category_rules.get('required_elements', []))
        return list(set(elements))

    def get_tone(self, category: str = None) -> str:
        """Get preferred tone for content."""
        if category:
            category_rules = self.get_category_rules(category)
            return category_rules.get('tone', self.tone_preferences[0])
        return self.tone_preferences[0]

    def get_hashtag_limit(self, platform: str) -> int:
        """Get maximum number of hashtags for platform."""
        platform_rules = self.validate_platform_rules(platform)
        return platform_rules.get('max_hashtags', float('inf'))

    def requires_image(self, platform: str) -> bool:
        """Check if platform requires images."""
        platform_rules = self.validate_platform_rules(platform)
        return platform_rules.get('image_required', False)
    
import tensorflow as tf
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

class EngagementModelManager:
    """Manages the engagement prediction model lifecycle."""
    
    def __init__(self, model_path: str = 'models/engagement_predictor.h5'):
        """
        Initialize the engagement model manager.
        
        Args:
            model_path (str): Path to save/load the model
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self._ensure_model_directory()
        
    def _ensure_model_directory(self) -> None:
        """Ensure the model directory exists."""
        Path(os.path.dirname(self.model_path)).mkdir(parents=True, exist_ok=True)
        
    def get_model(self) -> tf.keras.Model:
        """Get the engagement prediction model, creating it if necessary."""
        try:
            if os.path.exists(self.model_path):
                return self._load_existing_model()
            return self._create_new_model()
        except Exception as e:
            self.logger.error(f"Error getting engagement model: {str(e)}")
            return self._create_new_model()
            
    def _load_existing_model(self) -> tf.keras.Model:
        """Load existing model from disk."""
        try:
            model = tf.keras.models.load_model(self.model_path)
            self.logger.info("Loaded existing engagement model")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return self._create_new_model()
            
    def _create_new_model(self) -> tf.keras.Model:
        """Create and save a new engagement prediction model."""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )
            
            # Save the model
            model.save(self.model_path)
            self.logger.info("Created and saved new engagement model")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating model: {str(e)}")
            raise
            
    def train_model(self, 
                    X_train: tf.Tensor, 
                    y_train: tf.Tensor,
                    validation_data: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
                    epochs: int = 50,
                    batch_size: int = 32) -> tf.keras.callbacks.History:
        """
        Train the engagement prediction model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_data: Optional tuple of (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Training history
        """
        try:
            model = self.get_model()
            
            # Configure callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    self.model_path,
                    monitor='val_loss' if validation_data else 'loss',
                    save_best_only=True
                )
            ]
            
            # Train the model
            history = model.fit(
                X_train,
                y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
            
            self.logger.info("Model training completed successfully")
            return history
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
            
    def predict_engagement(self, features: tf.Tensor) -> float:
        """
        Predict engagement score for given features.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Predicted engagement score
        """
        try:
            model = self.get_model()
            prediction = model.predict(tf.expand_dims(features, 0))[0][0]
            return float(prediction)
            
        except Exception as e:
            self.logger.error(f"Error predicting engagement: {str(e)}")
            return 0.0

def init_ai_features():
    """Initialize AI features in Streamlit."""
    if 'ai_manager' not in st.session_state:
        st.session_state.ai_manager = AIFeatureManager()

# Usage in main app
if __name__ == "__main__":
    init_ai_features()
