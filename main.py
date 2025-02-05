import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import logging
import os
from typing import Dict, Optional
import json
import yaml
from PIL import Image
import asyncio
import transformers
import subprocess
import tensorflow as tf

# Import custom modules
from auth import AuthManager, UserManager, render_login_page, render_registration_page
from video_manager import VideoProcessor, VideoDownloader, VideoEnhancer, render_video_downloader_page
from scheduler_manager import PostScheduler, SocialMediaManager, render_scheduler_page
from analytics_dashboard import AnalyticsDashboard, render_analytics_dashboard
from automation_manager import AutomationManager, CampaignManager, render_automation_page, render_campaign_page
from subscription_manager import SubscriptionManager, render_subscription_page
from security_manager import SecurityManager, setup_security_middleware
from ai_features import AIFeatureManager, init_ai_features
from content_generator import ContentGenerator, render_content_generator_page
from tensorflow_config import TensorFlowConfigManager


# Configure TensorFlow with a memory limit
config_manager = TensorFlowConfigManager(memory_limit=4)  # Limit GPU memory to 4GB
config_manager.configure_tensorflow()

# Define a simple Sequential model with unique layer names if needed by updating them later:
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(9,), name="dense_64"),
    tf.keras.layers.Dropout(0.2, name="dropout_0.2"),
    tf.keras.layers.Dense(32, activation='relu', name="dense_32"),
    tf.keras.layers.Dense(1, activation='sigmoid', name="dense_output")
])


class AppConfig:
    """Enhanced application configuration management with TensorFlow support."""
    def __init__(self):
        """Initialize the application configuration."""
        self.config: Dict = {}
        self.tf_config = TensorFlowConfigManager()  # Initialize TensorFlow config manager
        self.load_config()
        self.setup_logging()
        self.initialize_paths()
        self.setup_tensorflow()

    def load_config(self) -> None:
        """Load application configuration from YAML with fallback values."""
        try:
            config_path = Path('config/app_config.yaml')
            if not config_path.exists():
                self._create_default_config(config_path)
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Config loading error: {str(e)}")
            self.config = self._get_default_config()

    def _create_default_config(self, config_path: Path) -> None:
        """Create default configuration file if not exists."""
        config_path.parent.mkdir(exist_ok=True)
        default_config = self._get_default_config()
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

    def _get_default_config(self) -> Dict:
        """Return default configuration settings."""
        return {
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/app.log'
            },
            'paths': {
                'logs': 'logs',
                'uploads': 'uploads',
                'downloads': 'downloads',
                'temp': 'temp',
                'models': 'models'
            },
            'tensorflow': {
                'memory_growth': True,
                'eager_execution': True,
                'gpu_memory_limit': 0.7
            }
        }

    def setup_logging(self) -> None:
        """Configure application logging with rotation."""
        log_config = self.config.get('logging', {})
        log_path = Path(log_config.get('file', 'logs/app.log'))
        log_path.parent.mkdir(exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )

    def initialize_paths(self) -> None:
        """Create necessary application directories with proper permissions."""
        paths = self.config.get('paths', {})
        for path_name, path_value in paths.items():
            path = Path(path_value)
            try:
                path.mkdir(exist_ok=True, parents=True)
                os.chmod(path, 0o775)
            except Exception as e:
                logging.error(f"Failed to create or set permissions for path {path}: {str(e)}")

    def setup_tensorflow(self) -> None:
        """Configure TensorFlow settings using TensorFlowConfigManager."""
        try:
            self.tf_config.configure_tensorflow()
            success, message = self.tf_config.verify_configuration()
            if not success:
                logging.error(message)
                raise RuntimeError(message)
            system_info = self.tf_config.get_system_info()
            logging.info("TensorFlow configuration completed successfully")
            logging.info(f"System information: {system_info}")
        except Exception as e:
            logging.error(f"TensorFlow configuration error: {str(e)}")
            raise RuntimeError(f"Failed to configure TensorFlow properly: {str(e)}")

    def get_config(self, key: str, default: Optional[any] = None) -> any:
        """Safely retrieve configuration values."""
        try:
            return self.config[key]
        except KeyError:
            logging.warning(f"Configuration key '{key}' not found, using default value: {default}")
            return default

    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            with open('config/app_config.yaml', 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            logging.error(f"Failed to save configuration: {str(e)}")

    def get_tensorflow_config(self) -> TensorFlowConfigManager:
        """Get the TensorFlow configuration manager instance."""
        return self.tf_config


class SocialAutomationApp:
    """Main application class."""
    def __init__(self):
        self.config = AppConfig()
        self.setup_managers()
        self.initialize_session_state()
        self.setup_ui_components()

    def setup_managers(self):
        """Initialize various management components."""
        self.auth_manager = AuthManager()
        self.security_manager = SecurityManager()
        self.subscription_manager = SubscriptionManager()
        self.social_media_manager = SocialMediaManager()
        self.analytics_dashboard = AnalyticsDashboard()
        self.automation_manager = AutomationManager()
        self.ai_manager = AIFeatureManager()
        self.content_generator = ContentGenerator()

    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'subscription_plan' not in st.session_state:
            st.session_state.subscription_plan = None
        if 'dark_mode' not in st.session_state:
            st.session_state.dark_mode = False

    def setup_ui_components(self):
        """Setup UI components and theme."""
        st.set_page_config(
            page_title="Social Media Automation",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        try:
            with open('static/style.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        except FileNotFoundError:
            logging.error("Static CSS file not found: 'static/style.css'. Please ensure the file exists.")
        except Exception as e:
            logging.error(f"Failed to load custom CSS: {str(e)}")

    def render_sidebar(self):
        """Render application sidebar."""
        with st.sidebar:
            if st.session_state.authenticated:
                st.title(f"Welcome, {st.session_state.username}!")
                st.subheader("Navigation")
                page = st.radio(
                    "Select Page",
                    ["Dashboard", "Content Generator", "Video Tools", 
                     "Post Scheduler", "Automation", "Analytics", 
                     "Campaigns", "Settings"]
                )
                with st.expander("Settings"):
                    st.checkbox("Dark Mode", key="dark_mode")
                    if st.button("Logout"):
                        self.logout_user()
                return page
            return None

    def render_main_content(self, page: Optional[str]):
        """Render main content based on selected page."""
        if not page:
            return
        if page == "Dashboard":
            self.render_dashboard()
        elif page == "Content Generator":
            render_content_generator_page()
        elif page == "Video Tools":
            render_video_downloader_page()
        elif page == "Post Scheduler":
            render_scheduler_page()
        elif page == "Automation":
            render_automation_page()
        elif page == "Analytics":
            render_analytics_dashboard()
        elif page == "Campaigns":
            render_campaign_page()
        elif page == "Settings":
            self.render_settings()

    def render_dashboard(self):
        """Render main dashboard."""
        st.title("Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            self.render_quick_stat("Total Posts", "123")
        with col2:
            self.render_quick_stat("Scheduled Posts", "45")
        with col3:
            self.render_quick_stat("Active Automations", "7")
        with col4:
            self.render_quick_stat("Total Engagement", "1.2K")
        col1, col2 = st.columns(2)
        with col1:
            self.render_recent_activity()
        with col2:
            self.render_upcoming_posts()
        self.render_platform_overview()

    def render_quick_stat(self, label: str, value: str):
        """Render a quick stat box."""
        st.markdown(
            f"""
            <div class="stat-box">
                <h3>{label}</h3>
                <p class="stat-value">{value}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    def render_recent_activity(self):
        """Render recent activity section."""
        st.subheader("Recent Activity")
        conn = sqlite3.connect('social_automation.db')
        activities = pd.read_sql_query('''
            SELECT action_type, platform, timestamp 
            FROM activity_log 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 5
        ''', conn, params=(st.session_state.user_id,))
        conn.close()
        if not activities.empty:
            for _, activity in activities.iterrows():
                st.markdown(
                    f"**{activity['action_type']}** on {activity['platform']} "
                    f"({activity['timestamp']})"
                )
        else:
            st.info("No recent activity")

    def render_upcoming_posts(self):
        """Render upcoming scheduled posts."""
        st.subheader("Upcoming Posts")
        conn = sqlite3.connect('social_automation.db')
        posts = pd.read_sql_query('''
            SELECT platform, content_type, scheduled_time 
            FROM scheduled_posts 
            WHERE user_id = ? AND status = 'pending' 
            ORDER BY scheduled_time 
            LIMIT 5
        ''', conn, params=(st.session_state.user_id,))
        conn.close()
        if not posts.empty:
            for _, post in posts.iterrows():
                st.markdown(
                    f"**{post['content_type']}** on {post['platform']} "
                    f"({post['scheduled_time']})"
                )
        else:
            st.info("No upcoming posts")

    def render_platform_overview(self):
        """Render platform overview section."""
        st.subheader("Platform Overview")
        metrics = self.analytics_dashboard.fetch_analytics_data(
            st.session_state.user_id,
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
        if metrics:
            cols = st.columns(len(metrics))
            for i, (platform, data) in enumerate(metrics.items()):
                with cols[i]:
                    st.markdown(f"### {platform}")
                    st.metric(
                        "Followers",
                        f"{data['followers'].iloc[-1]:,}",
                        f"{data['followers'].iloc[-1] - data['followers'].iloc[0]:,}"
                    )
                    st.metric(
                        "Engagement Rate",
                        f"{data['engagement_rate'].mean():.2%}"
                    )
        else:
            st.info("No platform data available")

    def render_settings(self):
        """Render settings page."""
        st.title("Settings")
        st.subheader("Profile Settings")
        with st.form("profile_settings"):
            st.text_input("Email")
            st.selectbox("Timezone", ["UTC", "US/Eastern", "US/Pacific"])
            st.multiselect("Notification Preferences", ["Email", "Push", "In-App"])
            st.form_submit_button("Save Changes")
        st.subheader("Connected Accounts")
        self.render_connected_accounts()
        st.subheader("Subscription")
        self.render_subscription_details()

    def render_connected_accounts(self):
        """Render connected social media accounts."""
        platforms = ["Instagram", "Twitter", "TikTok"]
        for platform in platforms:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{platform}**")
            with col2:
                if st.button(f"Connect {platform}"):
                    self.connect_platform(platform)

    def render_subscription_details(self):
        """Render subscription details."""
        plan = st.session_state.subscription_plan
        if plan:
            st.markdown(f"**Current Plan:** {plan}")
            st.markdown("**Status:** Active")
            if st.button("Manage Subscription"):
                render_subscription_page()
        else:
            st.warning("No active subscription")
            if st.button("View Plans"):
                render_subscription_page()

    def connect_platform(self, platform: str):
        """Handle platform connection."""
        try:
            st.info(f"Connecting to {platform}...")
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")

    def logout_user(self):
        """Handle user logout."""
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.subscription_plan = None
        st.experimental_rerun()

    def run(self):
        """Run the application."""
        try:
            if not st.session_state.authenticated:
                tab1, tab2 = st.tabs(["Login", "Register"])
                with tab1:
                    render_login_page()
                with tab2:
                    render_registration_page()
                return
            page = self.render_sidebar()
            self.render_main_content(page)
        except Exception as e:
            logging.error(f"Application error: {str(e)}")
            st.error("An error occurred. Please try again later.")


if __name__ == "__main__":
    app = SocialAutomationApp()
    app.run()
