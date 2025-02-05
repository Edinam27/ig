# automation_manager.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Optional, Tuple, Union
import asyncio
import aiohttp
import json
import logging
from dataclasses import dataclass
import schedule
import time
import threading
from queue import Queue
import re
from collections import defaultdict

@dataclass
class AutomationRule:
    """Data class for automation rules."""
    id: Optional[int]
    user_id: int
    name: str
    trigger_type: str
    trigger_conditions: Dict
    actions: List[Dict]
    status: str
    priority: int
    created_at: datetime
    last_run: Optional[datetime]

@dataclass
class Campaign:
    """Data class for marketing campaigns."""
    id: Optional[int]
    user_id: int
    name: str
    start_date: datetime
    end_date: datetime
    platform: str
    budget: float
    target_audience: Dict
    content_strategy: Dict
    automation_rules: List[int]
    status: str
    metrics: Dict

class AutomationManager:
    def __init__(self):
        self.setup_database()
        self.rule_queue = Queue()
        self.worker_thread = threading.Thread(target=self._automation_worker, daemon=True)
        self.worker_thread.start()

    def setup_database(self):
        """Initialize automation and campaign database tables."""
        conn = sqlite3.connect('social_automation.db')
        c = conn.cursor()
        
        # Automation rules table
        c.execute('''
            CREATE TABLE IF NOT EXISTS automation_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT,
                trigger_type TEXT,
                trigger_conditions TEXT,
                actions TEXT,
                status TEXT,
                priority INTEGER,
                created_at TIMESTAMP,
                last_run TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Campaigns table
        c.execute('''
            CREATE TABLE IF NOT EXISTS campaigns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                platform TEXT,
                budget REAL,
                target_audience TEXT,
                content_strategy TEXT,
                automation_rules TEXT,
                status TEXT,
                metrics TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Campaign content table
        c.execute('''
            CREATE TABLE IF NOT EXISTS campaign_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id INTEGER,
                content_type TEXT,
                content_path TEXT,
                schedule_time TIMESTAMP,
                status TEXT,
                metrics TEXT,
                FOREIGN KEY (campaign_id) REFERENCES campaigns (id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def create_automation_rule(self, rule: AutomationRule) -> Tuple[bool, str]:
        """Create a new automation rule."""
        try:
            conn = sqlite3.connect('social_automation.db')
            c = conn.cursor()
            
            c.execute('''
                INSERT INTO automation_rules 
                (user_id, name, trigger_type, trigger_conditions, actions, 
                 status, priority, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.user_id,
                rule.name,
                rule.trigger_type,
                json.dumps(rule.trigger_conditions),
                json.dumps(rule.actions),
                rule.status,
                rule.priority,
                datetime.now().isoformat()
            ))
            
            rule_id = c.lastrowid
            conn.commit()
            conn.close()
            
            # Add to queue if active
            if rule.status == 'active':
                self.rule_queue.put((rule_id, rule))
            
            return True, f"Rule created successfully with ID: {rule_id}"

        except Exception as e:
            logging.error(f"Error creating automation rule: {str(e)}")
            return False, f"Failed to create rule: {str(e)}"

    def _automation_worker(self):
        """Background worker to process automation rules."""
        while True:
            try:
                # Process active rules
                conn = sqlite3.connect('social_automation.db')
                c = conn.cursor()
                
                c.execute('''
                    SELECT id, trigger_type, trigger_conditions, actions 
                    FROM automation_rules 
                    WHERE status = 'active'
                    ORDER BY priority DESC
                ''')
                
                active_rules = c.fetchall()
                conn.close()

                for rule_id, trigger_type, conditions, actions in active_rules:
                    try:
                        conditions = json.loads(conditions)
                        actions = json.loads(actions)
                        
                        if self._check_trigger_conditions(trigger_type, conditions):
                            self._execute_actions(rule_id, actions)
                    except Exception as e:
                        logging.error(f"Error processing rule {rule_id}: {str(e)}")

                time.sleep(60)  # Check rules every minute

            except Exception as e:
                logging.error(f"Automation worker error: {str(e)}")
                time.sleep(60)

    def _check_trigger_conditions(self, trigger_type: str, conditions: Dict) -> bool:
        """Check if trigger conditions are met."""
        try:
            if trigger_type == 'schedule':
                return self._check_schedule_trigger(conditions)
            elif trigger_type == 'engagement':
                return self._check_engagement_trigger(conditions)
            elif trigger_type == 'follower_count':
                return self._check_follower_trigger(conditions)
            elif trigger_type == 'hashtag':
                return self._check_hashtag_trigger(conditions)
            return False

        except Exception as e:
            logging.error(f"Error checking trigger conditions: {str(e)}")
            return False

    def _execute_actions(self, rule_id: int, actions: List[Dict]):
        """Execute automation rule actions."""
        try:
            for action in actions:
                action_type = action.get('type')
                params = action.get('params', {})

                if action_type == 'post_content':
                    self._execute_post_action(params)
                elif action_type == 'send_dm':
                    self._execute_dm_action(params)
                elif action_type == 'follow_users':
                    self._execute_follow_action(params)
                elif action_type == 'engage_content':
                    self._execute_engagement_action(params)

            # Update last run time
            conn = sqlite3.connect('social_automation.db')
            c = conn.cursor()
            c.execute('''
                UPDATE automation_rules 
                SET last_run = ? 
                WHERE id = ?
            ''', (datetime.now().isoformat(), rule_id))
            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"Error executing actions for rule {rule_id}: {str(e)}")

class CampaignManager:
    def __init__(self, automation_manager: AutomationManager):
        self.automation_manager = automation_manager

    def create_campaign(self, campaign: Campaign) -> Tuple[bool, str]:
        """Create a new marketing campaign."""
        try:
            conn = sqlite3.connect('social_automation.db')
            c = conn.cursor()
            
            c.execute('''
                INSERT INTO campaigns 
                (user_id, name, start_date, end_date, platform, budget,
                 target_audience, content_strategy, automation_rules, status, metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                campaign.user_id,
                campaign.name,
                campaign.start_date.isoformat(),
                campaign.end_date.isoformat(),
                campaign.platform,
                campaign.budget,
                json.dumps(campaign.target_audience),
                json.dumps(campaign.content_strategy),
                json.dumps(campaign.automation_rules),
                campaign.status,
                json.dumps(campaign.metrics)
            ))
            
            campaign_id = c.lastrowid
            conn.commit()
            conn.close()
            
            # Create associated automation rules
            self._setup_campaign_automation(campaign_id, campaign)
            
            return True, f"Campaign created successfully with ID: {campaign_id}"

        except Exception as e:
            logging.error(f"Error creating campaign: {str(e)}")
            return False, f"Failed to create campaign: {str(e)}"

    def _setup_campaign_automation(self, campaign_id: int, campaign: Campaign):
        """Set up automation rules for a campaign."""
        try:
            # Create content posting rules
            content_schedule = self._generate_content_schedule(campaign)
            for schedule_item in content_schedule:
                rule = AutomationRule(
                    id=None,
                    user_id=campaign.user_id,
                    name=f"Campaign {campaign_id} - Content Post",
                    trigger_type='schedule',
                    trigger_conditions={'schedule_time': schedule_item['time']},
                    actions=[{
                        'type': 'post_content',
                        'params': schedule_item['content']
                    }],
                    status='active',
                    priority=1,
                    created_at=datetime.now(),
                    last_run=None
                )
                self.automation_manager.create_automation_rule(rule)

            # Create engagement rules
            engagement_rule = AutomationRule(
                id=None,
                user_id=campaign.user_id,
                name=f"Campaign {campaign_id} - Engagement",
                trigger_type='engagement',
                trigger_conditions={
                    'min_engagement_rate': 0.02,
                    'hashtags': campaign.content_strategy.get('hashtags', [])
                },
                actions=[{
                    'type': 'engage_content',
                    'params': {
                        'action_types': ['like', 'comment'],
                        'limit': 50
                    }
                }],
                status='active',
                priority=2,
                created_at=datetime.now(),
                last_run=None
            )
            self.automation_manager.create_automation_rule(engagement_rule)

        except Exception as e:
            logging.error(f"Error setting up campaign automation: {str(e)}")

def render_automation_page():
    """Render the automation rules interface in Streamlit."""
    st.title("Automation Rules")
    
    # Initialize managers
    automation_manager = AutomationManager()

    # Create new rule form
    with st.form("create_rule_form"):
        st.subheader("Create New Automation Rule")
        
        rule_name = st.text_input("Rule Name")
        trigger_type = st.selectbox(
            "Trigger Type",
            ["schedule", "engagement", "follower_count", "hashtag"]
        )
        
        # Dynamic trigger conditions based on type
        trigger_conditions = {}
        if trigger_type == "schedule":
            schedule_time = st.time_input("Schedule Time")
            schedule_days = st.multiselect(
                "Schedule Days",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )
            trigger_conditions = {"time": schedule_time.strftime("%H:%M"), "days": schedule_days}
        elif trigger_type == "engagement":
            trigger_conditions["min_engagement_rate"] = st.number_input(
                "Minimum Engagement Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.02
            )
        
        # Action configuration
        action_type = st.selectbox(
            "Action Type",
            ["post_content", "send_dm", "follow_users", "engage_content"]
        )
        
        action_params = {}
        if action_type == "post_content":
            action_params["content_type"] = st.selectbox(
                "Content Type",
                ["image", "video", "story"]
            )
            action_params["caption_template"] = st.text_area("Caption Template")
        elif action_type == "send_dm":
            action_params["message_template"] = st.text_area("Message Template")
        
        priority = st.slider("Priority", 1, 10, 5)
        
        if st.form_submit_button("Create Rule"):
            rule = AutomationRule(
                id=None,
                user_id=st.session_state.user_id,
                name=rule_name,
                trigger_type=trigger_type,
                trigger_conditions=trigger_conditions,
                actions=[{"type": action_type, "params": action_params}],
                status="active",
                priority=priority,
                created_at=datetime.now(),
                last_run=None
            )
            
            success, message = automation_manager.create_automation_rule(rule)
            if success:
                st.success(message)
            else:
                st.error(message)

    # Display existing rules
    st.subheader("Existing Automation Rules")
    display_automation_rules()

def render_campaign_page():
    """Render the campaign management interface in Streamlit."""
    st.title("Campaign Management")
    
    # Initialize managers
    automation_manager = AutomationManager()
    campaign_manager = CampaignManager(automation_manager)

    # Create new campaign form
    with st.form("create_campaign_form"):
        st.subheader("Create New Campaign")
        
        campaign_name = st.text_input("Campaign Name")
        platform = st.selectbox("Platform", ["instagram", "twitter", "tiktok"])
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")
        
        budget = st.number_input("Budget", min_value=0.0, value=100.0)
        
        # Target audience configuration
        st.subheader("Target Audience")
        target_audience = {
            "age_range": st.slider("Age Range", 13, 65, (18, 35)),
            "locations": st.multiselect("Target Locations", ["US", "UK", "CA", "AU"]),
            "interests": st.multiselect("Interests", ["fashion", "tech", "food", "travel"])
        }
        
        # Content strategy configuration
        st.subheader("Content Strategy")
        content_strategy = {
            "post_frequency": st.selectbox("Post Frequency", ["daily", "weekly", "custom"]),
            "content_types": st.multiselect("Content Types", ["image", "video", "story"]),
            "hashtags": st.text_input("Hashtags (comma-separated)")
        }
        
        if st.form_submit_button("Create Campaign"):
            campaign = Campaign(
                id=None,
                user_id=st.session_state.user_id,
                name=campaign_name,
                start_date=datetime.combine(start_date, datetime.min.time()),
                end_date=datetime.combine(end_date, datetime.max.time()),
                platform=platform,
                budget=budget,
                target_audience=target_audience,
                content_strategy=content_strategy,
                automation_rules=[],
                status="active",
                metrics={}
            )
            
            success, message = campaign_manager.create_campaign(campaign)
            if success:
                st.success(message)
            else:
                st.error(message)

    # Display existing campaigns
    st.subheader("Active Campaigns")
    display_campaigns()

def display_automation_rules():
    """Display existing automation rules in a table."""
    conn = sqlite3.connect('social_automation.db')
    rules_df = pd.read_sql_query('''
        SELECT id, name, trigger_type, status, priority, last_run 
        FROM automation_rules 
        WHERE user_id = ? 
        ORDER BY priority DESC
    ''', conn, params=(st.session_state.user_id,))
    conn.close()

    if not rules_df.empty:
        st.dataframe(rules_df)
    else:
        st.info("No automation rules found")

def display_campaigns():
    """Display existing campaigns in a table."""
    conn = sqlite3.connect('social_automation.db')
    campaigns_df = pd.read_sql_query('''
        SELECT id, name, platform, start_date, end_date, status, budget 
        FROM campaigns 
        WHERE user_id = ? 
        ORDER BY start_date DESC
    ''', conn, params=(st.session_state.user_id,))
    conn.close()

    if not campaigns_df.empty:
        st.dataframe(campaigns_df)
    else:
        st.info("No active campaigns found")