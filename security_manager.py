# security_manager.py
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import jwt
import bcrypt
import logging
import requests
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import tensorflow as tf
import pandas as pd
from sklearn.ensemble import IsolationForest
from collections import deque
import re
import time
import hashlib
from urllib.parse import urlparse
import ipaddress
import threading
from ratelimit import limits, sleep_and_retry
import httpx
from user_agents import parse
import os

class SecurityManager:
    def __init__(self):
        self.setup_database()
        self.load_security_config()
        self.initialize_ml_models()
        self.setup_rate_limiters()
        
        # Initialize request tracking
        self.request_history = {}
        self.ip_blacklist = set()
        self.suspicious_patterns = self.load_suspicious_patterns()
        
        # Initialize session tracking
        self.active_sessions = {}
        self.failed_login_attempts = {}
        
        # JWT configuration
        self.jwt_secret = os.getenv('JWT_SECRET_KEY')
        self.jwt_algorithm = 'HS256'

    def load_suspicious_patterns(self) -> Dict[str, List[str]]:
        """Load suspicious patterns for security monitoring."""
        try:
            # Default patterns if config file doesn't exist
            default_patterns = {
                'request_patterns': [
                    r'(?i)(union\s+select|select\s+.*\s+from|drop\s+table)',  # SQL injection
                    r'(?i)(<script>|javascript:)',  # XSS attempts
                    r'(?i)(\.\.\/|\.\.\\)',  # Path traversal
                    r'(?i)(\$\{|\$\{.*?\}|\#{|\#{.*?\})',  # Template injection
                    r'(?i)(eval\(|exec\(|system\()'  # Code injection
                ],
                'user_agent_patterns': [
                    r'(?i)(curl|wget|python-requests|postman)',  # Tool signatures
                    r'(?i)(sqlmap|nikto|burp|acunetix)',  # Security tool signatures
                    r'(?i)(bot|crawler|spider)',  # Bot signatures
                    r'(?i)(\\x[0-9a-f]{2}|%[0-9a-f]{2})'  # Encoded content
                ],
                'ip_patterns': [
                    r'^(?:10|127|172\.(?:1[6-9]|2[0-9]|3[01])|192\.168)\.',  # Internal IPs
                    r'(?i)(tor|proxy|vpn)',  # Proxy/VPN signatures
                ],
                'content_patterns': [
                    r'(?i)(password|passwd|pwd|secret|key)=',  # Sensitive data
                    r'(?i)(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)',  # Email addresses
                    r'(?i)(\b\d{3}-\d{2}-\d{4}\b)',  # SSN format
                    r'(?i)(\b\d{16}\b)',  # Credit card format
                ]
            }

            # Try to load custom patterns from configuration file
            config_path = Path('config/security_patterns.json')
            if config_path.exists():
                with open(config_path, 'r') as f:
                    custom_patterns = json.load(f)
                    # Merge custom patterns with defaults, preferring custom
                    for category, patterns in custom_patterns.items():
                        if category in default_patterns:
                            default_patterns[category].extend(patterns)
                        else:
                            default_patterns[category] = patterns

            # Compile regex patterns for efficiency
            compiled_patterns = {}
            for category, patterns in default_patterns.items():
                compiled_patterns[category] = [re.compile(pattern) for pattern in patterns]

            logging.info(f"Loaded {sum(len(p) for p in compiled_patterns.values())} security patterns")
            return compiled_patterns

        except Exception as e:
            logging.error(f"Error loading suspicious patterns: {str(e)}")
            return {}
        
    def load_security_config(self):
        """Load security configuration settings."""
        self.config = {
            # Authentication settings
            'max_login_attempts': 5,
            'lockout_duration': 300,  # 5 minutes
            'session_timeout': 3600,  # 1 hour
            'password_min_length': 12,
            'require_2fa': True,
            'jwt_expiry': 86400,  # 24 hours
            
            # Request filtering
            'ip_blacklist_duration': 86400,  # 24 hours
            'suspicious_patterns': [
                r'(?i)(union\s+select|drop\s+table|delete\s+from)',  # SQL injection
                r'<script.*?>.*?</script>',  # XSS
                r'../|\.\.\\',  # Path traversal
                r'\{\{.*\}\}',  # Template injection
                r'eval\(.*\)',  # Code injection
            ],
            
            # File upload settings
            'allowed_file_types': ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mov'],
            'max_file_size': 50 * 1024 * 1024,  # 50MB
            
            # Rate limiting
            'rate_limit_windows': {
                'default': 3600,  # 1 hour
                'login': 300,     # 5 minutes
                'api': 60,        # 1 minute
            },
            'rate_limit_counts': {
                'default': 1000,  # requests per window
                'login': 5,       # attempts per window
                'api': 100,       # requests per window
            },
            
            # ML Security settings
            'anomaly_detection_threshold': 0.8,
            'bot_detection_confidence': 0.7,
            'behavior_analysis_window': 24 * 3600,  # 24 hours
            
            # Logging and monitoring
            'log_level': 'INFO',
            'alert_threshold': 0.7,
            'monitoring_interval': 300,  # 5 minutes
            
            # API Security
            'required_headers': ['X-API-Key', 'User-Agent'],
            'allowed_origins': ['https://example.com'],
            'max_payload_size': 1024 * 1024,  # 1MB
        }
        
        # Load environment-specific overrides
        env_config = os.getenv('SECURITY_CONFIG')
        if env_config:
            try:
                env_overrides = json.loads(env_config)
                self.config.update(env_overrides)
            except json.JSONDecodeError:
                logging.error("Failed to parse SECURITY_CONFIG environment variable")

    def setup_rate_limiters(self):
        """Initialize rate limiters for different endpoints and actions."""
        self.rate_limiters = {}
        
        # Setup rate limiters for each endpoint type
        for endpoint_type, window in self.config['rate_limit_windows'].items():
            self.rate_limiters[endpoint_type] = {
                'window': window,
                'max_requests': self.config['rate_limit_counts'][endpoint_type],
                'requests': {},  # {ip: [(timestamp, count)]}
                'blocked_until': {},  # {ip: timestamp}
            }
        
        # Start cleanup thread for rate limiters
        self._start_rate_limiter_cleanup()
        
    def check_suspicious_patterns(self, request_data: Dict) -> bool:
        """Check for suspicious patterns in request data."""
        try:
            if not self.suspicious_patterns:
                return False

            # Check request patterns
            path = request_data.get('path', '')
            method = request_data.get('method', '')
            request_str = f"{method} {path}"
            for pattern in self.suspicious_patterns.get('request_patterns', []):
                if pattern.search(request_str):
                    logging.warning(f"Suspicious request pattern detected: {request_str}")
                    return True

            # Check user agent patterns
            user_agent = request_data.get('user_agent', '')
            for pattern in self.suspicious_patterns.get('user_agent_patterns', []):
                if pattern.search(user_agent):
                    logging.warning(f"Suspicious user agent detected: {user_agent}")
                    return True

            # Check IP patterns
            ip = request_data.get('ip_address', '')
            for pattern in self.suspicious_patterns.get('ip_patterns', []):
                if pattern.search(ip):
                    logging.warning(f"Suspicious IP pattern detected: {ip}")
                    return True

            # Check content patterns in body or parameters
            content = json.dumps(request_data.get('body', {}))
            for pattern in self.suspicious_patterns.get('content_patterns', []):
                if pattern.search(content):
                    logging.warning(f"Suspicious content pattern detected in request")
                    return True

            return False

        except Exception as e:
            logging.error(f"Pattern check error: {str(e)}")
            return False

    def _start_rate_limiter_cleanup(self):
        """Start a background thread to clean up expired rate limit records."""
        def cleanup_task():
            while True:
                try:
                    current_time = time.time()
                    for endpoint_type, limiter in self.rate_limiters.items():
                        window = limiter['window']
                        cutoff_time = current_time - window
                        
                        # Clean up expired request records
                        for ip in list(limiter['requests'].keys()):
                            limiter['requests'][ip] = [
                                (ts, count) for ts, count in limiter['requests'][ip]
                                if ts > cutoff_time
                            ]
                            if not limiter['requests'][ip]:
                                del limiter['requests'][ip]
                        
                        # Clean up expired blocks
                        for ip in list(limiter['blocked_until'].keys()):
                            if limiter['blocked_until'][ip] <= current_time:
                                del limiter['blocked_until'][ip]
                                
                except Exception as e:
                    logging.error(f"Rate limiter cleanup error: {str(e)}")
                
                # Run cleanup every minute
                time.sleep(60)
        
        cleanup_thread = threading.Thread(
            target=cleanup_task,
            daemon=True,
            name="RateLimiterCleanup"
        )
        cleanup_thread.start()

    def setup_database(self):
        """Initialize security-related database tables."""
        conn = sqlite3.connect('social_automation.db')
        c = conn.cursor()
        
        # Security events table
        c.execute('''
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                event_type TEXT,
                event_details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                timestamp TIMESTAMP,
                risk_score FLOAT,
                action_taken TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # IP blacklist table
        c.execute('''
            CREATE TABLE IF NOT EXISTS ip_blacklist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip_address TEXT UNIQUE,
                reason TEXT,
                added_at TIMESTAMP,
                expires_at TIMESTAMP
            )
        ''')
        
        # User behavior patterns table
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_behavior_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                pattern_type TEXT,
                pattern_data TEXT,
                last_updated TIMESTAMP,
                confidence_score FLOAT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Security audit log
        c.execute('''
            CREATE TABLE IF NOT EXISTS security_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                event_type TEXT,
                description TEXT,
                severity TEXT,
                source_ip TEXT,
                user_agent TEXT,
                additional_data TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def initialize_ml_models(self):
        """Initialize machine learning models for behavior analysis."""
        # Isolation Forest for anomaly detection
        self.anomaly_detector = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        
        # Load pre-trained bot detection model
        try:
            self.bot_detection_model = tf.keras.models.load_model('models/bot_detection.h5')
        except:
            logging.warning("Bot detection model not found. Using fallback detection methods.")
            self.bot_detection_model = None

    @dataclass
    class SecurityEvent:
        """Data class for security events."""
        user_id: Optional[int]
        event_type: str
        event_details: Dict
        ip_address: str
        user_agent: str
        timestamp: datetime
        risk_score: float
        action_taken: str

    def analyze_request(self, request_data: Dict) -> Tuple[bool, float, str]:
        """Analyze incoming request for potential security threats."""
        try:
            risk_score = 0.0
            reasons = []

            # Extract request information
            ip = request_data.get('ip_address')
            user_agent = request_data.get('user_agent')
            path = request_data.get('path')
            method = request_data.get('method')
            headers = request_data.get('headers', {})
            
            # Check IP blacklist
            if self.is_ip_blacklisted(ip):
                return False, 1.0, "IP is blacklisted"

            # Check rate limits
            if self.is_rate_limited(ip, path):
                risk_score += 0.3
                reasons.append("Rate limit exceeded")

            # Analyze user agent
            ua_risk = self.analyze_user_agent(user_agent)
            risk_score += ua_risk
            if ua_risk > 0.5:
                reasons.append("Suspicious user agent")

            # Check for suspicious patterns
            if self.check_suspicious_patterns(request_data):
                risk_score += 0.4
                reasons.append("Suspicious request patterns")

            # Behavioral analysis
            if 'user_id' in request_data:
                behavior_risk = self.analyze_user_behavior(request_data['user_id'])
                risk_score += behavior_risk
                if behavior_risk > 0.3:
                    reasons.append("Unusual behavior pattern")

            # Machine learning analysis
            ml_risk = self.ml_risk_assessment(request_data)
            risk_score += ml_risk
            if ml_risk > 0.4:
                reasons.append("ML risk detection")

            # Normalize risk score
            risk_score = min(risk_score, 1.0)

            # Log security event
            self.log_security_event(SecurityEvent(
                user_id=request_data.get('user_id'),
                event_type='request_analysis',
                event_details={'risk_score': risk_score, 'reasons': reasons},
                ip_address=ip,
                user_agent=user_agent,
                timestamp=datetime.now(),
                risk_score=risk_score,
                action_taken='monitored' if risk_score < 0.7 else 'blocked'
            ))

            return risk_score < 0.7, risk_score, ", ".join(reasons)

        except Exception as e:
            logging.error(f"Error in request analysis: {str(e)}")
            return False, 1.0, "Analysis error"

    def analyze_user_agent(self, user_agent: str) -> float:
        """Analyze user agent string for bot-like characteristics."""
        try:
            if not user_agent:
                return 0.8

            ua = parse(user_agent)
            risk_score = 0.0

            # Check for known bot signatures
            if any(bot_sign in user_agent.lower() 
                  for bot_sign in ['bot', 'crawler', 'spider']):
                return 0.9

            # Check for suspicious browser/OS combinations
            if ua.browser.family == 'Other' or ua.os.family == 'Other':
                risk_score += 0.3

            # Check for missing or suspicious headers
            if not ua.is_mobile and not ua.is_pc:
                risk_score += 0.2

            # Additional checks for automation tools
            if any(tool in user_agent.lower() 
                  for tool in ['selenium', 'puppeteer', 'phantomjs']):
                risk_score += 0.5

            return min(risk_score, 1.0)

        except Exception as e:
            logging.error(f"User agent analysis error: {str(e)}")
            return 0.5

    def check_suspicious_patterns(self, request_data: Dict) -> bool:
        """Check for suspicious request patterns."""
        try:
            ip = request_data.get('ip_address')
            
            # Initialize IP history if not exists
            if ip not in self.request_history:
                self.request_history[ip] = deque(maxlen=100)
            
            # Add current request to history
            self.request_history[ip].append({
                'timestamp': datetime.now(),
                'path': request_data.get('path'),
                'method': request_data.get('method')
            })
            
            history = self.request_history[ip]
            
            # Check request frequency
            if len(history) >= 10:
                time_diff = (history[-1]['timestamp'] - history[0]['timestamp']).total_seconds()
                if time_diff < 1:  # More than 10 requests per second
                    return True
            
            # Check pattern repetition
            if len(history) >= 20:
                pattern = [f"{r['method']}:{r['path']}" for r in list(history)[-20:]]
                if len(set(pattern)) < 5:  # Less than 5 unique requests in last 20
                    return True
            
            return False

        except Exception as e:
            logging.error(f"Pattern check error: {str(e)}")
            return False

    @sleep_and_retry
    @limits(calls=100, period=60)
    def analyze_user_behavior(self, user_id: int) -> float:
        """Analyze user behavior for suspicious patterns."""
        try:
            conn = sqlite3.connect('social_automation.db')
            c = conn.cursor()
            
            # Get recent user actions
            c.execute('''
                SELECT event_type, timestamp 
                FROM security_events 
                WHERE user_id = ? 
                AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 100
            ''', (user_id, datetime.now() - timedelta(hours=24)))
            
            actions = c.fetchall()
            conn.close()

            if not actions:
                return 0.0

            # Convert to DataFrame for analysis
            df = pd.DataFrame(actions, columns=['event_type', 'timestamp'])
            
            risk_score = 0.0
            
            # Check action frequency
            action_counts = df['event_type'].value_counts()
            if len(action_counts) > 0:
                max_frequency = action_counts.max()
                if max_frequency > 50:  # More than 50 same actions
                    risk_score += 0.3
            
            # Check time patterns
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            hour_counts = df['hour'].value_counts()
            
            if len(hour_counts) == 1:  # All actions in same hour
                risk_score += 0.4
            
            # Use anomaly detection
            if len(df) > 10:
                features = pd.get_dummies(df['event_type'])
                anomaly_scores = self.anomaly_detector.fit_predict(features)
                if sum(anomaly_scores == -1) / len(anomaly_scores) > 0.2:
                    risk_score += 0.3

            return min(risk_score, 1.0)

        except Exception as e:
            logging.error(f"Behavior analysis error: {str(e)}")
            return 0.0

    def ml_risk_assessment(self, request_data: Dict) -> float:
        """Perform machine learning based risk assessment."""
        try:
            if not self.bot_detection_model:
                return 0.0

            # Prepare features for ML model
            features = self.extract_ml_features(request_data)
            
            # Make prediction
            prediction = self.bot_detection_model.predict(
                np.array([features])
            )[0]

            return float(prediction[0])

        except Exception as e:
            logging.error(f"ML risk assessment error: {str(e)}")
            return 0.0

    def extract_ml_features(self, request_data: Dict) -> List[float]:
        """Extract features for ML model."""
        features = []
        
        try:
            # Request timing features
            hour = datetime.now().hour
            features.append(hour / 24.0)
            
            # Request pattern features
            ip = request_data.get('ip_address')
            if ip in self.request_history:
                recent_requests = len(self.request_history[ip])
                features.append(min(recent_requests / 100.0, 1.0))
            else:
                features.append(0.0)
            
            # User agent features
            ua_risk = self.analyze_user_agent(request_data.get('user_agent', ''))
            features.append(ua_risk)
            
            # Add more features as needed...
            
            return features

        except Exception as e:
            logging.error(f"Feature extraction error: {str(e)}")
            return [0.0] * 3  # Return default features

    def log_security_event(self, event: SecurityEvent):
        """Log security event to database."""
        try:
            conn = sqlite3.connect('social_automation.db')
            c = conn.cursor()
            
            c.execute('''
                INSERT INTO security_events 
                (user_id, event_type, event_details, ip_address, 
                 user_agent, timestamp, risk_score, action_taken)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.user_id,
                event.event_type,
                json.dumps(event.event_details),
                event.ip_address,
                event.user_agent,
                event.timestamp,
                event.risk_score,
                event.action_taken
            ))
            
            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"Error logging security event: {str(e)}")

def setup_security_middleware():
    """Set up security middleware for Streamlit."""
    def security_middleware():
        if 'security_manager' not in st.session_state:
            st.session_state.security_manager = SecurityManager()

        # Get request information
        request_data = {
            'ip_address': get_client_ip(),
            'user_agent': st.request.headers.get('User-Agent'),
            'path': st.request.path,
            'method': st.request.method,
            'headers': dict(st.request.headers),
            'user_id': st.session_state.get('user_id')
        }

        # Analyze request
        allowed, risk_score, reasons = st.session_state.security_manager.analyze_request(
            request_data
        )

        if not allowed:
            st.error("Access denied: Suspicious activity detected")
            st.stop()

        # Add security headers
        st.set_page_config(
            page_title="Secure App",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': None
            }
        )

    return security_middleware

def get_client_ip():
    """Get client IP address."""
    try:
        return httpx.get('https://api.ipify.org').text
    except:
        return "unknown"

@dataclass
class SecurityEvent:
    """Data class for security events with validation and serialization."""
    user_id: Optional[int]
    event_type: str
    event_details: Dict
    ip_address: str
    user_agent: str
    timestamp: datetime
    risk_score: float
    action_taken: str

    def __post_init__(self):
        """Validate security event data after initialization."""
        # Validate event_type
        if not isinstance(self.event_type, str):
            raise ValueError("event_type must be a string")

        # Validate event_details
        if not isinstance(self.event_details, dict):
            raise ValueError("event_details must be a dictionary")

        # Validate IP address format
        if not isinstance(self.ip_address, str) or not self.ip_address:
            raise ValueError("Invalid IP address")

        # Validate risk_score range
        if not 0 <= self.risk_score <= 1:
            raise ValueError("risk_score must be between 0 and 1")

        # Validate action_taken
        if self.action_taken not in ['monitored', 'blocked', 'flagged', 'allowed']:
            raise ValueError("Invalid action_taken value")

    def to_dict(self) -> Dict:
        """Convert security event to dictionary format."""
        return {
            'user_id': self.user_id,
            'event_type': self.event_type,
            'event_details': self.event_details,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'timestamp': self.timestamp.isoformat(),
            'risk_score': self.risk_score,
            'action_taken': self.action_taken
        }

    def to_json(self) -> str:
        """Convert security event to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> 'SecurityEvent':
        """Create SecurityEvent instance from dictionary."""
        # Convert timestamp string to datetime
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'SecurityEvent':
        """Create SecurityEvent instance from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

# Initialize security features
def init_security():
    st.set_page_config(
        page_title="Secure Social Media Automation",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add security middleware
    security_middleware = setup_security_middleware()
    st.experimental_set_query_params = security_middleware

    # Initialize security manager
    if 'security_manager' not in st.session_state:
        st.session_state.security_manager = SecurityManager()

# Usage in main app
if __name__ == "__main__":
    init_security()