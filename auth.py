# auth.py
import streamlit as st
import sqlite3
import hashlib
import jwt
import datetime
from typing import Optional, Tuple, Dict
import logging
from email_validator import validate_email, EmailNotValidError
import pyotp
import qrcode
from pathlib import Path
import os

# Security configuration
class SecurityConfig:
    JWT_SECRET = "your-secret-key"  # In production, use environment variable
    JWT_EXPIRATION = 24  # hours
    SALT_LENGTH = 16
    MIN_PASSWORD_LENGTH = 8
    MAX_LOGIN_ATTEMPTS = 3
    LOCKOUT_TIME = 15  # minutes

class AuthManager:
    def __init__(self):
        self.failed_attempts = {}
        self.locked_accounts = {}

    def generate_password_hash(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Generate a secure password hash with salt."""
        if not salt:
            salt = os.urandom(SecurityConfig.SALT_LENGTH).hex()
        hash_obj = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return hash_obj.hex(), salt

    def verify_password(self, stored_hash: str, stored_salt: str, provided_password: str) -> bool:
        """Verify a password against its hash."""
        generated_hash, _ = self.generate_password_hash(provided_password, stored_salt)
        return stored_hash == generated_hash

    def generate_jwt_token(self, user_id: int, username: str) -> str:
        """Generate a JWT token for authenticated users."""
        payload = {
            'user_id': user_id,
            'username': username,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=SecurityConfig.JWT_EXPIRATION)
        }
        return jwt.encode(payload, SecurityConfig.JWT_SECRET, algorithm='HS256')

    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verify a JWT token and return payload if valid."""
        try:
            return jwt.decode(token, SecurityConfig.JWT_SECRET, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            logging.warning("Expired JWT token")
            return None
        except jwt.InvalidTokenError:
            logging.warning("Invalid JWT token")
            return None

class UserManager:
    def __init__(self, auth_manager: AuthManager):
        self.auth_manager = auth_manager
        self.setup_database()

    def setup_database(self):
        """Initialize the users table and related tables."""
        conn = sqlite3.connect('social_automation.db')
        c = conn.cursor()
        
        # Add 2FA and security-related columns
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                two_factor_secret TEXT,
                two_factor_enabled BOOLEAN DEFAULT 0,
                last_login TIMESTAMP,
                failed_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP,
                subscription_plan TEXT,
                subscription_status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()

    def register_user(self, username: str, password: str, email: str) -> Tuple[bool, str]:
        """Register a new user with validation."""
        try:
            # Validate email
            valid = validate_email(email)
            email = valid.email

            # Validate password strength
            if len(password) < SecurityConfig.MIN_PASSWORD_LENGTH:
                return False, "Password must be at least 8 characters long"

            if not any(c.isupper() for c in password):
                return False, "Password must contain at least one uppercase letter"

            if not any(c.isdigit() for c in password):
                return False, "Password must contain at least one number"

            # Generate password hash and salt
            password_hash, salt = self.auth_manager.generate_password_hash(password)

            conn = sqlite3.connect('social_automation.db')
            c = conn.cursor()
            
            c.execute('''
                INSERT INTO users (username, password_hash, salt, email)
                VALUES (?, ?, ?, ?)
            ''', (username, password_hash, salt, email))
            
            conn.commit()
            conn.close()
            
            return True, "Registration successful"

        except sqlite3.IntegrityError:
            return False, "Username or email already exists"
        except EmailNotValidError:
            return False, "Invalid email address"
        except Exception as e:
            logging.error(f"Registration error: {str(e)}")
            return False, "Registration failed"

    def login_user(self, username: str, password: str, two_factor_code: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
        """Handle user login with 2FA and security measures."""
        try:
            conn = sqlite3.connect('social_automation.db')
            c = conn.cursor()

            # Check if account is locked
            c.execute('''
                SELECT id, password_hash, salt, two_factor_enabled, two_factor_secret, 
                       failed_attempts, locked_until
                FROM users WHERE username = ?
            ''', (username,))
            
            user_data = c.fetchone()
            
            if not user_data:
                return False, "Invalid credentials", None

            user_id, stored_hash, salt, two_factor_enabled, two_factor_secret, failed_attempts, locked_until = user_data

            # Check account lockout
            if locked_until and datetime.datetime.strptime(locked_until, '%Y-%m-%d %H:%M:%S') > datetime.datetime.utcnow():
                return False, "Account is temporarily locked", None

            # Verify password
            if not self.auth_manager.verify_password(stored_hash, salt, password):
                self._handle_failed_login(c, conn, user_id, failed_attempts)
                return False, "Invalid credentials", None

            # Handle 2FA if enabled
            if two_factor_enabled:
                if not two_factor_code:
                    return False, "2FA code required", None
                
                totp = pyotp.TOTP(two_factor_secret)
                if not totp.verify(two_factor_code):
                    self._handle_failed_login(c, conn, user_id, failed_attempts)
                    return False, "Invalid 2FA code", None

            # Reset failed attempts on successful login
            c.execute('''
                UPDATE users 
                SET failed_attempts = 0, 
                    locked_until = NULL,
                    last_login = CURRENT_TIMESTAMP 
                WHERE id = ?
            ''', (user_id,))
            
            conn.commit()
            conn.close()

            # Generate JWT token
            token = self.auth_manager.generate_jwt_token(user_id, username)
            return True, "Login successful", token

        except Exception as e:
            logging.error(f"Login error: {str(e)}")
            return False, "Login failed", None

    def _handle_failed_login(self, cursor, conn, user_id: int, failed_attempts: int):
        """Handle failed login attempts and account lockout."""
        new_attempts = failed_attempts + 1
        locked_until = None

        if new_attempts >= SecurityConfig.MAX_LOGIN_ATTEMPTS:
            locked_until = (datetime.datetime.utcnow() + 
                          datetime.timedelta(minutes=SecurityConfig.LOCKOUT_TIME))

        cursor.execute('''
            UPDATE users 
            SET failed_attempts = ?,
                locked_until = ?
            WHERE id = ?
        ''', (new_attempts, locked_until, user_id))
        
        conn.commit()

    def setup_2fa(self, user_id: int) -> Tuple[bool, str, Optional[str]]:
        """Set up 2FA for a user."""
        try:
            secret = pyotp.random_base32()
            totp = pyotp.TOTP(secret)
            
            # Generate QR code
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(totp.provisioning_uri(name="SocialAutomation"))
            qr.make(fit=True)
            
            # Save QR code image
            img_path = Path(f"temp/2fa_qr_{user_id}.png")
            img = qr.make_image(fill_color="black", back_color="white")
            img.save(img_path)

            conn = sqlite3.connect('social_automation.db')
            c = conn.cursor()
            
            c.execute('''
                UPDATE users 
                SET two_factor_secret = ?,
                    two_factor_enabled = 1
                WHERE id = ?
            ''', (secret, user_id))
            
            conn.commit()
            conn.close()

            return True, "2FA setup successful", str(img_path)

        except Exception as e:
            logging.error(f"2FA setup error: {str(e)}")
            return False, "2FA setup failed", None

def render_login_page():
    """Render the login page in Streamlit."""
    st.title("Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        two_factor_code = st.text_input("2FA Code (if enabled)", "")
        
        submit = st.form_submit_button("Login")
        
        if submit:
            auth_manager = AuthManager()
            user_manager = UserManager(auth_manager)
            success, message, token = user_manager.login_user(username, password, two_factor_code)
            
            if success:
                st.session_state.authenticated = True
                st.session_state.token = token
                st.success(message)
                st.experimental_rerun()
            else:
                st.error(message)

def render_registration_page():
    """Render the registration page in Streamlit."""
    st.title("Register")
    
    with st.form("registration_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        password_confirm = st.text_input("Confirm Password", type="password")
        
        submit = st.form_submit_button("Register")
        
        if submit:
            if password != password_confirm:
                st.error("Passwords do not match")
                return

            auth_manager = AuthManager()
            user_manager = UserManager(auth_manager)
            success, message = user_manager.register_user(username, password, email)
            
            if success:
                st.success(message)
                st.info("Please proceed to login")
            else:
                st.error(message)