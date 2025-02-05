# subscription_manager.py
import streamlit as st
import stripe
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Optional, Tuple
import logging
import json
from dataclasses import dataclass
from enum import Enum
import hmac
import hashlib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import os

# Configure Stripe
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET')

class SubscriptionTier(Enum):
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"

@dataclass
class SubscriptionPlan:
    """Data class for subscription plans."""
    id: str
    name: str
    price: float
    billing_interval: str
    features: List[str]
    limits: Dict[str, int]
    stripe_price_id: str

class SubscriptionManager:
    def __init__(self):
        self.setup_database()
        self.load_subscription_plans()
        self.setup_stripe_products()

    def setup_database(self):
        """Initialize subscription-related database tables."""
        conn = sqlite3.connect('social_automation.db')
        c = conn.cursor()
        
        # Subscriptions table
        c.execute('''
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE,
                stripe_customer_id TEXT,
                stripe_subscription_id TEXT,
                plan_id TEXT,
                status TEXT,
                current_period_start TIMESTAMP,
                current_period_end TIMESTAMP,
                cancel_at_period_end BOOLEAN,
                payment_method_id TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Payment history table
        c.execute('''
            CREATE TABLE IF NOT EXISTS payment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                amount REAL,
                currency TEXT,
                stripe_payment_id TEXT,
                status TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Usage tracking table
        c.execute('''
            CREATE TABLE IF NOT EXISTS usage_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                feature TEXT,
                usage_count INTEGER,
                reset_date TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def load_subscription_plans(self):
        """Define subscription plans and their features."""
        self.subscription_plans = {
            SubscriptionTier.BASIC.value: SubscriptionPlan(
                id="basic_monthly",
                name="Basic Plan",
                price=10.00,
                billing_interval="month",
                features=[
                    "Video Downloads",
                    "Basic Analytics",
                    "Manual Posting"
                ],
                limits={
                    "video_downloads": 50,
                    "scheduled_posts": 10,
                    "auto_dm": 0
                },
                stripe_price_id="price_basic_monthly"
            ),
            SubscriptionTier.PRO.value: SubscriptionPlan(
                id="pro_monthly",
                name="Pro Plan",
                price=25.00,
                billing_interval="month",
                features=[
                    "Unlimited Video Downloads",
                    "Advanced Analytics",
                    "Post Scheduler",
                    "Auto DM",
                    "Basic Automation"
                ],
                limits={
                    "video_downloads": -1,  # unlimited
                    "scheduled_posts": 100,
                    "auto_dm": 1000
                },
                stripe_price_id="price_pro_monthly"
            ),
            SubscriptionTier.ENTERPRISE.value: SubscriptionPlan(
                id="enterprise_monthly",
                name="Enterprise Plan",
                price=50.00,
                billing_interval="month",
                features=[
                    "All Pro Features",
                    "Priority Support",
                    "White Label Option",
                    "Advanced Automation",
                    "API Access"
                ],
                limits={
                    "video_downloads": -1,
                    "scheduled_posts": -1,
                    "auto_dm": -1
                },
                stripe_price_id="price_enterprise_monthly"
            )
        }

    def setup_stripe_products(self):
        """Ensure Stripe products and prices are set up."""
        try:
            for plan in self.subscription_plans.values():
                # Create or update product
                product = stripe.Product.create(
                    name=plan.name,
                    description=", ".join(plan.features)
                )

                # Create or update price
                price = stripe.Price.create(
                    product=product.id,
                    unit_amount=int(plan.price * 100),  # Convert to cents
                    currency="usd",
                    recurring={"interval": plan.billing_interval}
                )

                plan.stripe_price_id = price.id

        except stripe.error.StripeError as e:
            logging.error(f"Stripe setup error: {str(e)}")

    async def create_subscription(self, user_id: int, plan_id: str, 
                                payment_method_id: str) -> Tuple[bool, str]:
        """Create a new subscription for a user."""
        try:
            conn = sqlite3.connect('social_automation.db')
            c = conn.cursor()

            # Check if user already has a subscription
            c.execute('SELECT stripe_customer_id FROM subscriptions WHERE user_id = ?', 
                     (user_id,))
            result = c.fetchone()

            if result:
                stripe_customer_id = result[0]
            else:
                # Get user details
                c.execute('SELECT email, username FROM users WHERE id = ?', (user_id,))
                user_email, username = c.fetchone()

                # Create Stripe customer
                customer = stripe.Customer.create(
                    email=user_email,
                    payment_method=payment_method_id,
                    invoice_settings={
                        'default_payment_method': payment_method_id
                    }
                )
                stripe_customer_id = customer.id

            # Create subscription
            subscription = stripe.Subscription.create(
                customer=stripe_customer_id,
                items=[{
                    'price': self.subscription_plans[plan_id].stripe_price_id,
                }],
                expand=['latest_invoice.payment_intent']
            )

            # Store subscription details
            c.execute('''
                INSERT OR REPLACE INTO subscriptions 
                (user_id, stripe_customer_id, stripe_subscription_id, plan_id, 
                 status, current_period_start, current_period_end, 
                 cancel_at_period_end, payment_method_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                stripe_customer_id,
                subscription.id,
                plan_id,
                subscription.status,
                datetime.fromtimestamp(subscription.current_period_start),
                datetime.fromtimestamp(subscription.current_period_end),
                subscription.cancel_at_period_end,
                payment_method_id
            ))

            conn.commit()
            conn.close()

            return True, "Subscription created successfully"

        except stripe.error.StripeError as e:
            logging.error(f"Stripe error: {str(e)}")
            return False, f"Payment failed: {str(e)}"
        except Exception as e:
            logging.error(f"Subscription creation error: {str(e)}")
            return False, f"Subscription creation failed: {str(e)}"

    def cancel_subscription(self, user_id: int) -> Tuple[bool, str]:
        """Cancel a user's subscription."""
        try:
            conn = sqlite3.connect('social_automation.db')
            c = conn.cursor()

            # Get subscription details
            c.execute('''
                SELECT stripe_subscription_id 
                FROM subscriptions 
                WHERE user_id = ?
            ''', (user_id,))
            result = c.fetchone()

            if not result:
                return False, "No active subscription found"

            stripe_subscription_id = result[0]

            # Cancel subscription in Stripe
            stripe.Subscription.modify(
                stripe_subscription_id,
                cancel_at_period_end=True
            )

            # Update local database
            c.execute('''
                UPDATE subscriptions 
                SET cancel_at_period_end = TRUE 
                WHERE user_id = ?
            ''', (user_id,))

            conn.commit()
            conn.close()

            return True, "Subscription cancelled successfully"

        except stripe.error.StripeError as e:
            logging.error(f"Stripe error: {str(e)}")
            return False, f"Cancellation failed: {str(e)}"

    def check_feature_access(self, user_id: int, feature: str) -> bool:
        """Check if user has access to a specific feature."""
        try:
            conn = sqlite3.connect('social_automation.db')
            c = conn.cursor()

            # Get user's subscription plan
            c.execute('''
                SELECT plan_id, status 
                FROM subscriptions 
                WHERE user_id = ? AND status = 'active'
            ''', (user_id,))
            result = c.fetchone()

            if not result:
                return False

            plan_id, status = result
            plan = self.subscription_plans[plan_id]

            # Check if feature is included in plan
            if feature in plan.features:
                # Check usage limits
                if feature in plan.limits:
                    limit = plan.limits[feature]
                    if limit == -1:  # unlimited
                        return True

                    # Check current usage
                    c.execute('''
                        SELECT usage_count 
                        FROM usage_tracking 
                        WHERE user_id = ? AND feature = ? AND reset_date > ?
                    ''', (user_id, feature, datetime.now()))
                    usage = c.fetchone()

                    return not usage or usage[0] < limit

            return False

        except Exception as e:
            logging.error(f"Feature access check error: {str(e)}")
            return False
        finally:
            conn.close()

    def track_feature_usage(self, user_id: int, feature: str):
        """Track usage of a specific feature."""
        try:
            conn = sqlite3.connect('social_automation.db')
            c = conn.cursor()

            # Update or insert usage tracking
            c.execute('''
                INSERT INTO usage_tracking (user_id, feature, usage_count, reset_date)
                VALUES (?, ?, 1, date('now', '+1 month'))
                ON CONFLICT (user_id, feature) DO UPDATE 
                SET usage_count = usage_count + 1
            ''', (user_id, feature))

            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"Usage tracking error: {str(e)}")

def render_subscription_page():
    """Render the subscription management interface in Streamlit."""
    st.title("Subscription Management")
    
    subscription_manager = SubscriptionManager()

    # Display current subscription status
    conn = sqlite3.connect('social_automation.db')
    c = conn.cursor()
    
    c.execute('''
        SELECT plan_id, status, current_period_end, cancel_at_period_end 
        FROM subscriptions 
        WHERE user_id = ?
    ''', (st.session_state.user_id,))
    current_subscription = c.fetchone()

    if current_subscription:
        plan_id, status, period_end, cancelling = current_subscription
        st.subheader("Current Subscription")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Plan: {subscription_manager.subscription_plans[plan_id].name}")
            st.write(f"Status: {status}")
        with col2:
            st.write(f"Renewal Date: {period_end}")
            if cancelling:
                st.warning("Subscription will cancel at end of period")

        # Display usage metrics
        st.subheader("Usage Statistics")
        display_usage_metrics(subscription_manager, st.session_state.user_id)

        # Cancel subscription button
        if not cancelling and st.button("Cancel Subscription"):
            success, message = subscription_manager.cancel_subscription(
                st.session_state.user_id
            )
            if success:
                st.success(message)
            else:
                st.error(message)

    else:
        st.subheader("Choose a Subscription Plan")
        
        # Display plan options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            display_plan_card(subscription_manager.subscription_plans["basic"])
        
        with col2:
            display_plan_card(subscription_manager.subscription_plans["pro"])
        
        with col3:
            display_plan_card(subscription_manager.subscription_plans["enterprise"])

def display_plan_card(plan: SubscriptionPlan):
    """Display a subscription plan card."""
    with st.container():
        st.markdown(f"### {plan.name}")
        st.write(f"${plan.price:.2f}/{plan.billing_interval}")
        
        st.write("Features:")
        for feature in plan.features:
            st.write(f"✓ {feature}")
        
        if st.button(f"Select {plan.name}"):
            handle_plan_selection(plan)

def handle_plan_selection(plan: SubscriptionPlan):
    """Handle subscription plan selection."""
    st.session_state.selected_plan = plan
    st.session_state.show_payment = True

def display_usage_metrics(subscription_manager: SubscriptionManager, user_id: int):
    """Display usage metrics for the current billing period."""
    conn = sqlite3.connect('social_automation.db')
    c = conn.cursor()
    
    c.execute('''
        SELECT feature, usage_count 
        FROM usage_tracking 
        WHERE user_id = ? AND reset_date > ?
    ''', (user_id, datetime.now()))
    
    usage_data = c.fetchall()
    conn.close()

    if usage_data:
        for feature, count in usage_data:
            limit = subscription_manager.subscription_plans[
                st.session_state.subscription_plan
            ].limits.get(feature, 0)
            
            if limit == -1:
                st.write(f"{feature}: {count} (Unlimited)")
            else:
                progress = count / limit
                st.progress(progress)
                st.write(f"{feature}: {count}/{limit}")
                
def handle_failed_payment(invoice):
    """Handle failed payment webhook event."""
    try:
        conn = sqlite3.connect('social_automation.db')
        c = conn.cursor()
        
        # Record failed payment
        c.execute('''
            INSERT INTO payment_history 
            (user_id, amount, currency, stripe_payment_id, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            invoice.customer,
            invoice.amount_due / 100.0,
            invoice.currency,
            invoice.payment_intent,
            'failed',
            datetime.fromtimestamp(invoice.created)
        ))
        
        # Update subscription status
        c.execute('''
            UPDATE subscriptions 
            SET status = 'past_due'
            WHERE stripe_customer_id = ?
        ''', (invoice.customer,))
        
        # Get user email
        c.execute('''
            SELECT email 
            FROM users 
            WHERE id IN (
                SELECT user_id 
                FROM subscriptions 
                WHERE stripe_customer_id = ?
            )
        ''', (invoice.customer,))
        email = c.fetchone()[0]
        
        conn.commit()
        conn.close()

        # Send failed payment email
        send_payment_failed_email(email, invoice)

    except Exception as e:
        logging.error(f"Failed payment handling error: {str(e)}")

def handle_subscription_cancelled(subscription):
    """Handle subscription cancellation webhook event."""
    try:
        conn = sqlite3.connect('social_automation.db')
        c = conn.cursor()
        
        # Update subscription status
        c.execute('''
            UPDATE subscriptions 
            SET status = 'cancelled',
                cancelled_at = ?,
                active_until = ?
            WHERE stripe_subscription_id = ?
        ''', (
            datetime.fromtimestamp(subscription.canceled_at),
            datetime.fromtimestamp(subscription.current_period_end),
            subscription.id
        ))
        
        # Get user email
        c.execute('''
            SELECT email 
            FROM users 
            WHERE id IN (
                SELECT user_id 
                FROM subscriptions 
                WHERE stripe_subscription_id = ?
            )
        ''', (subscription.id,))
        email = c.fetchone()[0]
        
        conn.commit()
        conn.close()

        # Send cancellation email
        send_subscription_cancelled_email(email, subscription)

    except Exception as e:
        logging.error(f"Subscription cancellation handling error: {str(e)}")

def send_payment_failed_email(email: str, invoice):
    """Send payment failed notification email."""
    try:
        msg = MIMEMultipart()
        msg['Subject'] = 'Payment Failed'
        msg['From'] = 'noreply@yourdomain.com'
        msg['To'] = email

        body = f"""
        Your payment has failed.

        Amount: ${invoice.amount_due / 100:.2f}
        Date: {datetime.fromtimestamp(invoice.created).strftime('%Y-%m-%d')}
        Invoice ID: {invoice.id}

        Please update your payment method to avoid service interruption.
        You can update your payment details here: [Payment Update Link]

        If you need assistance, please contact our support team.
        """

        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP('smtp.yourdomain.com', 587) as server:
            server.starttls()
            server.login('your-email', 'your-password')
            server.send_message(msg)

    except Exception as e:
        logging.error(f"Failed payment email error: {str(e)}")

def send_subscription_cancelled_email(email: str, subscription):
    """Send subscription cancellation notification email."""
    try:
        msg = MIMEMultipart()
        msg['Subject'] = 'Subscription Cancelled'
        msg['From'] = 'noreply@yourdomain.com'
        msg['To'] = email

        body = f"""
        Your subscription has been cancelled.

        Active until: {datetime.fromtimestamp(subscription.current_period_end).strftime('%Y-%m-%d')}
        
        You will continue to have access to your subscription benefits until the end of your current billing period.

        If you'd like to reactivate your subscription, you can do so here: [Reactivation Link]

        Thank you for being our customer. We hope to see you again soon!
        """

        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP('smtp.yourdomain.com', 587) as server:
            server.starttls()
            server.login('your-email', 'your-password')
            server.send_message(msg)

    except Exception as e:
        logging.error(f"Cancellation email error: {str(e)}")

def handle_stripe_webhook(payload: Dict, signature: str):
    """Handle Stripe webhook events."""
    try:
        event = stripe.Webhook.construct_event(
            payload, signature, STRIPE_WEBHOOK_SECRET
        )

        if event.type == 'invoice.payment_succeeded':
            handle_successful_payment(event.data.object)
        elif event.type == 'invoice.payment_failed':
            handle_failed_payment(event.data.object)
        elif event.type == 'customer.subscription.deleted':
            handle_subscription_cancelled(event.data.object)

    except Exception as e:
        logging.error(f"Webhook error: {str(e)}")
        raise e

def handle_successful_payment(invoice):
    """Handle successful payment webhook."""
    try:
        conn = sqlite3.connect('social_automation.db')
        c = conn.cursor()
        
        # Record payment
        c.execute('''
            INSERT INTO payment_history 
            (user_id, amount, currency, stripe_payment_id, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            invoice.customer,
            invoice.amount_paid / 100.0,
            invoice.currency,
            invoice.payment_intent,
            'succeeded',
            datetime.fromtimestamp(invoice.created)
        ))
        
        # Update subscription status
        c.execute('''
            UPDATE subscriptions 
            SET status = 'active', 
                current_period_end = ?
            WHERE stripe_customer_id = ?
        ''', (
            datetime.fromtimestamp(invoice.lines.data[0].period.end),
            invoice.customer
        ))
        
        conn.commit()
        conn.close()

        # Send success email
        send_payment_confirmation_email(invoice)

    except Exception as e:
        logging.error(f"Payment success handling error: {str(e)}")

def send_payment_confirmation_email(invoice):
    """Send payment confirmation email."""
    try:
        # Get user email
        conn = sqlite3.connect('social_automation.db')
        c = conn.cursor()
        c.execute('''
            SELECT email 
            FROM users 
            WHERE id IN (
                SELECT user_id 
                FROM subscriptions 
                WHERE stripe_customer_id = ?
            )
        ''', (invoice.customer,))
        email = c.fetchone()[0]
        conn.close()

        # Create email message
        msg = MIMEMultipart()
        msg['Subject'] = 'Payment Confirmation'
        msg['From'] = 'noreply@yourdomain.com'
        msg['To'] = email

        body = f"""
        Thank you for your payment!

        Amount: ${invoice.amount_paid / 100:.2f}
        Date: {datetime.fromtimestamp(invoice.created).strftime('%Y-%m-%d')}
        Invoice ID: {invoice.id}

        Your subscription has been renewed successfully.
        """

        msg.attach(MIMEText(body, 'plain'))

        # Send email
        with smtplib.SMTP('smtp.yourdomain.com', 587) as server:
            server.starttls()
            server.login('your-email', 'your-password')
            server.send_message(msg)

    except Exception as e:
        logging.error(f"Email sending error: {str(e)}")