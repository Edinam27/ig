# analytics_manager.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp
import json
import logging
from dataclasses import dataclass
from collections import defaultdict
import calendar
import seaborn as sns
from scipy import stats

@dataclass
class AnalyticsMetrics:
    """Data class for analytics metrics."""
    followers: int
    engagement_rate: float
    likes: int
    comments: int
    shares: int
    views: int
    reach: int
    impressions: int
    saves: int
    profile_visits: int
    website_clicks: int
    timestamp: datetime

class AnalyticsManager:
    def __init__(self):
        self.setup_database()
        self.metrics_cache = {}
        self.update_interval = 3600  # 1 hour in seconds

    def setup_database(self):
        """Initialize analytics database tables."""
        conn = sqlite3.connect('social_automation.db')
        c = conn.cursor()
        
        # Detailed analytics table
        c.execute('''
            CREATE TABLE IF NOT EXISTS detailed_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                platform TEXT,
                post_id TEXT,
                followers INTEGER,
                engagement_rate REAL,
                likes INTEGER,
                comments INTEGER,
                shares INTEGER,
                views INTEGER,
                reach INTEGER,
                impressions INTEGER,
                saves INTEGER,
                profile_visits INTEGER,
                website_clicks INTEGER,
                timestamp TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Aggregated metrics table
        c.execute('''
            CREATE TABLE IF NOT EXISTS aggregated_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                platform TEXT,
                metric_type TEXT,
                value REAL,
                date DATE,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Audience demographics table
        c.execute('''
            CREATE TABLE IF NOT EXISTS audience_demographics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                platform TEXT,
                age_range TEXT,
                gender TEXT,
                location TEXT,
                count INTEGER,
                timestamp TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()

    async def fetch_platform_analytics(self, platform: str, user_id: int) -> Optional[AnalyticsMetrics]:
        """Fetch analytics from social media platforms."""
        try:
            if platform == 'instagram':
                return await self._fetch_instagram_analytics(user_id)
            elif platform == 'twitter':
                return await self._fetch_twitter_analytics(user_id)
            elif platform == 'tiktok':
                return await self._fetch_tiktok_analytics(user_id)
            return None
        except Exception as e:
            logging.error(f"Analytics fetch error for {platform}: {str(e)}")
            return None

    async def update_analytics(self, user_id: int):
        """Update analytics data for all platforms."""
        platforms = ['instagram', 'twitter', 'tiktok']
        
        for platform in platforms:
            metrics = await self.fetch_platform_analytics(platform, user_id)
            if metrics:
                await self._save_metrics(user_id, platform, metrics)

    async def _save_metrics(self, user_id: int, platform: str, metrics: AnalyticsMetrics):
        """Save metrics to database."""
        conn = sqlite3.connect('social_automation.db')
        c = conn.cursor()
        
        try:
            # Save detailed analytics
            c.execute('''
                INSERT INTO detailed_analytics 
                (user_id, platform, followers, engagement_rate, likes, 
                 comments, shares, views, reach, impressions, saves, 
                 profile_visits, website_clicks, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, platform, metrics.followers, metrics.engagement_rate,
                metrics.likes, metrics.comments, metrics.shares, metrics.views,
                metrics.reach, metrics.impressions, metrics.saves,
                metrics.profile_visits, metrics.website_clicks,
                metrics.timestamp.isoformat()
            ))
            
            # Update aggregated metrics
            self._update_aggregated_metrics(c, user_id, platform, metrics)
            
            conn.commit()
        except Exception as e:
            logging.error(f"Error saving metrics: {str(e)}")
            conn.rollback()
        finally:
            conn.close()

    def _update_aggregated_metrics(self, cursor, user_id: int, platform: str, metrics: AnalyticsMetrics):
        """Update aggregated metrics for reporting."""
        date = metrics.timestamp.date()
        
        metric_types = {
            'followers': metrics.followers,
            'engagement_rate': metrics.engagement_rate,
            'total_interactions': metrics.likes + metrics.comments + metrics.shares
        }
        
        for metric_type, value in metric_types.items():
            cursor.execute('''
                INSERT OR REPLACE INTO aggregated_metrics 
                (user_id, platform, metric_type, value, date)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, platform, metric_type, value, date))

class ReportGenerator:
    def __init__(self, analytics_manager: AnalyticsManager):
        self.analytics_manager = analytics_manager

    def generate_performance_report(self, user_id: int, start_date: datetime, 
                                  end_date: datetime) -> Dict:
        """Generate comprehensive performance report."""
        conn = sqlite3.connect('social_automation.db')
        
        # Fetch metrics
        metrics_df = pd.read_sql_query('''
            SELECT * FROM detailed_analytics 
            WHERE user_id = ? AND timestamp BETWEEN ? AND ?
        ''', conn, params=(user_id, start_date, end_date))
        
        # Calculate key statistics
        report = {
            'summary': self._calculate_summary_stats(metrics_df),
            'trends': self._analyze_trends(metrics_df),
            'best_performing_posts': self._identify_best_posts(metrics_df),
            'audience_insights': self._analyze_audience(user_id, conn),
            'recommendations': self._generate_recommendations(metrics_df)
        }
        
        conn.close()
        return report

    def _calculate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics from metrics."""
        return {
            'total_engagement': df['likes'].sum() + df['comments'].sum() + df['shares'].sum(),
            'avg_engagement_rate': df['engagement_rate'].mean(),
            'follower_growth': df['followers'].iloc[-1] - df['followers'].iloc[0],
            'total_impressions': df['impressions'].sum(),
            'total_reach': df['reach'].sum()
        }

    def _analyze_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze trends in metrics over time."""
        # Calculate daily metrics
        daily_metrics = df.groupby(pd.to_datetime(df['timestamp']).dt.date).agg({
            'engagement_rate': 'mean',
            'followers': 'last',
            'likes': 'sum',
            'comments': 'sum'
        })
        
        # Calculate trend lines
        trends = {}
        for column in daily_metrics.columns:
            slope, _, _, _, _ = stats.linregress(range(len(daily_metrics)), daily_metrics[column])
            trends[column] = {
                'trend': 'increasing' if slope > 0 else 'decreasing',
                'slope': slope
            }
        
        return trends

    def _identify_best_posts(self, df: pd.DataFrame) -> List[Dict]:
        """Identify best performing posts based on engagement."""
        df['total_engagement'] = df['likes'] + df['comments'] + df['shares']
        
        best_posts = df.nlargest(5, 'total_engagement')[
            ['post_id', 'total_engagement', 'likes', 'comments', 'shares', 'timestamp']
        ].to_dict('records')
        
        return best_posts

    def _analyze_audience(self, user_id: int, conn) -> Dict:
        """Analyze audience demographics and behavior."""
        demographics_df = pd.read_sql_query('''
            SELECT * FROM audience_demographics WHERE user_id = ?
        ''', conn, params=(user_id,))
        
        return {
            'age_distribution': demographics_df.groupby('age_range')['count'].sum().to_dict(),
            'gender_distribution': demographics_df.groupby('gender')['count'].sum().to_dict(),
            'top_locations': demographics_df.groupby('location')['count'].sum().nlargest(5).to_dict()
        }

    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on analytics."""
        recommendations = []
        
        # Analyze posting times
        engagement_by_hour = df.groupby(pd.to_datetime(df['timestamp']).dt.hour)['engagement_rate'].mean()
        best_hours = engagement_by_hour.nlargest(3).index.tolist()
        recommendations.append(f"Best posting times are at {', '.join(map(str, best_hours))}:00")
        
        # Analyze content performance
        if df['views'].mean() > df['likes'].mean() * 10:
            recommendations.append("Focus on converting views to engagement with stronger calls-to-action")
        
        # Add more recommendations based on other metrics
        return recommendations

def render_analytics_page():
    """Render the analytics dashboard in Streamlit."""
    st.title("Analytics Dashboard")
    
    # Initialize managers
    analytics_manager = AnalyticsManager()
    report_generator = ReportGenerator(analytics_manager)

    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now())

    # Generate report
    report = report_generator.generate_performance_report(
        st.session_state.user_id,
        start_date,
        end_date
    )

    # Display summary metrics
    st.subheader("Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Engagement", f"{report['summary']['total_engagement']:,}")
    with col2:
        st.metric("Avg Engagement Rate", f"{report['summary']['avg_engagement_rate']:.2%}")
    with col3:
        st.metric("Follower Growth", f"{report['summary']['follower_growth']:,}")
    with col4:
        st.metric("Total Reach", f"{report['summary']['total_reach']:,}")

    # Engagement Trends
    st.subheader("Engagement Trends")
    fig = create_engagement_trends_chart(report['trends'])
    st.plotly_chart(fig)

    # Best Performing Posts
    st.subheader("Top Performing Posts")
    display_best_posts(report['best_performing_posts'])

    # Audience Insights
    st.subheader("Audience Insights")
    display_audience_insights(report['audience_insights'])

    # Recommendations
    st.subheader("Recommendations")
    for rec in report['recommendations']:
        st.info(rec)

def create_engagement_trends_chart(trends_data: Dict) -> go.Figure:
    """Create an interactive trends chart using Plotly."""
    fig = make_subplots(rows=2, cols=2)
    
    metrics = ['engagement_rate', 'followers', 'likes', 'comments']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for metric, (row, col) in zip(metrics, positions):
        trend_data = trends_data[metric]
        fig.add_trace(
            go.Scatter(
                y=trend_data['values'],
                name=metric.replace('_', ' ').title(),
                mode='lines+markers'
            ),
            row=row, col=col
        )
    
    fig.update_layout(height=800, title_text="Engagement Metrics Over Time")
    return fig

def display_best_posts(posts: List[Dict]):
    """Display best performing posts in a formatted table."""
    if posts:
        df = pd.DataFrame(posts)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(
            df[['date', 'total_engagement', 'likes', 'comments', 'shares']],
            use_container_width=True
        )

def display_audience_insights(insights: Dict):
    """Display audience demographics using charts."""
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig_age = px.pie(
            values=list(insights['age_distribution'].values()),
            names=list(insights['age_distribution'].keys()),
            title="Age Distribution"
        )
        st.plotly_chart(fig_age)
    
    with col2:
        # Gender distribution
        fig_gender = px.pie(
            values=list(insights['gender_distribution'].values()),
            names=list(insights['gender_distribution'].keys()),
            title="Gender Distribution"
        )
        st.plotly_chart(fig_gender)
    
    # Location map
    st.subheader("Top Locations")
    location_df = pd.DataFrame(
        list(insights['top_locations'].items()),
        columns=['Location', 'Count']
    )
    st.dataframe(location_df)
    
    
    # Usage in main app
    if __name__ == "__main__":
        render_analytics_page()