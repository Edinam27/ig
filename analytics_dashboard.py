# analytics_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import logging
from dataclasses import dataclass
import json
from collections import defaultdict

@dataclass
class AnalyticsMetrics:
    """Data class for analytics metrics."""
    platform: str
    date: datetime
    followers: int
    engagement_rate: float
    impressions: int
    reach: int
    likes: int
    comments: int
    shares: int
    profile_visits: int
    website_clicks: int
    post_count: int

class AnalyticsDashboard:
    def __init__(self):
        self.setup_database_connection()
        self.initialize_cache()

    def setup_database_connection(self):
        """Initialize database connection."""
        self.conn = sqlite3.connect('social_automation.db')
        self.cursor = self.conn.cursor()

    def initialize_cache(self):
        """Initialize cache for frequently accessed data."""
        self.cache = {
            'metrics': {},
            'trends': {},
            'top_posts': {}
        }
        self.cache_expiry = 3600  # 1 hour

    def fetch_analytics_data(self, user_id: int, 
                           start_date: datetime, 
                           end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch analytics data for all platforms."""
        try:
            # Fetch metrics from database
            query = """
                SELECT platform, date, followers, engagement_rate, impressions,
                       reach, likes, comments, shares, profile_visits,
                       website_clicks, post_count
                FROM analytics_metrics
                WHERE user_id = ? AND date BETWEEN ? AND ?
            """
            
            df = pd.read_sql_query(
                query,
                self.conn,
                params=(user_id, start_date, end_date),
                parse_dates=['date']
            )
            
            # Split by platform
            platform_data = {
                platform: group for platform, group in df.groupby('platform')
            }
            
            return platform_data

        except Exception as e:
            logging.error(f"Error fetching analytics data: {str(e)}")
            return {}

    def calculate_key_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate key performance metrics."""
        try:
            metrics = {
                'total_engagement': data['likes'].sum() + data['comments'].sum() + data['shares'].sum(),
                'avg_engagement_rate': data['engagement_rate'].mean(),
                'follower_growth': data['followers'].iloc[-1] - data['followers'].iloc[0],
                'total_impressions': data['impressions'].sum(),
                'total_reach': data['reach'].sum(),
                'avg_likes_per_post': data['likes'].sum() / data['post_count'].sum() if data['post_count'].sum() > 0 else 0
            }
            
            return metrics

        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            return {}

    def create_trend_charts(self, data: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create trend charts for various metrics."""
        charts = {}
        
        try:
            # Engagement trend
            engagement_fig = go.Figure()
            engagement_fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['engagement_rate'],
                    mode='lines+markers',
                    name='Engagement Rate'
                )
            )
            engagement_fig.update_layout(
                title='Engagement Rate Trend',
                xaxis_title='Date',
                yaxis_title='Engagement Rate (%)'
            )
            charts['engagement'] = engagement_fig

            # Follower growth
            follower_fig = go.Figure()
            follower_fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['followers'],
                    mode='lines',
                    name='Followers'
                )
            )
            follower_fig.update_layout(
                title='Follower Growth',
                xaxis_title='Date',
                yaxis_title='Followers'
            )
            charts['followers'] = follower_fig

            # Reach and Impressions
            reach_fig = make_subplots(specs=[[{"secondary_y": True}]])
            reach_fig.add_trace(
                go.Bar(
                    x=data['date'],
                    y=data['reach'],
                    name='Reach'
                )
            )
            reach_fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['impressions'],
                    name='Impressions',
                    mode='lines+markers'
                ),
                secondary_y=True
            )
            reach_fig.update_layout(title='Reach and Impressions')
            charts['reach_impressions'] = reach_fig

            return charts

        except Exception as e:
            logging.error(f"Error creating trend charts: {str(e)}")
            return {}

    def analyze_post_performance(self, user_id: int, 
                               platform: str, 
                               date_range: Tuple[datetime, datetime]) -> pd.DataFrame:
        """Analyze individual post performance."""
        try:
            query = """
                SELECT post_id, content_type, posted_at, likes, comments, 
                       shares, impressions, engagement_rate
                FROM post_analytics
                WHERE user_id = ? AND platform = ? 
                AND posted_at BETWEEN ? AND ?
                ORDER BY engagement_rate DESC
            """
            
            posts_df = pd.read_sql_query(
                query,
                self.conn,
                params=(user_id, platform, *date_range),
                parse_dates=['posted_at']
            )
            
            return posts_df

        except Exception as e:
            logging.error(f"Error analyzing post performance: {str(e)}")
            return pd.DataFrame()

    def generate_insights(self, data: pd.DataFrame) -> List[str]:
        """Generate actionable insights from analytics data."""
        insights = []
        
        try:
            # Best posting times
            engagement_by_hour = data.groupby(
                data['date'].dt.hour
            )['engagement_rate'].mean()
            
            best_hours = engagement_by_hour.nlargest(3)
            insights.append(
                f"Best posting times: {', '.join(f'{h}:00' for h in best_hours.index)}"
            )

            # Content type performance
            if 'content_type' in data.columns:
                type_performance = data.groupby('content_type')['engagement_rate'].mean()
                best_type = type_performance.idxmax()
                insights.append(
                    f"Best performing content type: {best_type}"
                )

            # Growth trends
            follower_growth = (
                data['followers'].iloc[-1] - data['followers'].iloc[0]
            ) / data['followers'].iloc[0] * 100
            
            insights.append(
                f"Follower growth rate: {follower_growth:.1f}%"
            )

            return insights

        except Exception as e:
            logging.error(f"Error generating insights: {str(e)}")
            return ["Unable to generate insights"]

    def create_comparison_charts(self, data: Dict[str, pd.DataFrame]) -> Dict[str, go.Figure]:
        """Create platform comparison charts."""
        charts = {}
        
        try:
            # Engagement comparison
            engagement_comp = go.Figure(data=[
                go.Bar(
                    name=platform,
                    x=df['date'],
                    y=df['engagement_rate']
                ) for platform, df in data.items()
            ])
            engagement_comp.update_layout(
                title='Engagement Rate by Platform',
                barmode='group'
            )
            charts['platform_engagement'] = engagement_comp

            # Growth comparison
            growth_comp = go.Figure()
            for platform, df in data.items():
                growth_comp.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df['followers'],
                        name=platform,
                        mode='lines'
                    )
                )
            growth_comp.update_layout(title='Follower Growth by Platform')
            charts['platform_growth'] = growth_comp

            return charts

        except Exception as e:
            logging.error(f"Error creating comparison charts: {str(e)}")
            return {}

def render_analytics_dashboard():
    """Render the analytics dashboard in Streamlit."""
    st.title("Analytics Dashboard")
    
    # Initialize dashboard
    dashboard = AnalyticsDashboard()
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime.now() - timedelta(days=30)
        )
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Fetch data
    data = dashboard.fetch_analytics_data(
        st.session_state.user_id,
        start_date,
        end_date
    )
    
    if not data:
        st.warning("No data available for the selected date range.")
        return
    
    # Platform selection
    platforms = list(data.keys())
    selected_platform = st.selectbox("Select Platform", platforms)
    
    # Display key metrics
    st.subheader("Key Metrics")
    metrics = dashboard.calculate_key_metrics(data[selected_platform])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Engagement", f"{metrics['total_engagement']:,}")
    with col2:
        st.metric("Avg Engagement Rate", f"{metrics['avg_engagement_rate']:.2%}")
    with col3:
        st.metric("Follower Growth", f"{metrics['follower_growth']:,}")
    with col4:
        st.metric("Total Reach", f"{metrics['total_reach']:,}")
    
    # Display trend charts
    st.subheader("Performance Trends")
    charts = dashboard.create_trend_charts(data[selected_platform])
    
    for chart in charts.values():
        st.plotly_chart(chart, use_container_width=True)
    
    # Display top performing posts
    st.subheader("Top Performing Posts")
    top_posts = dashboard.analyze_post_performance(
        st.session_state.user_id,
        selected_platform,
        (start_date, end_date)
    )
    
    if not top_posts.empty:
        st.dataframe(top_posts.head())
    
    # Display insights
    st.subheader("Insights")
    insights = dashboard.generate_insights(data[selected_platform])
    for insight in insights:
        st.info(insight)
    
    # Platform comparison
    if len(platforms) > 1:
        st.subheader("Platform Comparison")
        comparison_charts = dashboard.create_comparison_charts(data)
        for chart in comparison_charts.values():
            st.plotly_chart(chart, use_container_width=True)

if __name__ == "__main__":
    render_analytics_dashboard()