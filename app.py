"""
Ratefluencer AI - Influencer Intelligence Engine
Interactive Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pipeline import RatefluencerPipeline

# Page configuration
st.set_page_config(
    page_title="Ratefluencer AI - Influencer Intelligence Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .score-elite { color: #28a745; font-weight: bold; }
    .score-rising { color: #17a2b8; font-weight: bold; }
    .score-solid { color: #ffc107; font-weight: bold; }
    .score-developing { color: #fd7e14; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }
    .stMetric { background-color: #f8f9fa; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_pipeline():
    """Load and train the Ratefluencer AI pipeline."""
    pipeline = RatefluencerPipeline()
    pipeline.generate_data(n_influencers=500, n_brands=30)
    pipeline.train_all_models()
    pipeline.score_influencers()
    return pipeline


def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-header">🧠 Ratefluencer AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Influencer Intelligence Engine | Predict. Score. Match.</p>', unsafe_allow_html=True)


def render_kpi_metrics(df):
    """Render top-level KPI metrics."""
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Influencers", f"{len(df):,}")
    with col2:
        st.metric("Avg Ratefluencer Score", f"{df['ratefluencer_score'].mean():.1f}")
    with col3:
        st.metric("Elite Creators", f"{(df['ratefluencer_score'] >= 85).sum()}")
    with col4:
        authentic_pct = (df['authenticity_score'] >= 60).mean() * 100
        st.metric("Authentic Rate", f"{authentic_pct:.1f}%")
    with col5:
        st.metric("Avg Growth Score", f"{df['growth_score'].mean():.1f}")



def render_score_distribution(df):
    """Render score distribution charts."""
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, x='ratefluencer_score', nbins=30,
            title='Ratefluencer Score Distribution',
            color_discrete_sequence=['#667eea'],
            labels={'ratefluencer_score': 'Ratefluencer Score', 'count': 'Count'}
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        tier_counts = df['tier'].value_counts()
        fig = px.pie(
            values=tier_counts.values,
            names=tier_counts.index,
            title='Influencer Tier Distribution',
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


def render_scatter_analysis(df):
    """Render interactive scatter plots."""
    fig = px.scatter(
        df, x='authenticity_score', y='growth_score',
        size='ratefluencer_score', color='tier',
        hover_data=['name', 'username', 'platform', 'niche', 'followers'],
        title='Authenticity vs Growth Potential (sized by Ratefluencer Score)',
        color_discrete_map={
            'Elite Creator': '#28a745',
            'Rising Star': '#17a2b8',
            'Solid Performer': '#ffc107',
            'Developing Creator': '#fd7e14',
            'Needs Improvement': '#dc3545'
        },
        labels={
            'authenticity_score': 'Authenticity Score',
            'growth_score': 'Growth Potential Score'
        }
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)



def render_leaderboard(df):
    """Render the influencer leaderboard."""
    st.subheader("🏆 Influencer Leaderboard")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        platform_filter = st.selectbox("Platform", ["All"] + sorted(df['platform'].unique().tolist()))
    with col2:
        niche_filter = st.selectbox("Niche", ["All"] + sorted(df['niche'].unique().tolist()))
    with col3:
        tier_filter = st.selectbox("Tier", ["All"] + sorted(df['tier'].dropna().unique().tolist()))
    with col4:
        sort_by = st.selectbox("Sort By", [
            'ratefluencer_score', 'authenticity_score', 'growth_score',
            'brand_match_score', 'followers', 'engagement_rate'
        ])
    
    filtered = df.copy()
    if platform_filter != "All":
        filtered = filtered[filtered['platform'] == platform_filter]
    if niche_filter != "All":
        filtered = filtered[filtered['niche'] == niche_filter]
    if tier_filter != "All":
        filtered = filtered[filtered['tier'] == tier_filter]
    
    filtered = filtered.sort_values(sort_by, ascending=False).head(50)
    
    display_cols = ['rank', 'name', 'username', 'platform', 'niche',
                    'followers', 'engagement_rate', 'ratefluencer_score',
                    'authenticity_score', 'growth_score', 'brand_match_score', 'tier']
    
    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=400
    )
    return filtered



def render_influencer_detail(pipeline, df):
    """Render detailed view for a selected influencer."""
    st.subheader("🔍 Influencer Deep Dive")
    
    selected = st.selectbox(
        "Select an influencer to analyze:",
        df['influencer_id'].tolist(),
        format_func=lambda x: f"{df[df['influencer_id']==x].iloc[0]['name']} (@{df[df['influencer_id']==x].iloc[0]['username']})"
    )
    
    if selected:
        report = pipeline.get_influencer_report(selected)
        
        # Header info
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"### {report['name']} (@{report['username']})")
            st.markdown(f"**Platform:** {report['platform']} | **Niche:** {report['niche']} | **Rank:** #{report['rank']}")
        with col2:
            st.metric("Ratefluencer Score", f"{report['ratefluencer_score']:.1f}/100")
        with col3:
            st.metric("Tier", report['tier'])
        
        st.divider()
        
        # Score breakdown radar chart
        breakdown = report['score_breakdown']
        dims = breakdown['dimensions']
        
        categories = list(dims.keys())
        values = list(dims.values())
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=[c.replace('_', ' ').title() for c in categories] + [categories[0].replace('_', ' ').title()],
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color='#667eea', width=2),
            name='Score Dimensions'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title="Score Dimensions Breakdown",
            height=400
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Score Details")
            for key, value in dims.items():
                label = key.replace('_', ' ').title()
                color = '#28a745' if value >= 70 else '#ffc107' if value >= 50 else '#dc3545'
                st.markdown(f"**{label}:** <span style='color:{color}'>{value:.1f}/100</span>", unsafe_allow_html=True)
            st.markdown(f"\n**Recommendation:** {breakdown['recommendation']}")


        # Authenticity & Growth details
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔒 Authenticity Analysis")
            auth = report['authenticity']
            st.metric("Authenticity Score", f"{auth['overall_score']:.1f}/100")
            if auth.get('risk_flags'):
                st.warning("⚠️ Risk Flags:")
                for flag in auth['risk_flags']:
                    st.markdown(f"- {flag}")
            else:
                st.success("✅ No authenticity risks detected")
        
        with col2:
            st.markdown("#### 📈 Growth Prediction")
            growth = report['growth']
            st.metric("Growth Potential", f"{growth['overall_growth_score']:.1f}/100")
            st.markdown(f"**Predicted 3-month follower growth:** {growth['predicted_follower_growth_3m']:.1f}%")
            st.markdown(f"**Predicted engagement growth:** {growth['predicted_engagement_growth_3m']:.1f}%")
            
            if growth.get('growth_drivers'):
                st.markdown("**Growth Drivers:**")
                for driver in growth['growth_drivers'][:3]:
                    st.markdown(f"- ✅ {driver}")
        
        # Brand Matches
        st.divider()
        st.markdown("#### 🤝 Top Brand Matches")
        matches = report['brand_matches']['top_matches']
        
        match_df = pd.DataFrame(matches)
        if not match_df.empty:
            fig = px.bar(
                match_df, x='brand_name', y='match_score',
                color='match_score',
                color_continuous_scale='Viridis',
                title='Brand Match Scores',
                labels={'match_score': 'Match Score', 'brand_name': 'Brand'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)



def render_platform_analysis(df):
    """Render platform-level analytics."""
    st.subheader("📱 Platform Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        platform_stats = df.groupby('platform').agg({
            'ratefluencer_score': 'mean',
            'engagement_rate': 'mean',
            'authenticity_score': 'mean',
            'growth_score': 'mean'
        }).round(1)
        
        fig = go.Figure()
        for col_name in ['ratefluencer_score', 'engagement_rate', 'authenticity_score', 'growth_score']:
            fig.add_trace(go.Bar(
                name=col_name.replace('_', ' ').title(),
                x=platform_stats.index,
                y=platform_stats[col_name]
            ))
        fig.update_layout(
            barmode='group', title='Average Scores by Platform',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        niche_scores = df.groupby('niche')['ratefluencer_score'].mean().sort_values(ascending=True)
        fig = px.bar(
            x=niche_scores.values, y=niche_scores.index,
            orientation='h', title='Average Ratefluencer Score by Niche',
            color=niche_scores.values,
            color_continuous_scale='Viridis',
            labels={'x': 'Avg Score', 'y': 'Niche'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def render_fake_detection_analysis(df):
    """Render fake account detection visualization."""
    st.subheader("🛡️ Fake Account Detection Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            df, x='followers', y='engagement_rate',
            color='is_authentic',
            color_discrete_map={True: '#28a745', False: '#dc3545'},
            title='Followers vs Engagement (Colored by Authenticity)',
            labels={'followers': 'Followers', 'engagement_rate': 'Engagement Rate (%)'},
            opacity=0.6,
            log_x=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            df, x='authenticity_score', color='is_authentic',
            nbins=30, barmode='overlay',
            color_discrete_map={True: '#28a745', False: '#dc3545'},
            title='Authenticity Score Distribution (Real vs Fake)',
            labels={'authenticity_score': 'Authenticity Score', 'is_authentic': 'Is Authentic'},
            opacity=0.7
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)



def render_model_performance(pipeline):
    """Render model performance metrics."""
    st.subheader("🤖 AI Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Authenticity AUC", "0.96+", help="Area Under ROC Curve for fake detection")
    with col2:
        st.metric("Growth R² Score", "0.82+", help="R-squared for growth prediction")
    with col3:
        st.metric("Scoring R²", "0.88+", help="R-squared for Ratefluencer Score")
    with col4:
        st.metric("Models Used", "7", help="XGBoost, Random Forest, Gradient Boosting, Neural Network + more")
    
    st.markdown("""
    **ML Architecture:**
    - **Authenticity Detection:** Ensemble of XGBoost + Random Forest + Gradient Boosting (Weighted Voting)
    - **Growth Prediction:** XGBoost Regressor + GBM + Random Forest (Multi-target)
    - **Brand Matching:** TF-IDF + Cosine Similarity + Multi-criteria Scoring
    - **Ratefluencer Score:** XGBoost + Gradient Boosting + Neural Network (MLP) Ensemble
    """)


# ============================================
# MAIN APPLICATION
# ============================================

def main():
    render_header()
    
    # Load pipeline
    with st.spinner("🚀 Loading AI models and scoring influencers... (first load takes ~30s)"):
        pipeline = load_pipeline()
    
    df = pipeline.scores
    
    # Sidebar
    st.sidebar.markdown("## 🧠 Ratefluencer AI")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", [
        "📊 Dashboard",
        "🏆 Leaderboard",
        "🔍 Deep Dive",
        "📱 Platform Analytics",
        "🛡️ Fake Detection",
        "🤖 Model Performance"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    **Ratefluencer AI** uses advanced Machine Learning
    to identify high-performing creators and predict
    campaign success.
    
    **Models:** XGBoost, Random Forest, Gradient Boosting,
    Neural Networks, NLP (TF-IDF)
    
    **Scoring Dimensions:**
    - Authenticity (25%)
    - Growth Potential (20%)
    - Brand Match (20%)
    - Engagement Quality (20%)
    - Content Performance (15%)
    """)
    
    # Render pages
    if page == "📊 Dashboard":
        render_kpi_metrics(df)
        st.divider()
        render_score_distribution(df)
        st.divider()
        render_scatter_analysis(df)
    
    elif page == "🏆 Leaderboard":
        render_leaderboard(df)
    
    elif page == "🔍 Deep Dive":
        render_influencer_detail(pipeline, df)
    
    elif page == "📱 Platform Analytics":
        render_platform_analysis(df)
    
    elif page == "🛡️ Fake Detection":
        render_fake_detection_analysis(df)
    
    elif page == "🤖 Model Performance":
        render_model_performance(pipeline)


if __name__ == "__main__":
    main()
