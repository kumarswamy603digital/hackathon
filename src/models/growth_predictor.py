"""
Growth Prediction Engine
Predicts future follower growth, engagement growth, and audience expansion.
Outputs: Growth Potential Score (0-100)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import pickle
import os


class GrowthPredictor:
    """
    Multi-model ensemble for predicting influencer growth potential.
    
    Predicts:
    - Future follower growth trajectory
    - Future engagement growth
    - Audience expansion potential
    
    Uses time-series inspired features and trend analysis to forecast
    growth over the next 3-6 months.
    """
    
    def __init__(self):
        self.follower_growth_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        self.engagement_growth_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.audience_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=7,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
    
    def _engineer_features(self, df):
        """Create growth-prediction features."""
        features = pd.DataFrame()
        
        # Current metrics
        features['log_followers'] = np.log1p(df['followers'])
        features['engagement_rate'] = df['engagement_rate']
        features['posting_frequency'] = df['posting_frequency']
        
        # Current growth trajectory
        features['follower_growth_rate'] = df['follower_growth_rate']
        features['engagement_growth_rate'] = df['engagement_growth_rate']
        
        # Account maturity
        features['account_age_months'] = df['account_age_months']
        features['maturity_factor'] = np.log1p(df['account_age_months'])
        
        # Content performance signals
        features['saves_ratio'] = df['avg_saves'] / (df['avg_likes'] + 1)
        features['shares_ratio'] = df['avg_shares'] / (df['avg_likes'] + 1)
        features['views_to_followers'] = df['avg_views'] / (df['followers'] + 1)
        
        # Engagement depth (higher depth = more viral potential)
        features['engagement_depth'] = (
            (df['avg_comments'] * 3 + df['avg_shares'] * 5 + df['avg_saves'] * 4)
            / (df['avg_likes'] + 1)
        )
        
        # Audience quality signals
        features['audience_quality'] = df['audience_quality']
        features['comment_quality'] = df['comment_quality']
        
        # Market position
        features['followers_per_month_age'] = df['followers'] / (df['account_age_months'] + 1)
        
        # Momentum indicators
        features['growth_momentum'] = (
            df['follower_growth_rate'] * 0.5 + df['engagement_growth_rate'] * 0.5
        )
        
        # Content consistency
        features['consistency_score'] = np.clip(
            df['posting_frequency'] / 7, 0, 2  # Normalize to daily posting
        )
        
        # Virality indicators
        features['viral_potential'] = (
            features['shares_ratio'] * 0.4 +
            features['saves_ratio'] * 0.3 +
            (df['avg_views'] / (df['followers'] + 1)) * 0.3
        )
        
        # Platform-specific encoding
        platform_dummies = pd.get_dummies(df['platform'], prefix='platform')
        for col in ['platform_Instagram', 'platform_YouTube', 'platform_TikTok', 'platform_LinkedIn']:
            if col in platform_dummies.columns:
                features[col] = platform_dummies[col]
            else:
                features[col] = 0
        
        # Niche growth potential (some niches grow faster)
        niche_growth_map = {
            "Technology": 1.2, "Fashion": 1.0, "Fitness": 1.1,
            "Food": 0.9, "Travel": 0.8, "Beauty": 1.0,
            "Gaming": 1.3, "Finance": 1.4, "Education": 1.2,
            "Lifestyle": 0.9, "Health": 1.1, "Music": 1.0,
            "Photography": 0.8, "Business": 1.3, "Art": 0.9
        }
        features['niche_growth_factor'] = df['niche'].map(niche_growth_map).fillna(1.0)
        
        # Verified boost
        features['verified'] = df['verified'].astype(int)
        
        # Brand collaboration history (indicates marketability)
        features['brand_collaborations'] = df['brand_collaborations']
        
        return features
    
    def _generate_growth_targets(self, df):
        """Generate realistic growth targets for training."""
        # Future follower growth (3-month projection as %)
        future_follower_growth = (
            df['follower_growth_rate'] * 0.6 +  # Current trend weight
            df['engagement_rate'] * 0.8 +  # Engagement drives growth
            (df['avg_shares'] / (df['avg_likes'] + 1)) * 20 +  # Virality
            df['audience_quality'] * 3 -  # Quality audience grows organically
            np.log1p(df['followers']) * 0.3 +  # Larger accounts grow slower (%)
            np.random.normal(0, 1, len(df))  # Noise
        )
        future_follower_growth = np.clip(future_follower_growth, -5, 30)
        
        # Future engagement growth
        future_engagement_growth = (
            df['engagement_growth_rate'] * 0.5 +
            df['comment_quality'] * 3 +
            df['posting_frequency'] / 14 * 2 -
            (df['followers'] > 500000).astype(int) * 2 +
            np.random.normal(0, 0.5, len(df))
        )
        future_engagement_growth = np.clip(future_engagement_growth, -5, 15)
        
        # Audience expansion score (0-1)
        audience_expansion = (
            (df['avg_shares'] / (df['avg_likes'] + 1)) * 5 +
            (df['avg_views'] / (df['followers'] + 1)) * 2 +
            df['audience_quality'] * 0.3 +
            np.random.normal(0, 0.05, len(df))
        )
        audience_expansion = np.clip(audience_expansion, 0, 1)
        
        return future_follower_growth, future_engagement_growth, audience_expansion
    
    def train(self, df):
        """Train the growth prediction models."""
        features = self._engineer_features(df)
        self.feature_columns = features.columns.tolist()
        
        X = features.values
        X_scaled = self.scaler.fit_transform(X)
        
        # Generate targets
        y_follower, y_engagement, y_audience = self._generate_growth_targets(df)
        
        # Train follower growth model
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_follower, test_size=0.2, random_state=42
        )
        self.follower_growth_model.fit(X_train, y_train)
        follower_r2 = r2_score(y_test, self.follower_growth_model.predict(X_test))
        
        # Train engagement growth model
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_engagement, test_size=0.2, random_state=42
        )
        self.engagement_growth_model.fit(X_train, y_train)
        engagement_r2 = r2_score(y_test, self.engagement_growth_model.predict(X_test))
        
        # Train audience expansion model
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_audience, test_size=0.2, random_state=42
        )
        self.audience_model.fit(X_train, y_train)
        audience_r2 = r2_score(y_test, self.audience_model.predict(X_test))
        
        self.is_trained = True
        
        return {
            "follower_growth_r2": round(follower_r2, 4),
            "engagement_growth_r2": round(engagement_r2, 4),
            "audience_expansion_r2": round(audience_r2, 4)
        }
    
    def predict_growth(self, df):
        """
        Predict growth potential for influencers.
        Returns a Growth Potential Score (0-100).
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        features = self._engineer_features(df)
        X = self.scaler.transform(features.values)
        
        # Predict individual components
        follower_growth_pred = self.follower_growth_model.predict(X)
        engagement_growth_pred = self.engagement_growth_model.predict(X)
        audience_expansion_pred = self.audience_model.predict(X)
        
        # Normalize predictions to 0-100
        follower_score = np.clip((follower_growth_pred + 5) / 35 * 100, 0, 100)
        engagement_score = np.clip((engagement_growth_pred + 5) / 20 * 100, 0, 100)
        audience_score = np.clip(audience_expansion_pred * 100, 0, 100)
        
        # Weighted combination for overall Growth Potential Score
        growth_score = (
            follower_score * 0.4 +
            engagement_score * 0.35 +
            audience_score * 0.25
        )
        
        return growth_score.round(1)
    
    def get_growth_breakdown(self, influencer_row):
        """Get detailed growth prediction breakdown."""
        df = pd.DataFrame([influencer_row])
        features = self._engineer_features(df)
        X = self.scaler.transform(features.values)
        
        follower_growth_pred = self.follower_growth_model.predict(X)[0]
        engagement_growth_pred = self.engagement_growth_model.predict(X)[0]
        audience_expansion_pred = self.audience_model.predict(X)[0]
        
        overall_score = self.predict_growth(df)[0]
        
        return {
            "overall_growth_score": overall_score,
            "predicted_follower_growth_3m": round(follower_growth_pred, 2),
            "predicted_engagement_growth_3m": round(engagement_growth_pred, 2),
            "audience_expansion_potential": round(audience_expansion_pred * 100, 1),
            "growth_drivers": self._identify_growth_drivers(influencer_row),
            "growth_risks": self._identify_growth_risks(influencer_row)
        }
    
    def _identify_growth_drivers(self, row):
        """Identify key factors driving growth potential."""
        drivers = []
        if row.get('engagement_rate', 0) > 4:
            drivers.append("High engagement rate indicates strong audience connection")
        if row.get('avg_shares', 0) / (row.get('avg_likes', 1)) > 0.05:
            drivers.append("High share ratio suggests viral content potential")
        if row.get('posting_frequency', 0) >= 5:
            drivers.append("Consistent posting frequency builds audience habits")
        if row.get('audience_quality', 0) > 0.85:
            drivers.append("High-quality audience amplifies organic reach")
        if row.get('follower_growth_rate', 0) > 5:
            drivers.append("Strong current growth momentum")
        return drivers if drivers else ["Steady performance with room for optimization"]
    
    def _identify_growth_risks(self, row):
        """Identify potential risks to growth."""
        risks = []
        if row.get('engagement_growth_rate', 0) < -2:
            risks.append("Declining engagement may signal audience fatigue")
        if row.get('posting_frequency', 0) < 2:
            risks.append("Low posting frequency limits visibility")
        if row.get('audience_quality', 0) < 0.6:
            risks.append("Low audience quality may limit organic growth")
        if row.get('followers', 0) > 1000000 and row.get('engagement_rate', 0) < 1.5:
            risks.append("Large account with low engagement - growth plateau risk")
        return risks if risks else ["No significant growth risks identified"]
    
    def save_model(self, path="models/growth_model.pkl"):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            "follower_growth_model": self.follower_growth_model,
            "engagement_growth_model": self.engagement_growth_model,
            "audience_model": self.audience_model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "is_trained": self.is_trained
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path="models/growth_model.pkl"):
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.follower_growth_model = model_data["follower_growth_model"]
        self.engagement_growth_model = model_data["engagement_growth_model"]
        self.audience_model = model_data["audience_model"]
        self.scaler = model_data["scaler"]
        self.feature_columns = model_data["feature_columns"]
        self.is_trained = model_data["is_trained"]
