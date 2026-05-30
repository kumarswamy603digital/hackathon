"""
ML-Based Ratefluencer Score™
Ensemble model combining all features to generate an overall Influencer Score (0-100).
Predicts campaign success potential based on multiple dimensions.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle
import os


class RatefluencerScorer:
    """
    The Ratefluencer Score™ - An AI-powered composite scoring system.
    
    Combines multiple ML models and scoring dimensions to predict overall
    influencer value for brand partnerships:
    
    Score Components:
    - Authenticity Score (weighted 25%)
    - Growth Potential Score (weighted 20%)
    - Brand Match Score (weighted 20%)
    - Engagement Quality Score (weighted 20%)
    - Content Performance Score (weighted 15%)
    
    The final score uses an ensemble of:
    - XGBoost for complex pattern recognition
    - Gradient Boosting for stable predictions
    - Neural Network for non-linear relationships
    """
    
    def __init__(self):
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=250,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42
        )
        self.gb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.nn_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
    
    def _engineer_features(self, df, authenticity_scores=None, growth_scores=None, brand_scores=None):
        """Create comprehensive features for the final Ratefluencer Score."""
        features = pd.DataFrame()
        
        # Sub-model scores (if available)
        if authenticity_scores is not None:
            features['authenticity_score'] = authenticity_scores
        if growth_scores is not None:
            features['growth_score'] = growth_scores
        if brand_scores is not None:
            features['brand_match_score'] = brand_scores
        
        # Core engagement metrics
        features['engagement_rate'] = df['engagement_rate']
        features['log_followers'] = np.log1p(df['followers'])
        
        # Engagement quality
        features['comment_to_like_ratio'] = df['avg_comments'] / (df['avg_likes'] + 1)
        features['share_to_like_ratio'] = df['avg_shares'] / (df['avg_likes'] + 1)
        features['save_to_like_ratio'] = df['avg_saves'] / (df['avg_likes'] + 1)
        features['view_to_follower_ratio'] = df['avg_views'] / (df['followers'] + 1)
        
        # Engagement depth score
        features['engagement_depth'] = (
            features['comment_to_like_ratio'] * 30 +
            features['share_to_like_ratio'] * 40 +
            features['save_to_like_ratio'] * 30
        )
        
        # Content performance
        features['content_performance'] = (
            df['avg_views'] / (df['followers'] + 1) * 50 +
            df['avg_saves'] / (df['avg_likes'] + 1) * 30 +
            df['avg_shares'] / (df['avg_likes'] + 1) * 20
        )
        
        # Growth indicators
        features['follower_growth_rate'] = df['follower_growth_rate']
        features['engagement_growth_rate'] = df['engagement_growth_rate']
        features['growth_stability'] = np.abs(
            df['follower_growth_rate'] - df['engagement_growth_rate']
        )
        
        # Account credibility
        features['audience_quality'] = df['audience_quality']
        features['comment_quality'] = df['comment_quality']
        features['account_age_months'] = df['account_age_months']
        features['verified'] = df['verified'].astype(int)
        
        # Posting consistency
        features['posting_frequency'] = df['posting_frequency']
        features['consistency_score'] = np.clip(df['posting_frequency'] / 7, 0, 2)
        
        # Market position
        features['brand_collaborations'] = df['brand_collaborations']
        features['collab_per_month'] = df['brand_collaborations'] / (df['account_age_months'] + 1)
        
        # Following health
        features['following_ratio'] = df['following'] / (df['followers'] + 1)
        
        # Interaction efficiency
        features['total_interactions'] = (
            df['avg_likes'] + df['avg_comments'] * 3 + 
            df['avg_shares'] * 5 + df['avg_saves'] * 4
        )
        features['interaction_per_follower'] = features['total_interactions'] / (df['followers'] + 1)
        
        return features
    
    def _generate_ratefluencer_targets(self, df):
        """Generate Ratefluencer Score targets for training."""
        # Multi-dimensional scoring
        
        # Engagement quality (0-25)
        engagement_score = np.clip(
            df['engagement_rate'] * 3 +
            (df['avg_comments'] / (df['avg_likes'] + 1)) * 50 +
            (df['avg_shares'] / (df['avg_likes'] + 1)) * 80,
            0, 25
        )
        
        # Authenticity (0-25)
        authenticity_score = np.clip(
            df['audience_quality'] * 15 +
            df['comment_quality'] * 10 -
            np.maximum(0, df['follower_growth_rate'] - 10) * 0.5,
            0, 25
        )
        
        # Growth potential (0-20)
        growth_score = np.clip(
            df['follower_growth_rate'] * 1.5 +
            df['engagement_growth_rate'] * 2 +
            (df['avg_saves'] / (df['avg_likes'] + 1)) * 50,
            0, 20
        )
        
        # Brand value (0-20)
        brand_value = np.clip(
            df['brand_collaborations'] * 0.5 +
            np.log1p(df['followers']) * 1.5 +
            df['verified'].astype(int) * 3,
            0, 20
        )
        
        # Content quality (0-10)
        content_score = np.clip(
            (df['avg_views'] / (df['followers'] + 1)) * 20 +
            df['posting_frequency'] / 7 * 5,
            0, 10
        )
        
        # Total Ratefluencer Score
        total_score = engagement_score + authenticity_score + growth_score + brand_value + content_score
        total_score = np.clip(total_score + np.random.normal(0, 2, len(df)), 0, 100)
        
        return total_score
    
    def train(self, df, authenticity_scores=None, growth_scores=None, brand_scores=None):
        """Train the Ratefluencer Score model."""
        features = self._engineer_features(df, authenticity_scores, growth_scores, brand_scores)
        self.feature_columns = features.columns.tolist()
        
        X = features.values
        X_scaled = self.scaler.fit_transform(X)
        
        # Generate targets
        y = self._generate_ratefluencer_targets(df)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train ensemble models
        self.xgb_model.fit(X_train, y_train)
        self.gb_model.fit(X_train, y_train)
        self.nn_model.fit(X_train, y_train)
        
        # Evaluate
        xgb_pred = self.xgb_model.predict(X_test)
        gb_pred = self.gb_model.predict(X_test)
        nn_pred = self.nn_model.predict(X_test)
        
        ensemble_pred = 0.4 * xgb_pred + 0.35 * gb_pred + 0.25 * nn_pred
        
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_test, ensemble_pred)
        mae = mean_absolute_error(y_test, ensemble_pred)
        
        self.is_trained = True
        
        return {
            "r2_score": round(r2, 4),
            "mae": round(mae, 4),
            "feature_importance": dict(zip(
                self.feature_columns,
                self.xgb_model.feature_importances_.tolist()
            ))
        }
    
    def predict_score(self, df, authenticity_scores=None, growth_scores=None, brand_scores=None):
        """
        Generate the Ratefluencer Score™ for each influencer.
        Returns a score from 0 to 100.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        features = self._engineer_features(df, authenticity_scores, growth_scores, brand_scores)
        X = self.scaler.transform(features.values)
        
        # Ensemble prediction
        xgb_pred = self.xgb_model.predict(X)
        gb_pred = self.gb_model.predict(X)
        nn_pred = self.nn_model.predict(X)
        
        # Weighted ensemble
        final_score = 0.4 * xgb_pred + 0.35 * gb_pred + 0.25 * nn_pred
        final_score = np.clip(final_score, 0, 100)
        
        return final_score.round(1)
    
    def get_score_breakdown(self, influencer_row, authenticity_score=None, 
                           growth_score=None, brand_score=None):
        """Get detailed breakdown of the Ratefluencer Score."""
        
        # Calculate individual dimension scores
        engagement_quality = min(100, max(0,
            influencer_row.get('engagement_rate', 0) * 12 +
            influencer_row.get('avg_comments', 0) / (influencer_row.get('avg_likes', 1) + 1) * 200
        ))
        
        content_performance = min(100, max(0,
            influencer_row.get('avg_views', 0) / (influencer_row.get('followers', 1) + 1) * 200 +
            influencer_row.get('avg_saves', 0) / (influencer_row.get('avg_likes', 1) + 1) * 300
        ))
        
        consistency = min(100, max(0,
            influencer_row.get('posting_frequency', 0) / 7 * 50 +
            influencer_row.get('account_age_months', 0) / 24 * 50
        ))
        
        breakdown = {
            "ratefluencer_score": None,  # Will be calculated
            "dimensions": {
                "authenticity": authenticity_score if authenticity_score else 75,
                "growth_potential": growth_score if growth_score else 60,
                "brand_matchability": brand_score if brand_score else 65,
                "engagement_quality": round(engagement_quality, 1),
                "content_performance": round(content_performance, 1),
                "consistency": round(consistency, 1)
            },
            "tier": "",
            "recommendation": ""
        }
        
        # Calculate weighted score
        dims = breakdown["dimensions"]
        weighted_score = (
            dims["authenticity"] * 0.25 +
            dims["growth_potential"] * 0.20 +
            dims["brand_matchability"] * 0.20 +
            dims["engagement_quality"] * 0.20 +
            dims["content_performance"] * 0.10 +
            dims["consistency"] * 0.05
        )
        
        breakdown["ratefluencer_score"] = round(weighted_score, 1)
        
        # Assign tier
        if weighted_score >= 85:
            breakdown["tier"] = "Elite Creator"
            breakdown["recommendation"] = "Top-tier influencer. Ideal for premium brand campaigns."
        elif weighted_score >= 70:
            breakdown["tier"] = "Rising Star"
            breakdown["recommendation"] = "Strong performer with excellent growth potential."
        elif weighted_score >= 55:
            breakdown["tier"] = "Solid Performer"
            breakdown["recommendation"] = "Reliable creator suitable for mid-range campaigns."
        elif weighted_score >= 40:
            breakdown["tier"] = "Developing Creator"
            breakdown["recommendation"] = "Growing presence. Good for micro-influencer campaigns."
        else:
            breakdown["tier"] = "Needs Improvement"
            breakdown["recommendation"] = "Consider audience building before brand partnerships."
        
        return breakdown
    
    def save_model(self, path="models/ratefluencer_model.pkl"):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            "xgb_model": self.xgb_model,
            "gb_model": self.gb_model,
            "nn_model": self.nn_model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "is_trained": self.is_trained
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path="models/ratefluencer_model.pkl"):
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.xgb_model = model_data["xgb_model"]
        self.gb_model = model_data["gb_model"]
        self.nn_model = model_data["nn_model"]
        self.scaler = model_data["scaler"]
        self.feature_columns = model_data["feature_columns"]
        self.is_trained = model_data["is_trained"]
