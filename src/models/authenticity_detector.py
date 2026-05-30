"""
Authenticity Detection Model
Detects fake followers, engagement pods, bot activity, and artificial engagement spikes.
Outputs: Authenticity Score (0-100)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import pickle
import os


class AuthenticityDetector:
    """
    Multi-model ensemble for detecting fake/inauthentic influencer accounts.
    
    Uses a combination of:
    - XGBoost for pattern recognition in engagement metrics
    - Random Forest for robust classification
    - Gradient Boosting for fine-grained probability estimation
    
    Features analyzed:
    - Follower-to-engagement ratio anomalies
    - Growth pattern irregularities
    - Comment quality indicators
    - Audience demographics consistency
    - Posting frequency patterns
    """
    
    def __init__(self):
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        self.rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=42
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
    
    def _engineer_features(self, df):
        """Create advanced features for authenticity detection."""
        features = pd.DataFrame()
        
        # Engagement ratio features
        features['engagement_rate'] = df['engagement_rate']
        features['likes_to_followers'] = df['avg_likes'] / (df['followers'] + 1)
        features['comments_to_likes'] = df['avg_comments'] / (df['avg_likes'] + 1)
        features['shares_to_likes'] = df['avg_shares'] / (df['avg_likes'] + 1)
        features['saves_to_likes'] = df['avg_saves'] / (df['avg_likes'] + 1)
        features['views_to_followers'] = df['avg_views'] / (df['followers'] + 1)
        
        # Following ratio (suspicious if too high)
        features['following_ratio'] = df['following'] / (df['followers'] + 1)
        
        # Growth patterns
        features['follower_growth_rate'] = df['follower_growth_rate']
        features['engagement_growth_rate'] = df['engagement_growth_rate']
        features['growth_engagement_gap'] = df['follower_growth_rate'] - df['engagement_growth_rate']
        
        # Account maturity
        features['account_age_months'] = df['account_age_months']
        features['followers_per_month'] = df['followers'] / (df['account_age_months'] + 1)
        
        # Quality indicators
        features['audience_quality'] = df['audience_quality']
        features['comment_quality'] = df['comment_quality']
        
        # Posting patterns
        features['posting_frequency'] = df['posting_frequency']
        
        # Interaction depth
        features['interaction_depth'] = (
            df['avg_comments'] + df['avg_shares'] + df['avg_saves']
        ) / (df['avg_likes'] + 1)
        
        # Suspicious pattern indicators
        features['high_followers_low_engagement'] = (
            (df['followers'] > 50000) & (df['engagement_rate'] < 1)
        ).astype(int)
        
        features['rapid_growth_declining_engagement'] = (
            (df['follower_growth_rate'] > 10) & (df['engagement_growth_rate'] < 0)
        ).astype(int)
        
        # Log transforms for skewed features
        features['log_followers'] = np.log1p(df['followers'])
        features['log_following'] = np.log1p(df['following'])
        
        # Verified status
        features['verified'] = df['verified'].astype(int)
        
        # Brand collaboration history
        features['brand_collaborations'] = df['brand_collaborations']
        
        return features
    
    def train(self, df):
        """Train the ensemble model on influencer data."""
        features = self._engineer_features(df)
        self.feature_columns = features.columns.tolist()
        
        X = features.values
        y = df['is_authentic'].astype(int).values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        self.xgb_model.fit(X_train, y_train)
        self.rf_model.fit(X_train, y_train)
        self.gb_model.fit(X_train, y_train)
        
        # Evaluate
        xgb_pred = self.xgb_model.predict_proba(X_test)[:, 1]
        rf_pred = self.rf_model.predict_proba(X_test)[:, 1]
        gb_pred = self.gb_model.predict_proba(X_test)[:, 1]
        
        # Ensemble prediction (weighted average)
        ensemble_pred = 0.4 * xgb_pred + 0.3 * rf_pred + 0.3 * gb_pred
        ensemble_binary = (ensemble_pred > 0.5).astype(int)
        
        auc_score = roc_auc_score(y_test, ensemble_pred)
        
        self.is_trained = True
        
        return {
            "auc_score": round(auc_score, 4),
            "report": classification_report(y_test, ensemble_binary, output_dict=True),
            "feature_importance": dict(zip(
                self.feature_columns,
                self.xgb_model.feature_importances_.tolist()
            ))
        }
    
    def predict_authenticity(self, df):
        """
        Predict authenticity score for influencers.
        Returns a score from 0 (likely fake) to 100 (highly authentic).
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        features = self._engineer_features(df)
        X = self.scaler.transform(features.values)
        
        # Ensemble predictions
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        gb_proba = self.gb_model.predict_proba(X)[:, 1]
        
        # Weighted ensemble
        authenticity_proba = 0.4 * xgb_proba + 0.3 * rf_proba + 0.3 * gb_proba
        
        # Convert to 0-100 score
        authenticity_scores = (authenticity_proba * 100).round(1)
        
        return authenticity_scores
    
    def get_authenticity_breakdown(self, influencer_row):
        """Get detailed breakdown of authenticity indicators."""
        df = pd.DataFrame([influencer_row])
        score = self.predict_authenticity(df)[0]
        
        breakdown = {
            "overall_score": score,
            "engagement_consistency": min(100, max(0, 
                100 - abs(df['engagement_rate'].values[0] - 3.5) * 15
            )),
            "growth_pattern": min(100, max(0,
                100 - max(0, df['follower_growth_rate'].values[0] - 8) * 5
            )),
            "audience_quality": df['audience_quality'].values[0] * 100,
            "comment_quality": df['comment_quality'].values[0] * 100,
            "following_ratio_health": min(100, max(0,
                100 - df['following'].values[0] / (df['followers'].values[0] + 1) * 200
            )),
            "risk_flags": []
        }
        
        # Add risk flags
        if df['follower_growth_rate'].values[0] > 15:
            breakdown["risk_flags"].append("Unusually rapid follower growth")
        if df['engagement_rate'].values[0] < 1 and df['followers'].values[0] > 50000:
            breakdown["risk_flags"].append("Very low engagement for follower count")
        if df['engagement_growth_rate'].values[0] < -3:
            breakdown["risk_flags"].append("Declining engagement rate")
        if df['audience_quality'].values[0] < 0.5:
            breakdown["risk_flags"].append("Low audience quality detected")
        if df['following'].values[0] / (df['followers'].values[0] + 1) > 0.5:
            breakdown["risk_flags"].append("Suspicious following-to-follower ratio")
        
        return breakdown
    
    def save_model(self, path="models/authenticity_model.pkl"):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            "xgb_model": self.xgb_model,
            "rf_model": self.rf_model,
            "gb_model": self.gb_model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "is_trained": self.is_trained
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path="models/authenticity_model.pkl"):
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.xgb_model = model_data["xgb_model"]
        self.rf_model = model_data["rf_model"]
        self.gb_model = model_data["gb_model"]
        self.scaler = model_data["scaler"]
        self.feature_columns = model_data["feature_columns"]
        self.is_trained = model_data["is_trained"]
