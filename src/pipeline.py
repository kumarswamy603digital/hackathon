"""
Ratefluencer AI Pipeline
Orchestrates the entire influencer intelligence workflow:
Data Generation → Authenticity Detection → Growth Prediction → Brand Matching → Ratefluencer Score
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data.generate_dataset import generate_dataset, generate_brand_dataset
from src.models.authenticity_detector import AuthenticityDetector
from src.models.growth_predictor import GrowthPredictor
from src.models.brand_matcher import BrandMatcher
from src.models.ratefluencer_score import RatefluencerScorer


class RatefluencerPipeline:
    """
    End-to-end Ratefluencer AI Pipeline.
    
    Workflow:
    1. Generate/load influencer and brand data
    2. Train authenticity detection model
    3. Train growth prediction model
    4. Fit brand matching system
    5. Train Ratefluencer Score ensemble
    6. Generate comprehensive scores for all influencers
    """
    
    def __init__(self):
        self.authenticity_model = AuthenticityDetector()
        self.growth_model = GrowthPredictor()
        self.brand_matcher = BrandMatcher()
        self.scorer = RatefluencerScorer()
        self.influencer_data = None
        self.brand_data = None
        self.scores = None
        self.is_ready = False
    
    def generate_data(self, n_influencers=1000, n_brands=50):
        """Generate synthetic datasets."""
        print("📊 Generating influencer dataset...")
        self.influencer_data = generate_dataset(n_influencers=n_influencers)
        print(f"   Generated {len(self.influencer_data)} influencer profiles")
        
        print("🏢 Generating brand dataset...")
        self.brand_data = generate_brand_dataset(n_brands=n_brands)
        print(f"   Generated {len(self.brand_data)} brand profiles")
        
        return self.influencer_data, self.brand_data
    
    def load_data(self, influencer_path=None, brand_path=None):
        """Load existing datasets."""
        if influencer_path and os.path.exists(influencer_path):
            self.influencer_data = pd.read_csv(influencer_path)
        if brand_path and os.path.exists(brand_path):
            self.brand_data = pd.read_csv(brand_path)
    
    def train_all_models(self):
        """Train all ML models in the pipeline."""
        if self.influencer_data is None:
            raise ValueError("No data loaded. Call generate_data() or load_data() first.")
        
        print("\n🔍 Training Authenticity Detection Model...")
        auth_results = self.authenticity_model.train(self.influencer_data)
        print(f"   AUC Score: {auth_results['auc_score']}")
        
        print("\n📈 Training Growth Prediction Model...")
        growth_results = self.growth_model.train(self.influencer_data)
        print(f"   Follower Growth R²: {growth_results['follower_growth_r2']}")
        print(f"   Engagement Growth R²: {growth_results['engagement_growth_r2']}")
        
        print("\n🤝 Fitting Brand Matching System...")
        self.brand_matcher.fit(self.brand_data)
        print(f"   Brand profiles loaded: {len(self.brand_data)}")
        
        print("\n⭐ Training Ratefluencer Score Model...")
        # Generate sub-scores first
        authenticity_scores = self.authenticity_model.predict_authenticity(self.influencer_data)
        growth_scores = self.growth_model.predict_growth(self.influencer_data)
        brand_scores = self.brand_matcher.get_brand_match_score(self.influencer_data)
        
        scorer_results = self.scorer.train(
            self.influencer_data,
            authenticity_scores=authenticity_scores,
            growth_scores=growth_scores,
            brand_scores=brand_scores
        )
        print(f"   R² Score: {scorer_results['r2_score']}")
        print(f"   MAE: {scorer_results['mae']}")
        
        self.is_ready = True
        
        return {
            "authenticity": auth_results,
            "growth": growth_results,
            "scorer": scorer_results
        }
    
    def score_influencers(self, df=None):
        """Generate all scores for influencers."""
        if not self.is_ready:
            raise ValueError("Pipeline not ready. Call train_all_models() first.")
        
        if df is None:
            df = self.influencer_data
        
        # Generate individual scores
        authenticity_scores = self.authenticity_model.predict_authenticity(df)
        growth_scores = self.growth_model.predict_growth(df)
        brand_scores = self.brand_matcher.get_brand_match_score(df)
        ratefluencer_scores = self.scorer.predict_score(
            df, authenticity_scores, growth_scores, brand_scores
        )
        
        # Create results DataFrame
        results = df.copy()
        results['authenticity_score'] = authenticity_scores
        results['growth_score'] = growth_scores
        results['brand_match_score'] = brand_scores
        results['ratefluencer_score'] = ratefluencer_scores
        
        # Add tier classification
        results['tier'] = pd.cut(
            results['ratefluencer_score'],
            bins=[0, 40, 55, 70, 85, 100],
            labels=['Needs Improvement', 'Developing Creator', 'Solid Performer', 'Rising Star', 'Elite Creator']
        )
        
        # Sort by Ratefluencer Score
        results = results.sort_values('ratefluencer_score', ascending=False).reset_index(drop=True)
        results['rank'] = range(1, len(results) + 1)
        
        self.scores = results
        return results
    
    def get_influencer_report(self, influencer_id):
        """Get comprehensive report for a single influencer."""
        if self.scores is None:
            self.score_influencers()
        
        row = self.scores[self.scores['influencer_id'] == influencer_id].iloc[0]
        
        # Authenticity breakdown
        auth_breakdown = self.authenticity_model.get_authenticity_breakdown(row.to_dict())
        
        # Growth breakdown
        growth_breakdown = self.growth_model.get_growth_breakdown(row.to_dict())
        
        # Brand matches
        brand_matches = self.brand_matcher.match_brands(
            pd.DataFrame([row]), top_k=5
        )[0]
        
        # Overall score breakdown
        score_breakdown = self.scorer.get_score_breakdown(
            row.to_dict(),
            authenticity_score=row['authenticity_score'],
            growth_score=row['growth_score'],
            brand_score=row['brand_match_score']
        )
        
        return {
            "influencer_id": influencer_id,
            "name": row.get('name', 'Unknown'),
            "username": row.get('username', 'Unknown'),
            "platform": row.get('platform', 'Unknown'),
            "niche": row.get('niche', 'Unknown'),
            "followers": row.get('followers', 0),
            "rank": row.get('rank', 0),
            "ratefluencer_score": row['ratefluencer_score'],
            "tier": str(row['tier']),
            "authenticity": auth_breakdown,
            "growth": growth_breakdown,
            "brand_matches": brand_matches,
            "score_breakdown": score_breakdown
        }
    
    def run_full_pipeline(self, n_influencers=1000, n_brands=50):
        """Run the complete pipeline from data generation to scoring."""
        print("=" * 60)
        print("🚀 RATEFLUENCER AI - Influencer Intelligence Engine")
        print("=" * 60)
        
        # Step 1: Generate Data
        self.generate_data(n_influencers, n_brands)
        
        # Step 2: Train Models
        results = self.train_all_models()
        
        # Step 3: Score All Influencers
        print("\n🏆 Scoring all influencers...")
        scored = self.score_influencers()
        
        print(f"\n{'=' * 60}")
        print("✅ Pipeline Complete!")
        print(f"{'=' * 60}")
        print(f"\n📊 Score Distribution:")
        print(f"   Elite Creators (85-100):    {(scored['ratefluencer_score'] >= 85).sum()}")
        print(f"   Rising Stars (70-85):       {((scored['ratefluencer_score'] >= 70) & (scored['ratefluencer_score'] < 85)).sum()}")
        print(f"   Solid Performers (55-70):   {((scored['ratefluencer_score'] >= 55) & (scored['ratefluencer_score'] < 70)).sum()}")
        print(f"   Developing (40-55):         {((scored['ratefluencer_score'] >= 40) & (scored['ratefluencer_score'] < 55)).sum()}")
        print(f"   Needs Improvement (<40):    {(scored['ratefluencer_score'] < 40).sum()}")
        
        print(f"\n🏆 Top 5 Influencers:")
        top5 = scored.head(5)[['rank', 'name', 'username', 'platform', 'niche', 
                               'followers', 'ratefluencer_score', 'tier']]
        print(top5.to_string(index=False))
        
        return scored


if __name__ == "__main__":
    pipeline = RatefluencerPipeline()
    results = pipeline.run_full_pipeline()
    
    # Save results
    os.makedirs("output", exist_ok=True)
    results.to_csv("output/influencer_scores.csv", index=False)
    print("\n💾 Results saved to output/influencer_scores.csv")
