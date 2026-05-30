"""
Brand Matching System
Recommends the most suitable brands for each creator using NLP, embeddings,
and similarity search.
Outputs: Brand Match Score (0-100)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import json


class BrandMatcher:
    """
    AI-powered Brand-Creator Matching Engine.
    
    Uses multi-dimensional matching based on:
    - Content niche alignment (NLP-based)
    - Audience demographic fit
    - Performance metrics compatibility
    - Brand values alignment
    - Platform preference matching
    
    Implements a hybrid approach combining:
    - TF-IDF vectorization for text-based matching
    - Cosine similarity for profile alignment
    - Weighted multi-criteria scoring
    """
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=100)
        self.brand_profiles = None
        self.brand_vectors = None
        self.is_fitted = False
    
    def _create_influencer_profile_text(self, row):
        """Create a text representation of an influencer's profile for NLP matching."""
        profile_parts = [
            row.get('niche', ''),
            row.get('content_category', ''),
            row.get('platform', ''),
            f"engagement_{self._engagement_tier(row.get('engagement_rate', 0))}",
            f"followers_{self._follower_tier(row.get('followers', 0))}",
            f"audience_quality_{self._quality_tier(row.get('audience_quality', 0))}",
        ]
        
        # Add audience demographics
        if row.get('audience_age_25_34', 0) > 0.35:
            profile_parts.append("young_professional")
        if row.get('audience_age_18_24', 0) > 0.3:
            profile_parts.append("gen_z")
        if row.get('audience_female', 0) > 0.6:
            profile_parts.append("female_audience")
        elif row.get('audience_male', 0) > 0.6:
            profile_parts.append("male_audience")
        
        return " ".join(profile_parts)
    
    def _create_brand_profile_text(self, row):
        """Create a text representation of a brand's requirements."""
        profile_parts = [
            row.get('category', ''),
            row.get('target_audience_age', ''),
            row.get('target_audience_gender', ''),
            row.get('budget_tier', ''),
        ]
        
        # Add preferred niches
        niches = row.get('preferred_niches', [])
        if isinstance(niches, str):
            try:
                niches = json.loads(niches.replace("'", '"'))
            except:
                niches = [niches]
        profile_parts.extend(niches)
        
        # Add platforms
        platforms = row.get('preferred_platforms', [])
        if isinstance(platforms, str):
            try:
                platforms = json.loads(platforms.replace("'", '"'))
            except:
                platforms = [platforms]
        profile_parts.extend(platforms)
        
        # Add values
        values = row.get('values', [])
        if isinstance(values, str):
            try:
                values = json.loads(values.replace("'", '"'))
            except:
                values = [values]
        profile_parts.extend(values)
        
        # Add campaign goals
        goals = row.get('campaign_goals', [])
        if isinstance(goals, str):
            try:
                goals = json.loads(goals.replace("'", '"'))
            except:
                goals = [goals]
        profile_parts.extend(goals)
        
        return " ".join(profile_parts)
    
    def _engagement_tier(self, rate):
        if rate > 5: return "high"
        elif rate > 2: return "medium"
        else: return "low"
    
    def _follower_tier(self, count):
        if count > 1000000: return "mega"
        elif count > 100000: return "macro"
        elif count > 10000: return "mid"
        elif count > 1000: return "micro"
        else: return "nano"
    
    def _quality_tier(self, quality):
        if quality > 0.85: return "premium"
        elif quality > 0.7: return "good"
        else: return "standard"
    
    def fit(self, brands_df):
        """Fit the brand matching system with brand profiles."""
        self.brand_profiles = brands_df.copy()
        
        # Create text profiles for brands
        brand_texts = brands_df.apply(self._create_brand_profile_text, axis=1).tolist()
        
        # Fit TF-IDF on brand profiles
        self.brand_vectors = self.tfidf.fit_transform(brand_texts)
        self.is_fitted = True
    
    def match_brands(self, influencer_df, top_k=5):
        """
        Match influencers with brands.
        Returns brand match scores and recommendations.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        results = []
        
        for idx, row in influencer_df.iterrows():
            # Create influencer text profile
            inf_text = self._create_influencer_profile_text(row)
            inf_vector = self.tfidf.transform([inf_text])
            
            # Calculate text-based similarity
            text_similarity = cosine_similarity(inf_vector, self.brand_vectors)[0]
            
            # Calculate criteria-based match scores
            criteria_scores = self._calculate_criteria_scores(row, self.brand_profiles)
            
            # Combined score (60% criteria, 40% text similarity)
            combined_scores = 0.6 * criteria_scores + 0.4 * text_similarity * 100
            
            # Get top matches
            top_indices = np.argsort(combined_scores)[::-1][:top_k]
            
            matches = []
            for i in top_indices:
                matches.append({
                    "brand_name": self.brand_profiles.iloc[i]['brand_name'],
                    "brand_category": self.brand_profiles.iloc[i]['category'],
                    "match_score": round(combined_scores[i], 1),
                    "text_similarity": round(text_similarity[i] * 100, 1),
                    "criteria_score": round(criteria_scores[i], 1)
                })
            
            results.append({
                "influencer_id": row.get('influencer_id', idx),
                "best_match_score": round(combined_scores[top_indices[0]], 1),
                "top_matches": matches
            })
        
        return results
    
    def _calculate_criteria_scores(self, influencer_row, brands_df):
        """Calculate multi-criteria match scores between an influencer and all brands."""
        scores = np.zeros(len(brands_df))
        
        for idx, brand in brands_df.iterrows():
            score = 0
            max_score = 0
            
            # 1. Niche alignment (25 points)
            max_score += 25
            preferred_niches = brand.get('preferred_niches', [])
            if isinstance(preferred_niches, str):
                try:
                    preferred_niches = json.loads(preferred_niches.replace("'", '"'))
                except:
                    preferred_niches = [preferred_niches]
            
            if influencer_row.get('niche', '') in preferred_niches:
                score += 25
            elif influencer_row.get('content_category', '') in preferred_niches:
                score += 15
            
            # 2. Platform match (15 points)
            max_score += 15
            preferred_platforms = brand.get('preferred_platforms', [])
            if isinstance(preferred_platforms, str):
                try:
                    preferred_platforms = json.loads(preferred_platforms.replace("'", '"'))
                except:
                    preferred_platforms = [preferred_platforms]
            
            if influencer_row.get('platform', '') in preferred_platforms:
                score += 15
            
            # 3. Follower count requirement (15 points)
            max_score += 15
            min_followers = brand.get('min_followers', 0)
            if influencer_row.get('followers', 0) >= min_followers:
                score += 15
            elif influencer_row.get('followers', 0) >= min_followers * 0.7:
                score += 8
            
            # 4. Engagement rate requirement (15 points)
            max_score += 15
            min_engagement = brand.get('min_engagement_rate', 0)
            if influencer_row.get('engagement_rate', 0) >= min_engagement:
                score += 15
            elif influencer_row.get('engagement_rate', 0) >= min_engagement * 0.7:
                score += 8
            
            # 5. Audience demographic match (20 points)
            max_score += 20
            target_age = brand.get('target_audience_age', '')
            target_gender = brand.get('target_audience_gender', '')
            
            # Age match
            age_score = 0
            if '18-24' in target_age and influencer_row.get('audience_age_18_24', 0) > 0.25:
                age_score = 10
            elif '25-34' in target_age and influencer_row.get('audience_age_25_34', 0) > 0.3:
                age_score = 10
            elif '18-34' in target_age:
                combined = influencer_row.get('audience_age_18_24', 0) + influencer_row.get('audience_age_25_34', 0)
                if combined > 0.5:
                    age_score = 10
            else:
                age_score = 5  # Default partial match
            
            # Gender match
            gender_score = 0
            if target_gender == 'All':
                gender_score = 10
            elif target_gender == 'Female' and influencer_row.get('audience_female', 0) > 0.55:
                gender_score = 10
            elif target_gender == 'Male' and influencer_row.get('audience_male', 0) > 0.55:
                gender_score = 10
            else:
                gender_score = 5
            
            score += age_score + gender_score
            
            # 6. Audience quality bonus (10 points)
            max_score += 10
            if influencer_row.get('audience_quality', 0) > 0.8:
                score += 10
            elif influencer_row.get('audience_quality', 0) > 0.6:
                score += 5
            
            # Normalize to 0-100
            scores[idx] = (score / max_score) * 100 if max_score > 0 else 0
        
        return scores
    
    def get_brand_match_score(self, influencer_df):
        """Get overall brand matchability score for each influencer (0-100)."""
        results = self.match_brands(influencer_df, top_k=3)
        scores = []
        for r in results:
            # Average of top 3 match scores
            top_scores = [m['match_score'] for m in r['top_matches']]
            avg_score = np.mean(top_scores) if top_scores else 0
            scores.append(min(100, avg_score))
        return np.array(scores).round(1)
    
    def get_match_explanation(self, influencer_row, brand_row):
        """Generate human-readable explanation of why a match works."""
        explanations = []
        
        preferred_niches = brand_row.get('preferred_niches', [])
        if isinstance(preferred_niches, str):
            try:
                preferred_niches = json.loads(preferred_niches.replace("'", '"'))
            except:
                preferred_niches = [preferred_niches]
        
        if influencer_row.get('niche', '') in preferred_niches:
            explanations.append(f"Content niche '{influencer_row['niche']}' aligns with brand preferences")
        
        if influencer_row.get('engagement_rate', 0) > brand_row.get('min_engagement_rate', 0):
            explanations.append(f"Engagement rate ({influencer_row['engagement_rate']:.1f}%) exceeds brand minimum")
        
        if influencer_row.get('audience_quality', 0) > 0.8:
            explanations.append("High audience quality ensures genuine reach")
        
        if influencer_row.get('followers', 0) > brand_row.get('min_followers', 0):
            explanations.append(f"Follower count meets brand reach requirements")
        
        return explanations if explanations else ["Partial match based on audience demographics"]
