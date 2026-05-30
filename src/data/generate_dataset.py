"""
Synthetic Influencer Dataset Generator
Generates realistic influencer profiles with engagement metrics, audience data,
and labels for training ML models.
"""

import numpy as np
import pandas as pd
from faker import Faker
import random
import json

fake = Faker()
np.random.seed(42)
random.seed(42)

# Content categories and niches
NICHES = [
    "Technology", "Fashion", "Fitness", "Food", "Travel",
    "Beauty", "Gaming", "Finance", "Education", "Lifestyle",
    "Health", "Music", "Photography", "Business", "Art"
]

PLATFORMS = ["Instagram", "YouTube", "TikTok", "LinkedIn"]

BRAND_CATEGORIES = [
    "Tech & SaaS", "Fashion & Apparel", "Health & Wellness", "Food & Beverage",
    "Travel & Hospitality", "Beauty & Skincare", "Gaming & Entertainment",
    "Finance & Fintech", "Education & EdTech", "Luxury & Premium"
]


def generate_authentic_influencer():
    """Generate a genuine influencer with realistic metrics."""
    followers = int(np.random.lognormal(mean=10, sigma=1.5))
    followers = min(followers, 10_000_000)
    
    # Authentic engagement rates decrease with follower count
    base_engagement = max(0.5, 8 - np.log10(followers + 1) * 1.5)
    engagement_rate = base_engagement * np.random.uniform(0.7, 1.3)
    engagement_rate = np.clip(engagement_rate, 0.5, 15)
    
    # Authentic following ratio
    following = int(followers * np.random.uniform(0.01, 0.3))
    
    # Realistic posting frequency (posts per week)
    posting_frequency = np.random.uniform(2, 14)
    
    # Likes, comments, shares derived from engagement
    avg_likes = int(followers * engagement_rate / 100 * np.random.uniform(0.6, 0.9))
    avg_comments = int(avg_likes * np.random.uniform(0.02, 0.08))
    avg_shares = int(avg_likes * np.random.uniform(0.01, 0.05))
    avg_saves = int(avg_likes * np.random.uniform(0.05, 0.15))
    avg_views = int(followers * np.random.uniform(0.15, 0.5))
    
    # Growth metrics (monthly)
    follower_growth_rate = np.random.uniform(0.5, 8)  # % per month
    engagement_growth_rate = np.random.uniform(-1, 5)  # % per month
    
    # Audience quality indicators
    audience_quality = np.random.uniform(0.7, 0.98)
    comment_quality = np.random.uniform(0.6, 0.95)
    
    # Audience demographics
    audience_age_18_24 = np.random.uniform(0.1, 0.4)
    audience_age_25_34 = np.random.uniform(0.2, 0.45)
    audience_age_35_plus = 1 - audience_age_18_24 - audience_age_25_34
    
    audience_female = np.random.uniform(0.3, 0.7)
    audience_male = 1 - audience_female
    
    return {
        "is_authentic": True,
        "followers": followers,
        "following": following,
        "engagement_rate": round(engagement_rate, 3),
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
        "avg_shares": avg_shares,
        "avg_saves": avg_saves,
        "avg_views": avg_views,
        "posting_frequency": round(posting_frequency, 1),
        "follower_growth_rate": round(follower_growth_rate, 2),
        "engagement_growth_rate": round(engagement_growth_rate, 2),
        "audience_quality": round(audience_quality, 3),
        "comment_quality": round(comment_quality, 3),
        "audience_age_18_24": round(audience_age_18_24, 3),
        "audience_age_25_34": round(audience_age_25_34, 3),
        "audience_age_35_plus": round(audience_age_35_plus, 3),
        "audience_female": round(audience_female, 3),
        "audience_male": round(audience_male, 3),
    }


def generate_fake_influencer():
    """Generate an influencer with purchased followers/engagement."""
    followers = int(np.random.lognormal(mean=11, sigma=1))
    followers = min(followers, 5_000_000)
    
    # Suspicious patterns: high followers but inconsistent engagement
    engagement_rate = np.random.uniform(0.1, 2.5)  # Unusually low or artificially high
    
    # Suspicious following ratio (follow-back schemes)
    following = int(followers * np.random.uniform(0.3, 0.9))
    
    # Irregular posting
    posting_frequency = np.random.uniform(0.5, 20)
    
    # Metrics that don't match
    avg_likes = int(followers * np.random.uniform(0.001, 0.03))
    avg_comments = int(avg_likes * np.random.uniform(0.001, 0.02))  # Very low comments
    avg_shares = int(avg_likes * np.random.uniform(0.001, 0.01))
    avg_saves = int(avg_likes * np.random.uniform(0.001, 0.02))
    avg_views = int(followers * np.random.uniform(0.02, 0.1))
    
    # Sudden growth spikes (purchased followers)
    follower_growth_rate = np.random.uniform(15, 50)  # Unrealistic growth
    engagement_growth_rate = np.random.uniform(-5, 0)  # Declining engagement
    
    # Poor audience quality
    audience_quality = np.random.uniform(0.1, 0.5)
    comment_quality = np.random.uniform(0.1, 0.4)
    
    # Random/bot demographics
    audience_age_18_24 = np.random.uniform(0.05, 0.2)
    audience_age_25_34 = np.random.uniform(0.1, 0.3)
    audience_age_35_plus = 1 - audience_age_18_24 - audience_age_25_34
    
    audience_female = np.random.uniform(0.4, 0.6)
    audience_male = 1 - audience_female
    
    return {
        "is_authentic": False,
        "followers": followers,
        "following": following,
        "engagement_rate": round(engagement_rate, 3),
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
        "avg_shares": avg_shares,
        "avg_saves": avg_saves,
        "avg_views": avg_views,
        "posting_frequency": round(posting_frequency, 1),
        "follower_growth_rate": round(follower_growth_rate, 2),
        "engagement_growth_rate": round(engagement_growth_rate, 2),
        "audience_quality": round(audience_quality, 3),
        "comment_quality": round(comment_quality, 3),
        "audience_age_18_24": round(audience_age_18_24, 3),
        "audience_age_25_34": round(audience_age_25_34, 3),
        "audience_age_35_plus": round(audience_age_35_plus, 3),
        "audience_female": round(audience_female, 3),
        "audience_male": round(audience_male, 3),
    }


def generate_dataset(n_influencers=1000, fake_ratio=0.3):
    """Generate a complete influencer dataset."""
    n_fake = int(n_influencers * fake_ratio)
    n_authentic = n_influencers - n_fake
    
    influencers = []
    
    for i in range(n_authentic):
        profile = generate_authentic_influencer()
        profile["influencer_id"] = f"INF_{i:05d}"
        profile["username"] = fake.user_name()
        profile["name"] = fake.name()
        profile["niche"] = random.choice(NICHES)
        profile["platform"] = random.choice(PLATFORMS)
        profile["account_age_months"] = random.randint(6, 84)
        profile["verified"] = random.random() < 0.3
        profile["brand_collaborations"] = random.randint(0, 50)
        profile["content_category"] = random.choice(NICHES)
        influencers.append(profile)
    
    for i in range(n_fake):
        profile = generate_fake_influencer()
        profile["influencer_id"] = f"INF_{n_authentic + i:05d}"
        profile["username"] = fake.user_name()
        profile["name"] = fake.name()
        profile["niche"] = random.choice(NICHES)
        profile["platform"] = random.choice(PLATFORMS)
        profile["account_age_months"] = random.randint(1, 36)
        profile["verified"] = random.random() < 0.05
        profile["brand_collaborations"] = random.randint(0, 5)
        profile["content_category"] = random.choice(NICHES)
        influencers.append(profile)
    
    # Shuffle the dataset
    random.shuffle(influencers)
    
    df = pd.DataFrame(influencers)
    return df


def generate_brand_dataset(n_brands=50):
    """Generate brand profiles for matching."""
    brands = []
    for i in range(n_brands):
        brand = {
            "brand_id": f"BRD_{i:04d}",
            "brand_name": fake.company(),
            "category": random.choice(BRAND_CATEGORIES),
            "target_audience_age": random.choice(["18-24", "25-34", "35-44", "25-44", "18-34"]),
            "target_audience_gender": random.choice(["Female", "Male", "All"]),
            "budget_tier": random.choice(["Low", "Medium", "High", "Premium"]),
            "preferred_niches": random.sample(NICHES, k=random.randint(1, 4)),
            "preferred_platforms": random.sample(PLATFORMS, k=random.randint(1, 3)),
            "min_followers": random.choice([1000, 5000, 10000, 50000, 100000]),
            "min_engagement_rate": round(random.uniform(1.0, 5.0), 1),
            "values": random.sample([
                "sustainability", "innovation", "luxury", "affordability",
                "authenticity", "diversity", "creativity", "professionalism"
            ], k=random.randint(2, 4)),
            "campaign_goals": random.sample([
                "brand_awareness", "lead_generation", "sales",
                "engagement", "content_creation", "product_launch"
            ], k=random.randint(1, 3)),
        }
        brands.append(brand)
    
    return pd.DataFrame(brands)


if __name__ == "__main__":
    print("Generating influencer dataset...")
    df = generate_dataset(n_influencers=1000)
    df.to_csv("src/data/influencers.csv", index=False)
    print(f"Generated {len(df)} influencer profiles")
    print(f"Authentic: {df['is_authentic'].sum()}, Fake: {(~df['is_authentic']).sum()}")
    
    print("\nGenerating brand dataset...")
    brands_df = generate_brand_dataset(n_brands=50)
    brands_df.to_csv("src/data/brands.csv", index=False)
    print(f"Generated {len(brands_df)} brand profiles")
    
    print("\nDataset generation complete!")
    print(f"\nSample influencer:\n{df.iloc[0].to_dict()}")
