!pip install pandas scikit-learn==1.2.2
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class BehaviorRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity_df = None
        
    def fit(self, interactions_df):
        """
        Build the recommendation profiles from raw user interaction data.
        Expects a DataFrame with columns: ['user_id', 'item_name', 'rating']
        """
        print("⚙️ Processing interaction behavior data...")
        # Pivot the data into a User-Item interaction matrix
        # Fill NaN values with 0, assuming no interaction/neutral stance
        self.user_item_matrix = interactions_df.pivot_table(
            index='user_id', 
            columns='item_name', 
            values='rating'
        ).fillna(0)
        
        # Calculate Cosine Similarity between user behavioral vectors
        similarity_matrix = cosine_similarity(self.user_item_matrix)
        
        # Cast matrix into a queryable DataFrame structured with User IDs
        self.user_similarity_df = pd.DataFrame(
            similarity_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        print("✅ Behavioral affinity matrix calculated successfully.")

    def get_recommendations(self, target_user, num_suggestions=2):
        """
        Generate personalized content recommendations for a target user.
        """
        if target_user not in self.user_item_matrix.index:
            return f"❌ User ID '{target_user}' not found in training dataset."
            
        # 1. Grab similar users ranked by alignment score (exclude the target user themselves)
        similar_users = self.user_similarity_df[target_user].sort_values(ascending=False).drop(target_user)
        
        # 2. Identify items the target user hasn't interacted with yet (value is 0)
        target_user_profile = self.user_item_matrix.loc[target_user]
        unseen_items = target_user_profile[target_user_profile == 0].index
        
        item_scores = {}
        
        # 3. Calculate weighted score for unseen items based on similar user interactions
        for item in unseen_items:
            total_similarity = 0
            weighted_rating_sum = 0
            
            for other_user, similarity_score in similar_users.items():
                # Get the other user's rating for this specific item
                other_user_rating = self.user_item_matrix.loc[other_user, item]
                
                if other_user_rating > 0:
                    total_similarity += similarity_score
                    weighted_rating_sum += other_user_rating * similarity_score
            
            # Normalize scores to prevent volume bias
            if total_similarity > 0:
                item_scores[item] = weighted_rating_sum / total_similarity
            else:
                item_scores[item] = 0
                
        # Sort items based on predicted interest scores
        sorted_recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:num_suggestions]

# =====================================================================
# Main Execution / Live Simulation Lab
# =====================================================================
if __name__ == "__main__":
    # Mocking historical user interaction records (e.g., Movie streaming behaviors)
    raw_data = {
        'user_id': [
            'User_A', 'User_A', 'User_A',
            'User_B', 'User_B', 'User_B',
            'User_C', 'User_C', 'User_C',
            'User_D', 'User_D'
        ],
        'item_name': [
            'Inception', 'The Matrix', 'Interstellar',
            'Inception', 'The Matrix', 'Toy Story',
            'Toy Story', 'Shrek', 'The Matrix',
            'Interstellar', 'Inception'
        ],
        # Rating behavior scale from 1 to 5 stars
        'rating': [5, 4, 5, 4, 5, 2, 5, 4, 2, 4, 3]
    }
    
    df = pd.DataFrame(raw_data)
    
    # Initialize and train our pipeline engine
    engine = BehaviorRecommender()
    engine.fit(df)
    
    # Target profile: User_B loves Sci-Fi (Inception/Matrix) but hasn't seen Interstellar
    target = 'User_B'
    recommendations = engine.get_recommendations(target_user=target, num_suggestions=2)
    
    print(f"\n📈 Personalization Output for: {target}")
    print("-" * 45)
    for index, (item, score) in enumerate(recommendations, 1):
        print(f"{index}. Suggested Content: '{item}' | Predicted Affinity Score: {score:.2f}/5.0")