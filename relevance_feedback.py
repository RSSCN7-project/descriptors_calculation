import numpy as np
import json
import os

class RelevanceFeedbackManager:
    def __init__(self):
        # Default feature weights
        self.default_weights = {
            "histogram": 0.3,
            "dominant_colors": 0.1,
            "gabor_descriptors": 0.2,
            "hu_moments": 0.1,
            "texture_energy": 0.395,
            "circularity": 0.005
        }
        
        # Feedback file to persist learning
        self.feedback_file = 'user_feedback.json'
        self.feedback_history = self.load_feedback_history()
    
    def load_feedback_history(self):
        """Load previous feedback history."""
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_feedback_history(self):
        """Save feedback history to file."""
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_history, f)
    
    def update_weights(self, query_descriptors, feedback_data):
        current_weights = self.default_weights.copy()
        
        relevant_images = [item for item in feedback_data if item['feedback'] == 'relevant']
        irrelevant_images = [item for item in feedback_data if item['feedback'] == 'irrelevant']
        
        if not relevant_images and not irrelevant_images:
            return current_weights
        
        # Si l'une des listes est vide, retournez les poids par défaut
        if not relevant_images or not irrelevant_images:
            return current_weights
        
        feature_variations = {
            "histogram": self._compute_feature_variation(relevant_images, irrelevant_images, "histogram"),
            "dominant_colors": self._compute_feature_variation(relevant_images, irrelevant_images, "dominant_colors"),
            "gabor_descriptors": self._compute_feature_variation(relevant_images, irrelevant_images, "gabor_descriptors"),
            "hu_moments": self._compute_feature_variation(relevant_images, irrelevant_images, "hu_moments"),
            "texture_energy": self._compute_feature_variation(relevant_images, irrelevant_images, "texture_energy"),
            "circularity": self._compute_feature_variation(relevant_images, irrelevant_images, "circularity")
        }
        
        # Normalisez et mettez à jour les poids
        total_variation = sum(abs(var) for var in feature_variations.values())
        if total_variation > 0:
            for feature, variation in feature_variations.items():
                current_weights[feature] = max(0.01, current_weights[feature] + (variation / total_variation))
        
        # Normalisez les poids pour qu'ils s'additionnent à 1
        total = sum(current_weights.values())
        current_weights = {k: v/total for k, v in current_weights.items()}
        
        return current_weights


    
    def _compute_feature_variation(self, relevant_images, irrelevant_images, feature_name):
        """
        Compute variation of a specific feature between relevant and irrelevant images.
        
        :param relevant_images: List of relevant image descriptors
        :param irrelevant_images: List of irrelevant image descriptors
        :param feature_name: Name of the feature to analyze
        :return: Variation score
        """
        if not (relevant_images or irrelevant_images):
            return 0
        
        def compute_mean_feature(images):
            if not images:
                return None
            features = [np.array(img['descriptors'][feature_name]) for img in images if 'descriptors' in img]
            if not features:  # Vérifie si la liste des caractéristiques est vide
                return None
            return np.mean(features, axis=0)


        relevant_mean = compute_mean_feature(relevant_images)
        irrelevant_mean = compute_mean_feature(irrelevant_images)
        
        # If no data, return 0
        if relevant_mean is None or irrelevant_mean is None:
            return 0
        
        # Compute variation (difference between means)
        variation = np.linalg.norm(relevant_mean - irrelevant_mean)
        
        return variation