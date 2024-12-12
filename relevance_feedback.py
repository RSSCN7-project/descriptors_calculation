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
            json.dump(self.feedback_history, f, indent=4)

    def update_weights(self, query_descriptors, feedback_data):
        """
        Update feature weights based on user feedback.
        :param query_descriptors: Descriptors of the query image.
        :param feedback_data: List of feedback for images with their descriptors.
        :return: Updated feature weights.
        """
        current_weights = self.default_weights.copy()

        # Separate feedback into relevant, neutral, and irrelevant
        relevant_images = [item for item in feedback_data if item['feedback'] == 'relevant']
        neutral_images = [item for item in feedback_data if item['feedback'] == 'neutral']
        irrelevant_images = [item for item in feedback_data if item['feedback'] == 'irrelevant']

        # If no feedback is provided, return default weights
        if not relevant_images and not irrelevant_images:
            return current_weights

        # Compute feature variations using irrelevant feedback
        feature_variations = {}
        for feature in self.default_weights.keys():
            feature_variations[feature] = self._compute_feature_variation(relevant_images, irrelevant_images, feature)

        # Normalize and update weights based on irrelevant feedback
        total_variation = sum(abs(var) for var in feature_variations.values())
        if total_variation > 0:
            for feature, variation in feature_variations.items():
                # Reduce weights influenced by irrelevant feedback
                adjustment = -(variation / total_variation) if variation < 0 else 0
                current_weights[feature] = max(0.01, current_weights[feature] + adjustment)

        # Ensure weights remain normalized
        total_weight = sum(current_weights.values())
        current_weights = {k: v / total_weight for k, v in current_weights.items()}

        return current_weights

    def _compute_feature_variation(self, relevant_images, irrelevant_images, feature_name):
        """
        Compute variation of a specific feature between relevant and irrelevant images.

        :param relevant_images: List of relevant image descriptors.
        :param irrelevant_images: List of irrelevant image descriptors.
        :param feature_name: Name of the feature to analyze.
        :return: Variation score.
        """
        def compute_mean_feature(images):
            """
            Compute the mean feature vector for a list of images.

            :param images: List of image descriptors.
            :return: Mean feature vector or None if no valid features are found.
            """
            features = [
                np.array(img['descriptors'][feature_name])
                for img in images if 'descriptors' in img and feature_name in img['descriptors']
            ]

            if len(features) == 0:
                return None

            # Attempt to stack and compute mean
            try:
                features = np.array(features)
                if features.ndim == 2:  # Ensure it's a 2D array
                    return np.mean(features, axis=0)
            except ValueError:
                print(f"Inconsistent descriptor shapes for feature: {feature_name}")
            return None

        # Compute means for relevant and irrelevant images
        relevant_mean = compute_mean_feature(relevant_images)
        irrelevant_mean = compute_mean_feature(irrelevant_images)

        # If no valid data is found, return 0
        if relevant_mean is None or irrelevant_mean is None:
            return 0

        # Compute variation (norm of the difference between means)
        return np.linalg.norm(relevant_mean - irrelevant_mean)
