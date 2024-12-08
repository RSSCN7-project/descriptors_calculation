import numpy as np
from scipy.spatial import distance
import cv2

def bhattacharyya_distance(hist1, hist2):
    return sum(cv2.compareHist(np.array(hist1[color], dtype=np.float32),
                               np.array(hist2[color], dtype=np.float32),
                               cv2.HISTCMP_BHATTACHARYYA) for color in ("b", "g", "r"))

def dominant_color_distance(colors1, colors2):
    return distance.cdist(np.array(colors1), np.array(colors2), 'euclidean').mean()

def gabor_distance(gabor1, gabor2):
    return distance.euclidean(np.array(gabor1), np.array(gabor2))

def hu_moments_distance(hu1, hu2):
    return distance.euclidean(np.array(hu1), np.array(hu2))

def texture_energy_distance(te1, te2):
    return np.linalg.norm(np.array(te1) - np.array(te2))

def circularity_distance(ci1, ci2):
    return np.linalg.norm(np.array(ci1) - np.array(ci2))

def compute_similarity_score(query_descriptors, database_descriptors):
    """
    Compute similarity score between two sets of descriptors.
    Lower score means more similar.
    """
    hist_dist = bhattacharyya_distance(query_descriptors["histogram"], database_descriptors["histogram"])
    color_dist = dominant_color_distance(query_descriptors["dominant_colors"], database_descriptors["dominant_colors"])
    gabor_dist = gabor_distance(query_descriptors["gabor_descriptors"], database_descriptors["gabor_descriptors"])
    hu_dist = hu_moments_distance(query_descriptors["hu_moments"], database_descriptors["hu_moments"])
    te_dist = texture_energy_distance(query_descriptors["texture_energy"], database_descriptors["texture_energy"])
    ci_dist = circularity_distance(query_descriptors["circularity"], database_descriptors["circularity"])

    # Weighted combination of distances
    # Lower total distance means more similar
    return 0.3 * hist_dist + 0.2 * color_dist + 0.2 * gabor_dist + \
           0.1 * hu_dist + 0.1 * te_dist + 0.1 * ci_dist