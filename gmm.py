"""Gaussian Mixture Model song recommendation."""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_RECOMMENDATION_COUNT
from src.data_handling.features import find_song_index, prepare_feature_matrix


class GMMRecommender:
    """Recommend songs using Gaussian mixture components."""
