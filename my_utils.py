import numpy as np


def haversine_distance(lat1, lon1, lat2_col, lon2_col):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2_col, lon2_col = np.radians(lat2_col), np.radians(lon2_col)

    # Haversine formula
    dlat = lat2_col - lat1
    dlon = lon2_col - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2_col) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371  # Earth radius in kilometers
    d = R * c

    return d


def max_min_norm(data, logger=None):
    # Min-Max Normalization
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    if logger:
        logger.info(f"Min values: {min_vals.tolist()}, Max values: {max_vals.tolist()}")
    with np.errstate(invalid='ignore'):
        norm_data = np.where(max_vals - min_vals != 0, (data - min_vals) / (max_vals - min_vals), 0)
    return norm_data
