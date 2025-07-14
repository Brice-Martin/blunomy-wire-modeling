import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def cluster_wires(df: pd.DataFrame, eps=0.5, min_samples=10) -> np.ndarray:
    """
    Clusters LiDAR points into wires using DBSCAN.
    1 - We select the coords 'x' & 'y' since the wires are mostly organized in those planes
    2 - We apply a PCA on the points centered on their gravity center
    3 - We find the angle between the 1st component of the PCA and the x-axis and rotate the wires 
    4 - The points now appear as separated horizontal bands
    5 - We apply DBSCAN on that 2D vizualisation (after standardizing the dimensions)

    Args:
        df (pd.DataFrame): Input point cloud with ['x', 'y', 'z'] columns
        eps (float): Max distance between two samples for them to be considered in the same neighborhood
        min_samples (int): Minimum number of points to form a dense region

    Returns:
        np.ndarray: Array of cluster labels (warning : -1 means noise)
    """

    # We center the points because PCA applies on (0,0) 
    points = df[['x', 'y']].values
    center = points.mean(axis=0)
    points_centered = points - center

    pca = PCA(n_components=2)
    pca.fit(points_centered)

    # Firt principal axis => wires "direction"
    angle_rad = np.arctan2(pca.components_[0][1], pca.components_[0][0])
    angle_deg = np.degrees(angle_rad) # just in case

    # Reverse the angle to redress (negative rotation)
    theta = -angle_rad
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    points_rotated = points_centered @ rotation_matrix.T

    coords_scaled = StandardScaler().fit_transform(points_rotated) # To be able to compare the distances in the different dimensions

    #Finally, we apply DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(coords_scaled)

    return labels