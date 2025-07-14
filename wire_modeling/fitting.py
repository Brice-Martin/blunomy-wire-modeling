import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree


def catenary_function(x, c, x0, y0):
    return c * np.cosh((x - x0) / c) + y0


def project_cluster_to_best_plane(df_cluster: pd.DataFrame) -> tuple[np.ndarray, PCA]:
    """
    Projects a 3D wire cluster on its best-fit 2D plane using PCA.

    Parameters:
        df_cluster (pd.DataFrame): DataFrame with ['x', 'y', 'z'] columns

    Returns:
        tuple:
            - np.ndarray: 2D projected points (shape: [n_points, 2])
            - PCA: fitted PCA object (can be used to transform back to 3D)
    """

    # We center the points because PCA applies on (0,0,0) 
    coords = df_cluster[['x', 'y', 'z']].values
    coords_centered = coords - coords.mean(axis=0)

    pca = PCA(n_components=3)
    pca.fit(coords_centered)

    # Keep only the projection in the best-fit plane (first 2 components)
    projected_2d = pca.transform(coords_centered)[:, :2]

    return projected_2d, pca


def fit_catenary(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Fits a catenary curve to 2D projected points.

    Args:
        x (np.ndarray): x-coordinates in the PCA plane
        y (np.ndarray): y-coordinates in the PCA plane

    Returns:
        tuple:
            - params (np.ndarray): [c, x0, y0] parameters of the fitted catenary
            - y_fit (np.ndarray): predicted y values for each x
    """

    # Parameters initialization : 
    c_init = (np.max(y) - np.min(y)) / 2
    x0_init = np.mean(x)
    y0_init = np.min(y)

    p0 = [c_init, x0_init, y0_init]

    # We determine the parameters of the 2D catenary function
    try:
        params, _ = curve_fit(catenary_function, x, y, p0=p0)
        y_fit = catenary_function(x, *params)
        return params, y_fit
    except RuntimeError:
        print("/!\ Catenary fit failed")
        return None, None



def generate_catenary_equation_3d(params, pca_model, mean):

    """
    Generates a 3D vector equation of the fitted catenary using the PCA components.

    Parameters:
        params (list): [a, x0, y0] parameters of the 2D fitted catenary
        pca_model (PCA): PCA object fitted on the 3D cluster
        mean (np.ndarray): 3D centroid of the cluster (shape: (3,))

    Returns:
        tuple of sympy expressions (x_expr, y_expr, z_expr)
    """

    # Extraction of the parameters
    c, x0, y0 = params
    u1 = pca_model.components_[0]  # direction along the wire
    u2 = pca_model.components_[1]  # direction of the sag
    a = mean

    # Symbolic variable
    x = sp.Symbol('x')

    # 2D catenary
    x0 = sp.N(x0, 4)
    c = sp.N(c, 4)
    y0 = sp.N(y0, 4)
    f_x = sp.N(c * sp.cosh((x - x0) / c) + y0)

    # 3D vector equations
    x_expr = a[0] + x * u1[0] + f_x * u2[0]
    y_expr = a[1] + x * u1[1] + f_x * u2[1]
    z_expr = a[2] + x * u1[2] + f_x * u2[2]

    return x_expr, y_expr, z_expr


def compute_rmse_3d(cluster_points: np.ndarray, catenary_points: np.ndarray) -> float:

    """
    Computes the RMSE between original 3D cluster points and the fitted catenary curve in 3D.
    
    Parameters:
        cluster_points (np.ndarray): Original 3D points (N, 3)
        catenary_points (np.ndarray): Fitted 3D curve points (M, 3)
    
    Returns:
        float: RMSE (in the same unit as input, probably meters)
    """

    # For each point of the original cluster, we find the closest point on the adjusted catenary. Then we use that distance
    # to calculate the RMSE
    tree = cKDTree(catenary_points)
    distances, _ = tree.query(cluster_points)
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse
    

