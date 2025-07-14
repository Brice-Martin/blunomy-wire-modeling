import numpy as np
import matplotlib.pyplot as plt
import argparse
from sympy import pprint, N, lambdify, Symbol

from wire_modeling.preprocessing import load_point_cloud
from wire_modeling.clustering import cluster_wires
from wire_modeling.fitting import fit_catenary, project_cluster_to_best_plane, generate_catenary_equation_3d, compute_rmse_3d


def process_lidar_file(filepath: str):
    
    """
    Args : 
        filepath (str): .parquet file which contains the point cloud dataset

    Returns :
        Number of clusters
        For each cluster : 
            - Their 3D catenary equations
            - Display the cluster and its calculated 3D catenary 
            - The 3D RMSE between the cluster and its catenary
    """
    # Load the parquet file and determine the clusters
    df = load_point_cloud(filepath)
    labels = cluster_wires(df, eps=0.3, min_samples=10) #Parameters chosen manually so it works on every file be they can be adjusted depending on the situation
    df['cluster'] = labels

    # Check if some points have been classified as noise
    if -1 in labels:
        print("/!\ Some points have not been classified")

    clusters = [label for label in np.unique(labels)]

    print(f"\n{len(clusters)} WIRES DETECTED in file: {filepath}\n")

    # Start of the loop, the calculations will be applied on every cluster (wire)
    for cluster_id in clusters:
        cluster_df = df[df['cluster'] == cluster_id]

        # Calculation of the plane of best fit for each wire
        projected_2d, pca_model = project_cluster_to_best_plane(cluster_df) 
        x = projected_2d[:, 0]
        y = projected_2d[:, 1]

        # Fit catenary
        params, y_fit = fit_catenary(x, y)
        if params is None:
            print(f"Could not fit catenary for cluster {cluster_id}")
            continue

        # Mean for 3D re-alignment
        center_3d = cluster_df[['x', 'y', 'z']].values.mean(axis=0)

        # Determination of the 3D equation
        x_expr, y_expr, z_expr = generate_catenary_equation_3d(
            params, pca_model, center_3d
        )

        # Round to 4 decimals for the equations display:
        x_expr_eq = N(x_expr, 4)
        y_expr_eq = N(y_expr, 4)
        z_expr_eq = N(z_expr, 4)

        # Display the 3D equations
        print(f"\n CLUSTER {cluster_id} :")
        print("-" * 40)
        print(f"=> Number of points : {len(cluster_df)}")
        print(f"=> 3D Catenary Equations:")
        print("X(x) = "); pprint(x_expr_eq, use_unicode=True)
        print("Y(x) = "); pprint(y_expr_eq, use_unicode=True)
        print("Z(x) = "); pprint(z_expr_eq, use_unicode=True)

        # Visualization of the cluster next to the calculated 3D catenary 
        x_sym = Symbol('x')

        # lambdify allows us to evaluate the function on every point along the wire
        cat_fx = lambdify(x_sym, x_expr, 'numpy')
        cat_fy = lambdify(x_sym, y_expr, 'numpy')
        cat_fz = lambdify(x_sym, z_expr, 'numpy')

        x_vals = np.linspace(x.min(), x.max(), 100)
        cat_3d = np.vstack([cat_fx(x_vals), cat_fy(x_vals), cat_fz(x_vals)]).T

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(cluster_df['x'], cluster_df['y'], cluster_df['z'], s=1, label="Original points")
        ax.plot(cat_3d[:, 0], cat_3d[:, 1], cat_3d[:, 2], color='orange', label="Catenary 3D", linewidth=2)
        ax.set_title(f"Cluster {cluster_id} - Fitted Catenary")
        ax.legend()
        plt.show()


        # 3D RMSE Calculation
        cluster_points = cluster_df[['x', 'y', 'z']].values
        rmse_3d = compute_rmse_3d(cluster_points, cat_3d)
        print(f"=> Catenary RMSE (3D)   : {rmse_3d:.4f} meters")
        print("-" * 80)
        print()

def main():
    parser = argparse.ArgumentParser(description="Run wire modeling on a LiDAR .parquet file")
    parser.add_argument("file", help="Path to .parquet file")
    args = parser.parse_args()

    process_lidar_file(args.file)
