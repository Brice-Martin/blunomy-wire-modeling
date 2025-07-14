# blunomy-wire-modeling

### Description
This project provides a full pipeline to detect and model overhead electric wires as 3D catenary curves using LiDAR point clouds. It is modular, tested, and structured as a Python package for reproducibility and reuse.


### What is the package doing
Given a .parquet LiDAR file (e.g., from a drone scan), the pipeline:
1. Loads and preprocesses the 3D point cloud
2. Clusters wire points using PCA + DBSCAN
3. For each wire cluster:
    - Projects points onto its best-fit 2D plane (PCA)
    - Fits a catenary curve
    - Reconstructs the wire as a 3D parametric curve
    - Prints the 3D vector equations
    - Plots the original points and fitted catenary curve
    - Calculates the RMSE between original 3D cluster points and the fitted catenary curve in 3D


### Explaination of the 3D Catenary Equation
In 3D, each wire is modeled as a vector function.

Each of the three functions describes the real spatial X, Y, Z coordinates of a point on the wire depending on a 1D parameter x. This x parameter follows the direction of the wire (main PCA axis) in the plane of best fit.


### File Structure 
wire_modeling/
    preprocessing.py        # Loads .parquet point cloud
    clustering.py           # Clustering wires (DBSCAN, PCA)
    fitting.py              # Plane of best fit, Catenary fitting, 3D equation generation, RMSE calculation
scripts/
    run_pipeline.py         # Makes the pipeline callable via process_lidar_file()
tests/                      # To keep ensuring that the functions are working
notebooks/
    01_exploration.ipynb    # For all the tests along the creation of the package
requirements.txt            # Gives all the necessary packages
setup.py                    
README.md


### Installation

Clone the repository and install the package locally : 
(I advise you to create a virtual environnement to do so)

Here are the commands you have to use : 
    1. clone the repository, you can use : git clone <blunomy-wire-modeling_url> (use the url of the repository)
    2. Place yourself in the repository : cd blunomy-wire-modeling
    3. Install the package : pip install -e .
    4. Try your new installed package : run-wire-modeling data/lidar_cable_points_extrahard.parquet

    Note1: You can change the second part of that last command if the point cloud .parquet file you want to use it on is not at that place.
    Note2: You can change the "extrahard" part of the last command to try the package on 3 others pre downloaded files ("easy", "medium", "hard")


The required libraries should be installed automatically with the package, but, if not, you can install them manually using :

    pip install -r requirements.txt


### How to use the package on .parquet files

You can reuse this package as a Python module:

    from wire_modeling.run_pipeline import process_lidar_file
    process_lidar_file("link_to_.parquet_file")

It will:

    - Detect wires (clusters) and give you their amount
    - Plot 3D curve for each
    - Print 3D equations with rounded values for each
    - Print calculated RMSE between original 3D cluster points and the fitted catenary curve


### Limitation 

Works very well when the wires à parallel or almost parallel to each others but I think it would require some changes if it is not the case.