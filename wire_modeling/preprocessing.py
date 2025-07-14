import pandas as pd

def load_point_cloud(file_path: str) -> pd.DataFrame:
    """
    Load a LiDAR point cloud with a .parquet file.

    Parameters:
        file_path (str): path to the .parquet file

    Returns:
        pd.DataFrame: DataFrame with x, y, z as columns
    """
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Error while loading the file : {e}")