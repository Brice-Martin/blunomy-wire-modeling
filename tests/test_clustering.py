import numpy as np
import pandas as pd
from wire_modeling.clustering import cluster_wires

def test_cluster_wires_no_noise():
    df = pd.DataFrame({
        'x': np.random.normal(0, 1, 100),
        'y': np.random.normal(0, 1, 100),
        'z': np.random.normal(10, 0.1, 100),
    })
    labels = cluster_wires(df)
    assert len(np.unique(labels)) > 0