import numpy as np
from wire_modeling.fitting import fit_catenary

def test_fit_catenary_outputs():
    x = np.linspace(-10, 10, 100)
    y = 2 * np.cosh((x - 1) / 2) + 5

    params, y_fit = fit_catenary(x, y)
    
    assert params is not None
    assert len(params) == 3
    assert y_fit.shape == y.shape

