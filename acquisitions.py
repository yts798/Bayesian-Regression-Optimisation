import numpy as np
from scipy.stats import norm

# Functional Structure

def probability_improvement(X: np.ndarray, X_sample: np.ndarray,
                            gpr: object, xi: float = 0.01) -> np.ndarray:
    """
    Probability improvement acquisition function.

    Computes the PI at points X based on existing samples X_sample using
    a Gaussian process surrogate model

    Arguments:
    ----------
        X: ndarray of shape (m, d)
            The point for which the expected improvement needs to be computed.

        X_sample: ndarray of shape (n, d)
            Sample locations

        gpr: GPRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.

        xi: float. Default 0.01
            Exploitation-exploration trade-off parameter.

    Returns:
    --------
        PI: ndarray of shape (m,)
    """
    # TODO Q2.4
    # Implement the probability of improvement acquisition function
    mean, std = gpr.predict(X, return_std = True)
    yp = gpr.predict(X_sample, return_std = False)
    if std.all()>0:
        Z = (mean[0] - yp.max() - xi)/std
        return norm.cdf(Z)
    else:
        return 0  
    
    raise NotImplementedError


def expected_improvement(X: np.ndarray, X_sample: np.ndarray,
                         gpr: object, xi: float = 0.01) -> np.ndarray:
    """
    Expected improvement acquisition function.

    Computes the EI at points X based on existing samples X_sample using
    a Gaussian process surrogate model

    Arguments:
    ----------
        X: ndarray of shape (m, d)
            The point for which the expected improvement needs to be computed.

        X_sample: ndarray of shape (n, d)
            Sample locations

        gpr: GPRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.

        xi: float. Default 0.01
            Exploitation-exploration trade-off parameter.

    Returns:
    --------
        EI : ndarray of shape (m,)
    """

    # TODO Q2.4
    # Implement the expected improvement acquisition function

    mean, std = gpr.predict(X, return_std = True)
    yp = gpr.predict(X_sample, return_std = False)
    if std.all()>0:
        Z = (mean[0] - yp.max() - xi)/std
        ip = (mean[0] - yp.max() - xi)
        return ip * norm.cdf(Z) + std * norm.pdf(Z)
    else:
        return 0
    

    raise NotImplementedError
