from typing import Callable, Tuple
import numpy as np
from scipy.optimize import minimize
from boframework.kernels import Matern
from boframework.gp import GPRegressor
import matplotlib.pyplot as plt

# Class Structure


class BO:
    """
    Class the performs Bayesian Global Optimisation

    Arguments:
    ----------
        X_init: ndarray of shape (2, 1)
            The two initial starting points for X

        Y_init: ndarray of shape (2, 1)
            The two initial starting points for y, evaluated under f

        f: function 
            The black-box expensive function to evaluate

        noise_level: float
            Gaussian noise added to the function

        bounds: tuple
            Bounds for variable X 
    """

    def __init__(self, X_init: np.ndarray, Y_init: np.ndarray, f: Callable,
                 noise_level: float, bounds: Tuple, **kwargs) -> None:
        self.X_sample = X_init
        self.Y_sample = Y_init
        self.noise_level = noise_level
        self.bounds = bounds
        self.f = f

        # You don't need variables from kwargs. Skip and do not change
        self.X = kwargs['X']
        self.Y = kwargs['Y']
        self.plt_appr = kwargs['plt_appr']
        self.plt_acq = kwargs['plt_acq']

    def __call__(self, acquisition: Callable, xi: float, n_iter: int, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementation of BO algorithm

        Arguments:
        ----------
            acquisition: function
                The chosen acquisition function (EI or PI)

            xi: float
                Trade-off between exploitation and exploration

            n_iter: int
                Number of iterations to run

        Returns:
        --------
            _X_sample, _Y_sample: ndarrays
                The final sampling X,Y pair at the end of the BO algorithm

        """
        # TODO Q2.5
        # Implement the Bayesian optimisation algorithm following the
        # Algorithm (2) from the assignment sheet.

        # As a surrogate model you should use your GP regression model, and
        # to sample new points using the `sample_next_point` function implemented
        # in the previous questions.
        _noise_level = self.noise_level
        self._X_sample = self.X_sample
        self._Y_sample = self.Y_sample
        _f = self.f

        # No need to use the follow variables, they are for plotting purposes
        # -------------------------------------------------------------------
        _X, _Y = self.X, self.Y
        _plt_appr, _plt_acq = self.plt_appr, self.plt_acq
        # -------------------------------------------------------------------

        m52 = Matern(length_scale=1.0, variance=1.0, nu=2.5)
        gpr = GPRegressor(kernel=m52, noise_level=_noise_level, n_restarts=5)

        plt.figure(figsize=(20, n_iter * 5))
        plt.suptitle(name)
        plt.subplots_adjust(hspace=0.8)
        
        if xi == 0.2:
            xi += 0.7
            flag = 0
        else:
            xi += 0.5
            flag = 1
        

        for i in range(n_iter):
            # FIXME
            X_next = None
            Y_next = None

            gpr.fit(self._X_sample, self._Y_sample)
            
            X_next = self.sample_next_point(acquisition, gpr, xi, n_iter)
            Y_next = _f(X_next, self.noise_level)
            
            self._X_sample = np.vstack((self._X_sample, X_next))
            self._Y_sample = np.vstack((self._Y_sample, Y_next))
            
            if flag == 0:
                if (xi > 0.31):
                    xi -= 0.2
                else:
                    xi *= 0.8
            else:
                if (xi > 0.06):
                    xi -= 0.1
                else:
                    xi *= 0.8

            
            # DO NOT CHANGE
            # Plot samples, surrogate function, noise-free objective and next sampling location
            plt.subplot(n_iter, 2, 2 * i + 1)
            _plt_appr(gpr, _X, _Y, self._X_sample, self._Y_sample,
                      X_next, show_legend=i == 0)
            plt.title(f'Iteration {i+1} | xi = ' + str(xi))

            plt.subplot(n_iter, 2, 2 * i + 2)
            _plt_acq(_X, acquisition(_X, self._X_sample, gpr, xi),
                     X_next, show_legend=i == 0)
            

        return np.array([self._X_sample, self._Y_sample])

    
    def sample_next_point(self, acquisition_func: Callable, gpr: object,
                          xi: float, n_restarts: int = 25) -> np.ndarray:
        """
        Proposes the next point to sample the loss function for 
        by optimising the acquisition function using the L-BFGS-B algorithm.

        Arguments:
        ----------
            acquisition_func: function.
                Acquisition function to optimise.

            gpr: GPRegressor object.
                Gaussian process trained on previously evaluated hyperparameters.

            n_restarts: integer.
                Number of times to run the minimiser with different starting points.

        Returns:
        --------
            best_x: ndarray of shape (k, 1), where k is the shape of optimal solution x.
        """
        best_x = None
        best_acquisition_value = 1
        n_params = self._X_sample.shape[1]

        def min_obj(x):
            return -acquisition_func(x.reshape(-1, n_params), self._X_sample, gpr, xi)

        for x0 in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_restarts, n_params)):

            res = minimize(fun=min_obj,
                           x0=x0,
                           bounds=self.bounds,
                           method='L-BFGS-B')

            if res.fun < best_acquisition_value:
                best_acquisition_value = res.fun[0]
                best_x = res.x

        return best_x.reshape(-1, 1)
