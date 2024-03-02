from abc import abstractmethod
from collections import namedtuple
import numpy as np
from inspect import signature
import copy

# Class Structure
# DO NOT CHANGE


class Hyperparameter(namedtuple("Hyperparameter", ("name", "value_type", "bounds"))):
    """
    A kernel hyperparameter's specification in form of a namedtuple.

    Arguments:
    ----------
    name : str
        The name of the hyperparameter. Note that a kernel using a
        hyperparameter with name "x" must have the attributes self.x and
        self.x_bounds

    value_type : str
        The type of the hyperparameter. Currently, only "numeric"
        hyperparameters are supported.

    bounds : pair of floats >= 0
        The lower and upper bound on the parameter. 
    """

    # A raw namedtuple is very memory efficient as it packs the attributes
    # in a struct to get rid of the __dict__ of attributes in particular it
    # does not copy the string for the keys on each instance.
    # By deriving a namedtuple class just to introduce the __init__ method we
    # would also reintroduce the __dict__ on the instance. By telling the
    # Python interpreter that this subclass uses static __slots__ instead of
    # dynamic attributes. Furthermore we don't need any additional slot in the
    # subclass so we set __slots__ to the empty tuple.
    __slots__ = ()

    def __new__(cls, name, value_type, bounds):
        if not isinstance(bounds, str):
            bounds = np.atleast_2d(bounds)
        return super(Hyperparameter, cls).__new__(cls, name, value_type, bounds)


class Kernel:
    """
    Base class for all kernels.
    """

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Arguments:
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns:
        --------
        params : dict
            Parameter names mapped to their values.
        """
        params = dict()

        # introspect the constructor arguments to find the model parameters
        # to represent
        cls = self.__class__
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        init_sign = signature(init)
        args, varargs = [], []
        for parameter in init_sign.parameters.values():
            if parameter.kind != parameter.VAR_KEYWORD and parameter.name != "self":
                args.append(parameter.name)
            if parameter.kind == parameter.VAR_POSITIONAL:
                varargs.append(parameter.name)

        for arg in args:
            params[arg] = getattr(self, arg)

        return params

    def set_params(self, **params):
        """
        Set the parameters of this kernel.

        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            split = key.split("__", 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError(
                        "Invalid parameter %s for kernel %s. "
                        "Check the list of available parameters "
                        "with `kernel.get_params().keys()`." % (name, self)
                    )
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError(
                        "Invalid parameter %s for kernel %s. "
                        "Check the list of available parameters "
                        "with `kernel.get_params().keys()`."
                        % (key, self.__class__.__name__)
                    )
                setattr(self, key, value)
        return self

    def clone_with_theta(self, theta):
        """
        Returns a clone of self with given hyperparameters theta.

        Arguments:
        ----------
        theta : ndarray of shape (n_dims,)
            The hyperparameters
        """
        cloned = copy.deepcopy(self)
        cloned.theta = theta
        return cloned

    @property
    def n_dims(self):
        """Returns the number of hyperparameters of the kernel."""
        return self.theta.shape[0]

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter specifications."""
        r = [
            getattr(self, attr)
            for attr in dir(self)
            if attr.startswith("hyperparameter_")
        ]
        return r

    @property
    def theta(self):
        """
        Returns the (flattened, log-transformed) hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns:
        --------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        theta = []
        params = self.get_params()
        for hyperparameter in self.hyperparameters:
            theta.append(params[hyperparameter.name])
        if len(theta) > 0:
            return np.log(np.hstack(theta))
        else:
            return np.array([])

    @theta.setter
    def theta(self, theta):
        """
        Sets the (flattened, log-transformed) hyperparameters.

        Arguments:
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        params = self.get_params()
        i = 0
        for hyperparameter in self.hyperparameters:
            params[hyperparameter.name] = np.exp(theta[i])
            i += 1

        if i != len(theta):
            raise ValueError(
                "theta has not the correct number of entries."
                " Should be %d; given are %d" % (i, len(theta))
            )
        self.set_params(**params)

    @property
    def bounds(self):
        """
        Returns the log-transformed bounds on the theta.

        Returns:
        --------
        bounds : ndarray of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        bounds = [
            hyperparameter.bounds
            for hyperparameter in self.hyperparameters]
        if len(bounds) > 0:
            return np.log(np.vstack(bounds))
        else:
            return np.array([])

    @abstractmethod
    def __call__(self, X, Y=None):
        """Evaluate the kernel."""
