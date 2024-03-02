import numpy as np
import boframework.gp as gp
import boframework.kernels as kernels
import boframework.acquisitions as acquisitions
import boframework.bayesopt as bayesopt
import unittest


class TestQuestion2_1(unittest.TestCase):
    def test_matern32_X(self):
        kernel = kernels.Matern(nu=1.5)
        X = np.array([[-0.5], [2.2]])
        ret = kernel(X)
        true = np.array([[1.0, 0.05285537943757422],
                        [0.05285537943757422, 1.0]])
        self.assertEqual(ret.shape, true.shape)
        self.assertTrue(np.allclose(ret, true))

    def test_matern52_X(self):
        kernel = kernels.Matern(nu=2.5)
        X = np.array([[-0.5], [2.2]])
        ret = kernel(X)
        true = np.array([[1.0, 0.04581560233508662],
                        [0.04581560233508662, 1.0]])
        self.assertEqual(ret.shape, true.shape)
        self.assertTrue(np.allclose(ret, true))

    def test_matern32_XY(self):
        kernel = kernels.Matern(nu=1.5)
        X = np.array([[-0.5], [2.2]])
        Y = np.array([[2.2], [-0.5]])
        ret = kernel(X, Y)
        true = np.array([[0.05285537943757422, 1.0],
                        [1.0, 0.05285537943757422]])
        self.assertEqual(ret.shape, true.shape)
        self.assertTrue(np.allclose(ret, true))

    def test_matern52_XY(self):
        kernel = kernels.Matern(nu=2.5)
        X = np.array([[-0.5], [2.2]])
        Y = np.array([[2.2], [-0.5]])
        ret = kernel(X, Y)
        true = np.array([[0.04581560233508662, 1.0],
                        [1.0, 0.04581560233508662]])
        self.assertEqual(ret.shape, true.shape)
        self.assertTrue(np.allclose(ret, true))


class TestQuestion2_2(unittest.TestCase):
    def test_log_marginal_likelihood(self):
        theta = np.array([0.95006168, -0.28380026])
        kernel = kernels.Matern(
            length_scale=2.5858691463324135, variance=0.7529170156780511, nu=2.5)
        gpr = gp.GPRegressor(kernel=kernel, noise_level=0.1)
        gpr._kernel = kernel
        gpr._X_train = np.array([[-0.5], [2.2]])
        gpr._y_train = np.array([[-1.31584035], [-0.09311054]])
        ll_ret = gpr.log_marginal_likelihood(theta)
        ll_true = -2.7580272769922836
        self.assertAlmostEqual(ll_ret, ll_true)

    def test_predict(self):
        # assumed we already stored training data and fit model for parameters
        kernel = kernels.Matern(
            length_scale=1.4153126377384697, variance=0.7700529492001186, nu=2.5)
        gpr = gp.GPRegressor(kernel=kernel, noise_level=0.1)
        gpr._kernel = kernel
        gpr._X_train = np.array([[-0.5], [2.2]])
        gpr._y_train = np.array([[-1.31584035], [-0.09311054]])
        # input X
        X = np.array([[-0.5], [2.2]])
        mu_ret, sigma_ret = gpr.predict(X, return_std=True)
        mu_true = np.array([[-1.16308135], [-0.10392013]])
        sigma_true = np.array([[0.29710927], [0.29710927]])
        self.assertTrue(np.allclose(mu_ret, mu_true))
        self.assertTrue(np.allclose(sigma_ret, sigma_true))


class TestQuestion2_4(unittest.TestCase):
    def test_pi(self):
        kernel = kernels.Matern(
            length_scale=1.4153126377384697, variance=0.7700529492001186, nu=2.5)
        gpr = gp.GPRegressor(kernel=kernel, noise_level=0.1)
        gpr._kernel = kernel
        gpr._X_train = np.array([[-0.5], [2.2]])
        gpr._y_train = np.array([[-1.31584035], [-0.09311054]])
        xi = 1
        X = np.array([[0.42291128]])
        X_sample = np.array([[-0.5], [2.2]])
        ret = acquisitions.probability_improvement(X, X_sample, gpr, xi)
        true = np.array([0.00157985])
        self.assertEqual(ret.shape, true.shape)
        self.assertTrue(np.allclose(ret, true))

    def test_ei(self):
        kernel = kernels.Matern(
            length_scale=1.4153126377384697, variance=0.7700529492001186, nu=2.5)
        gpr = gp.GPRegressor(kernel=kernel, noise_level=0.1)
        gpr._kernel = kernel
        gpr._X_train = np.array([[-0.5], [2.2]])
        gpr._y_train = np.array([[-1.31584035], [-0.09311054]])
        xi = 0.8
        X = np.array([[3.04862203]])
        X_sample = np.array([[-0.5], [2.2]])
        ret = acquisitions.expected_improvement(X, X_sample, gpr, xi)
        true = np.array([0.03620532])
        self.assertEqual(ret.shape, true.shape)
        self.assertTrue(np.allclose(ret, true))


class TestSampling(unittest.TestCase):
    def test_sample_next_point(self):
        noise_level = 0.1
        bounds = np.array([[-2.7, 6]])
        acquisition = acquisitions.probability_improvement
        xi = 1

        def f(X, noise_level=noise_level):
            return np.sin(X) + np.sin(2 * X) + noise_level * np.random.randn(*X.shape)

        m52 = kernels.Matern(length_scale=1.0, variance=1.0, nu=2.5)
        gpr = gp.GPRegressor(kernel=m52, noise_level=noise_level, n_restarts=5)
        gpr._kernel = m52
        gpr._X_train = np.array([[-0.5], [2.2], [5.99650841], [3.73424792]])
        gpr._y_train = np.array(
            [[-1.3158403526978055], [-0.09311053674614302], [-0.7164688995465633], [0.2453150501692566]])
        bayes_opt = bayesopt.BO(
            None, None, f, noise_level, bounds, X=None, Y=None, plt_appr=None, plt_acq=None)
        bayes_opt._X_sample = np.array(
            [[-0.5], [2.2], [5.99650841], [3.73424792]])
        ret = bayes_opt.sample_next_point(acquisition, gpr, xi)
        true = np.array([[-2.7]])
        self.assertEqual(ret.shape, true.shape)
        self.assertTrue(np.allclose(ret, true))


if __name__ == '__main__':
    unittest.main()
