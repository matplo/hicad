import unittest
import math
import numpy as np
from scipy.integrate import quad
from support import request
from clump_explorer.generator_adapter import core
from clump_explorer.diagnostics import q_inv


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rest = core.sample_rest_protons(12000, .1, core.StreamFactory(request()["realization"]))

    def test_mass_shell_isotropy_and_independent_radial_integral(self):
        p4 = self.rest
        p = np.linalg.norm(p4[:, 1:], axis=1)
        np.testing.assert_allclose(p4[:, 0]**2-p*p, core.PROTON_MASS**2, atol=5e-15)
        direction = p4[:, 1:]/p[:, None]
        np.testing.assert_allclose(direction.mean(axis=0), 0, atol=.018)
        np.testing.assert_allclose((direction**2).mean(axis=0), 1/3, atol=.015)
        m = core.PROTON_MASS
        law = lambda x: x*x*math.exp(-(math.sqrt(m*m+x*x)-m)/.1)
        norm = quad(law, 0, np.inf, epsabs=1e-11)[0]
        for edge in [.2, .4, .6, .9, 1.3]:
            expected = quad(law, 0, edge, epsabs=1e-11)[0]/norm
            self.assertLess(abs(float((p < edge).mean())-expected), .018)
        expected_mean = quad(lambda x: x*law(x), 0, np.inf)[0]/norm
        self.assertLess(abs(p.mean()-expected_mean), .012)

    def test_zero_inverse_boost_and_pair_invariance(self):
        p = self.rest[:100]
        np.testing.assert_array_equal(core.boost_four_vectors(p, [0, 0, 0]), p)
        beta = np.array([.2, -.3, .4])
        boosted = core.boost_four_vectors(p, beta)
        np.testing.assert_allclose(core.boost_four_vectors(boosted, -beta), p, atol=2e-15)
        np.testing.assert_allclose(q_inv(p[:-1], p[1:]), q_inv(boosted[:-1], boosted[1:]), atol=1e-14)

    def test_directions(self):
        rng = np.random.Generator(np.random.PCG64(9))
        v = np.array([core.sample_boost_vector(.5, "isotropic_3d", rng) for _ in range(10000)])
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), .5, atol=1e-15)
        np.testing.assert_allclose((v/.5).mean(axis=0), 0, atol=.018)
        np.testing.assert_allclose(((v/.5)**2).mean(axis=0), 1/3, atol=.015)
        self.assertEqual(core.sample_boost_vector(.5, "transverse", rng)[2], 0)

    def test_empty_singular_and_invalid_inputs(self):
        streams = core.StreamFactory(request()["realization"])
        self.assertEqual(core.sample_rest_protons(0, .1, streams).shape, (0, 4))
        for n, t in [(-1, .1), (1.2, .1), (True, .1), (1, 0), (1, np.nan)]:
            with self.assertRaises(core.ParameterError):
                core.sample_rest_protons(n, t, streams)
        for beta in ([1, 0, 0], [np.nan, 0, 0], [0, 1]):
            with self.assertRaises(core.ParameterError):
                core.boost_four_vectors(self.rest, beta)
        p = np.array([[core.PROTON_MASS, 0, 0, 0]])
        self.assertFalse(core.apply_acceptance(p, request()["clump_config"]["acceptance"])[0])
        self.assertTrue(np.isnan(core.kinematics(p)["eta"][0]))

    def test_truncated_poisson_is_conditioned_not_clipped(self):
        d = core.validate_distribution({"distribution": "shifted_poisson_truncated", "offset": 2,
                                       "mean": 4., "maximum": 5}, "n", 0, 200, True)
        rng = np.random.Generator(np.random.PCG64(5))
        n = np.array([core.draw_distribution(d, rng) for _ in range(20000)])
        prob = np.array([4**k/math.factorial(k) for k in range(4)]);prob /= prob.sum()
        np.testing.assert_allclose(np.bincount(n-2, minlength=4)/len(n), prob, atol=.012)
        with self.assertRaises(core.ParameterError):
            core.validate_distribution({"distribution": "shifted_poisson_truncated", "offset": 0,
                                        "mean": 10000, "maximum": 1}, "n", 0, 200, True)
