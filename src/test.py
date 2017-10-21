import unittest

from fem import *

class test_fem_constant(unittest.TestCase):

    def setUp(self):

        Nell = 4
        he = np.ones(Nell) / Nell
        Xe = get_node_locations_x(Nell, he)

        tic = time.time()
        self.K = K = define_stiffness_matrix(Nell, he)
        toc = time.time()
        # print he, Nell, K
        print "Time to define stiffness matrix: %.3f (s)" % (toc - tic)

        ffunc = ffunc_linear
        ffunc_args = np.array([0, 1])

        tic = time.time()
        self.F = F = define_forcing_vector(Nell, he, ffunc=ffunc, ffunc_args=ffunc_args)
        toc = time.time()
        # print F
        # print np.array([1 / 96., 1. / 16., 1 / 8., 3. / 16.])
        print "Time to define forcing vector: %.3f (s)" % (toc - tic)

        tic = time.time()
        self.d = d = solve_for_d(K, F)
        toc = time.time()
        # print d
        # print np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384.])
        print "Time to solve for d: %.3f (s)" % (toc - tic)

        tic = time.time()
        self.u = u = solve_for_displacements(d, Nell, he, g=0)
        toc = time.time()
        print u
        # print np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384., 0])
        print "Time to solve for u(x): %.3f (s)" % (toc - tic)

        # plot_displaccements(u, x, he, Nell, ffunc=ffunc, ffunc_args=ffunc_args)

        self.error = error = quadrature(Xe, he, u, ffunc_args=ffunc_args)
        print error

    # def testingError(self):
    #     self.assertAlmostEquals(self.error, 0.00570544330735, rtol=1E-5, atol=1E-5)

    def testing_error(self):
        np.testing.assert_allclose(self.F, np.array([1 / 96., 1. / 16., 1 / 8., 3. / 16.]), rtol=1E-5, atol=1E-5)
        np.testing.assert_allclose(self.d, np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384.]), rtol=1E-5, atol=1E-5)
        np.testing.assert_allclose(self.u, np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384., 0]), rtol=1E-5, atol=1E-5)
        np.testing.assert_allclose(self.error, 0.003269067718342104, rtol=1E-5, atol=1E-5)
