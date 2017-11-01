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


class test_setup_functions(unittest.TestCase):

    def setUp(self):

        Nell = 4
        p = 2
        case = 'cantilever L'
        he = np.ones(Nell) / Nell
        self.Xe_4_2 = Xe_4_2 = get_node_locations_x(Nell, he)
        self.knots_4_2 = knots_4_2 = get_knot_vector(Nell, Xe_4_2, p)
        self.GA_4_2 = GA_4_2 = get_greville_abscissae(self.knots_4_2, p)
        self.ID_4_2 = ID_4_2 = get_id(case, Nell, p)
        self.IEM_4_2 = IEM_4_2 = get_iem(Nell, p)
        self.LM_4_2 = LM_4_2 = get_lm(Nell, p, ID_4_2, IEM_4_2)

        Nell = 10
        p = 3
        he = np.ones(Nell) / Nell
        self.Xe_10_3 = Xe_10_3 = get_node_locations_x(Nell, he)
        self.knots_10_3 = knots_10_3 = get_knot_vector(Nell, Xe_10_3, p)
        self.GA_10_3 = GA_10_3 = get_greville_abscissae(self.knots_10_3, p)
        self.ID_10_3 = ID_10_3 = get_id(case, Nell, p)
        self.IEM_10_3 = IEM_10_3 = get_iem(Nell, p)
        self.LM_10_3 = LM_10_3 = get_lm(Nell, p, ID_10_3, IEM_10_3)

        print GA_10_3
        print knots_10_3
        print Xe_10_3
        print ID_10_3
        print IEM_10_3
        print LM_10_3

    def testing_Xe(self):
        np.testing.assert_equal(self.Xe_4_2, np.array([0., 0.25, 0.5, 0.75, 1.0]))
        np.testing.assert_allclose(self.Xe_10_3, np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                                   rtol=1E-5, atol=1E-5)

    def testing_knots(self):
        np.testing.assert_equal(self.knots_4_2, np.array([0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1.]))
        np.testing.assert_allclose(self.knots_10_3, np.array([0., 0., 0., 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                           1.0, 1.0, 1.0, 1.0]), rtol=1E-5, atol=1E-5)

    def testing_graville_abscissae(self):
        np.testing.assert_equal(self.GA_4_2, np.array([0., 0.125, 0.375, 0.625, 0.875, 1.]))
        np.testing.assert_allclose(self.GA_10_3, np.array([0., 0.03333333, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                           0.966666667, 1.0]), rtol=1E-5, atol=1E-5)

    def testing_id(self):
        np.testing.assert_equal(self.ID_4_2, np.array([1., 2., 3., 4., 0., 0.]))
        np.testing.assert_equal(self.ID_10_3, np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 0., 0., 0.]))

    def testing_iem(self):
        np.testing.assert_equal(self.IEM_4_2, np.array([[1.,  2.,  3.,  4.],
                                                        [ 2.,  3.,  4.,  5.],
                                                        [ 3.,  4.,  5.,  6.]]))
        np.testing.assert_equal(self.IEM_10_3, np.array([[  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.],
                                                         [  2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.],
                                                         [  3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.,  12.],
                                                         [  4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.,  12.,  13.]]))

    def testing_iem(self):
        np.testing.assert_equal(self.LM_4_2, np.array([[1.,  2.,  3.,  4.],
                                                        [ 2.,  3.,  4.,  0.],
                                                        [ 3.,  4.,  0.,  0.]]))
        np.testing.assert_equal(self.LM_10_3, np.array([[  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.],
                                                         [  2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  0.],
                                                         [  3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  0.,  0.],
                                                         [  4.,   5.,   6.,   7.,   8.,   9.,  10.,  0.,  0.,  0.]]))