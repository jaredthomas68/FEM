import unittest

from fem import *

# class test_fem_constant(unittest.TestCase):
#
#     def setUp(self):
#
#         Nell = 4
#         he = np.ones(Nell) / Nell
#         Xe = get_node_locations_x(Nell, he)
#
#         tic = time.time()
#         self.K = K = define_stiffness_matrix(Nell, he)
#         toc = time.time()
#         # print he, Nell, K
#         print "Time to define stiffness matrix: %.3f (s)" % (toc - tic)
#
#         ffunc = ffunc_linear
#         ffunc_args = np.array([0, 1])
#
#         tic = time.time()
#         self.F = F = define_forcing_vector(Nell, he, ffunc=ffunc, ffunc_args=ffunc_args)
#         toc = time.time()
#         # print F
#         # print np.array([1 / 96., 1. / 16., 1 / 8., 3. / 16.])
#         print "Time to define forcing vector: %.3f (s)" % (toc - tic)
#
#         tic = time.time()
#         self.d = d = solve_for_d(K, F)
#         toc = time.time()
#         # print d
#         # print np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384.])
#         print "Time to solve for d: %.3f (s)" % (toc - tic)
#
#         tic = time.time()
#         self.u = u = solve_for_displacements(d, Nell, he, g=0)
#         toc = time.time()
#         print u
#         # print np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384., 0])
#         print "Time to solve for u(x): %.3f (s)" % (toc - tic)
#
#         # plot_displaccements(u, x, he, Nell, ffunc=ffunc, ffunc_args=ffunc_args)
#
#         self.error = error = quadrature(Xe, he, u, ffunc_args=ffunc_args)
#         print error
#
#     # def testingError(self):
#     #     self.assertAlmostEquals(self.error, 0.00570544330735, rtol=1E-5, atol=1E-5)
#
#     def testing_error(self):
#         np.testing.assert_allclose(self.F, np.array([1 / 96., 1. / 16., 1 / 8., 3. / 16.]), rtol=1E-5, atol=1E-5)
#         np.testing.assert_allclose(self.d, np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384.]), rtol=1E-5, atol=1E-5)
#         np.testing.assert_allclose(self.u, np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384., 0]), rtol=1E-5, atol=1E-5)
#         np.testing.assert_allclose(self.error, 0.003269067718342104, rtol=1E-5, atol=1E-5)


class test_setup_functions(unittest.TestCase):

    def setUp(self):

        Nell = 4
        p = 2
        Ndof = 6
        case = 0
        he = np.ones(Nell) / Nell
        self.Xe_4_2 = Xe_4_2 = node_locations_x(Nell, he)
        self.knots_4_2 = knots_4_2 = knot_vector(Nell, Xe_4_2, p)
        self.GA_4_2 = GA_4_2 = greville_abscissae(self.knots_4_2, p)
        self.ID_4_2 = ID_4_2 = get_id(case, Nell, p)
        print self.ID_4_2
        self.IEM_4_2 = IEM_4_2 = ien_array(Nell, p, Ndof)
        print self.IEM_4_2
        self.LM_4_2 = LM_4_2 = lm_array(Nell, p, ID_4_2, IEM_4_2, Ndof)
        print self.LM_4_2

        Nell = 10
        p = 3
        he = np.ones(Nell) / Nell
        self.Xe_10_3 = Xe_10_3 = node_locations_x(Nell, he)
        self.knots_10_3 = knots_10_3 = knot_vector(Nell, Xe_10_3, p)
        self.GA_10_3 = GA_10_3 = greville_abscissae(self.knots_10_3, p)
        self.ID_10_3 = ID_10_3 = get_id(case, Nell, p)
        self.IEM_10_3 = IEM_10_3 = ien_array(Nell, p)
        self.LM_10_3 = LM_10_3 = lm_array(Nell, p, ID_10_3, IEM_10_3)

        case = 1
        self.ID_10_3_R = ID_10_3_R = get_id(case, Nell, p)
        self.IEM_10_3_R = IEM_10_3_R = ien_array(Nell, p)
        self.LM_10_3_R = LM_10_3_R = lm_array(Nell, p, ID_10_3_R, IEM_10_3_R)

        # set up to test get_D
        A = 1.
        E = 1E6
        mu = 0.1
        A1s = A2s = 5./6.
        d = 0.01
        I1, I2, _, J = moment_of_inertia_rod(d)
        self.D = get_D(A, E, mu, A1s, A2s, I1, I2, J)

        # set up to test get_B
        Na = 1.
        dNadxi = 0.5
        self.Ba = get_B(Na, dNadxi)

        # print GA_10_3
        # print knots_10_3
        # print Xe_10_3
        # print ID_10_3
        # print IEM_10_3
        # print LM_10_3

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
        np.testing.assert_equal(self.ID_10_3, np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 0., 0.]))
        np.testing.assert_equal(self.ID_10_3_R, np.array([0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]))

    def testing_iem(self):
        np.testing.assert_equal(self.IEM_4_2, np.array([[1.,  2.,  3.,  4.],
                                                        [ 2.,  3.,  4.,  5.],
                                                        [ 3.,  4.,  5.,  6.]]))
        np.testing.assert_equal(self.IEM_10_3, np.array([[  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.],
                                                         [  2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.],
                                                         [  3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.,  12.],
                                                         [  4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.,  12.,  13.]]))

    def testing_lm(self):
        np.testing.assert_equal(self.LM_4_2, np.array([[1.,  2.,  3.,  4.],
                                                       [ 2.,  3.,  4.,  0.],
                                                       [ 3.,  4.,  0.,  0.]]))
        np.testing.assert_equal(self.LM_10_3, np.array([[  1.,   2.,   3.,   4.,   5.,   6.,   7.,    8.,   9.,  10.],
                                                        [  2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.],
                                                        [  3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.,  0.],
                                                        [  4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.,  0.,   0.]]))
        np.testing.assert_equal(self.LM_10_3_R, np.array([[0., 0., 1., 2., 3., 4., 5., 6., 7.,   8.],
                                                          [0., 1., 2., 3., 4., 5., 6., 7., 8.,   9.],
                                                          [1., 2., 3., 4., 5., 6., 7., 8., 9.,  10.],
                                                          [2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]]))

    def testing_getD(self):
        np.testing.assert_allclose(self.D, np.array([[1E6, 0.,  0., 0., 0.,   0.],
                                                  [0., 8.33333333E-2, 0.,   0., 0., 0.],
                                                  [0., 0.,   8.33333333E-2, 0., 0., 0.],
                                                  [0., 0., 0., 4.90873852E-4, 0., 0.],
                                                  [0., 0., 0., 0., 4.90873852E-4, 0.],
                                                  [0., 0., 0., 0., 0., 9.81747704E-11]]), rtol=1E-6, atol=1E-6)

    def testing_getB(self):
        np.testing.assert_equal(self.Ba, np.array([[0.,     0.,     0.5,    0.,     0.,     0.    ],
                                                   [0.5,    0.,     0.,     0.,    -1.,     0.    ],
                                                   [0.,     0.5,    0.,     1.,     0.,     0.    ],
                                                   [0.,     0.,     0.,     0.5,    0.,     0.    ],
                                                   [0.,     0.,     0.,     0.,     0.5,    0.    ],
                                                   [0.,     0.,     0.,     0.,     0.,     0.5   ]]))

class test_basis_functions(unittest.TestCase):

    def setUp(self):

        p = 3
        spacing = 1E-3
        xi_a = np.arange(-1,1.,spacing)
        self.B = np.zeros([p+1, len(xi_a)])
        self.Bdxi = np.zeros([p+1, len(xi_a)])
        self.Bdxidxi = np.zeros([p+1, len(xi_a)])
        for xi, i in zip(xi_a, np.arange(0, len(xi_a))):
            self.B[:, i], self.Bdxi[:, i], self.Bdxidxi[:, i] = local_bernstein(xi, p)

        self.Bdxi_fd = np.gradient(self.B, spacing, axis=1)
        self.Bdxidxi_fd = np.gradient(self.Bdxi, spacing, axis=1)

        # plot Bernstein basis
        # for c in np.arange(0, p+1):
        #     plt.plot(xi_a, self.B[c, :], '--', label='Bdxi[%i]' %c)
        # for c in np.arange(0, p + 1):
        #     plt.plot(xi_a, self.Bdxi[c, :], '--', label='Bdxidxi[%i]' %c)
        # plt.legend()
        # plt.show()

        # plt.plot(xi_a, self.Bdxi[0, :], label='Bdxi[%i]' % 0)
        # plt.show()
        # print self.Bdxi_fd.shape, self.Bdxi.shape
        # print self.Bdxi
        # print self.Bdxi_fd

        # print 'derivatives:'
        # print self.Bdxi_fd[:, 0], self.Bdxi_fd[:, -1]
        # print self.Bdxidxi_fd[:, 0], self.Bdxidxi_fd[:, -1]
        # def testing_B(self):
        #     np.testing.assert_equal(self.Bdxi, self.Bdxi_fd, rtol=1E-5)

        # set up to test Bezier extraction
        Nell = 6
        for e in np.arange(1, Nell+1):
            self.N, self.Ndxi, self.Ndxidxi = local_bezier_extraction(p, e, Nell, self.B[:, 0], self.Bdxi[:, 0], self.Bdxidxi[:, 0])

        # print self.N, self.Ndxi, self.Ndxidxi

        #TODO finish testing Bezier extraction


    def testing_Bdxi(self):
        np.testing.assert_allclose(self.Bdxi, self.Bdxi_fd, rtol=1E-3, atol=1E-3)

    def testing_Bdxidxi(self):
        np.testing.assert_allclose(self.Bdxi, self.Bdxi_fd, rtol=1E-3, atol=1E-3)

    # def testing_Ndxi(self):
    #     np.testing.assert_allclose(__, __, rtol=1E-3, atol=1E-3)
    #
    # def testing_Ndxidxi(self):
    #     np.testing.assert_allclose(__, __, rtol=1E-3, atol=1E-3)

if __name__ == "__main__":

    unittest.main()