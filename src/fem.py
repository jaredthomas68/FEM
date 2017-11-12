import math as m
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import time
import matplotlib.pylab as plt

def ffunc_constant(x, a):
    """
    Constant valued forcing function
    :param x: point at which to evaluate the forcingg function
    :param a: parameter values, in this case the value of the constant
    :return: result of function evaluation, in this case the constant 'a'
    """
    f = a
    return f

def ffunc_linear(x, a=np.array([0, 1])):
    """
    Linear forcing function
    :param x: point at which to evaluate the forcingg function
    :param a: parameter values, in this case an array with two elements
    :return: the result of the function evaluation
    """
    f = a[0] + a[1]*x
    return f

def ffunc_quadratic(x, a=np.array([0, 0, 1])):
    """
    Quadratic forcing function
    :param x: point at which to evaluate the forcingg function
    :param a: parameter values, in this case an array with three elements
    :return: the result of the function evaluation
    """
    f = a[0] + a[1]*x + a[2]*x**2
    return f

def ffunc_cubic(x, a=np.array([0., 0., 0., 10.])):

    f = a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3

    return f

def ffunc_beam(x, a=np.array([10, 0.005])):
    """
    Forcing function defined for coding 2 part 2
    :param x: location in global reference frame
    :param a: [a, h]
    :return: forcing function value
    """
    f = a[0]*a[1]**3

    return f

def moment_of_inertia_rectangle(b, h):

    Ix = (b*h**3)/12.
    Iy = (h*b**3)/12.
    Ixy = 0.

    return Ix, Iy, Ixy

def fem_solver(Nell, he, Nint, p, ID, E, I, ffunc=ffunc_quadratic, ffunc_args=np.array([0., 0., 1.]), case=2):
    print "entering solver"
    # define LM array
    IEN = ien_array(Nell, p)
    LM = lm_array(Nell, p, ID, IEN)

    # initialize global stiffness matrix
    K = np.zeros((ID[ID[:]>0].size, ID[ID[:]>0].size))

    # initialize global force vector
    F = np.zeros(ID[ID[:] > 0].size)

    # get quadrature points and weights in local coordinants
    xi, w = quadrature_rule(Nint)

    # get node locations in global coordinates
    xe = node_locations_x(Nell, he)

    # get the knot vector
    S = knot_vector(Nell, xe, p)

    # find the Greville Abscissae
    ga = greville_abscissae(S, p)

    # loop over elements
    for e in np.arange(1, Nell+1):
        # print "in element loop"
        ke = np.zeros((p + 1, p + 1))
        fe = np.zeros(p + 1)

        # solve for local stifness matrix and force vector
        for i in np.arange(0, Nint):
            # print "in integration loop"
            B, Bdxi, Bdxidxi = local_bernstein(xi[i], p)
            N, Nedxi, Nedxidxi = local_bezier_extraction(p, e, Nell, B, Bdxi, Bdxidxi)
            Ndx, Ndxdx, dxdxi, x = global_bezier_extraction(ga[e-1:e+p],N, Nedxi, Nedxidxi)
            slice = ga[e-1:e+p]
            # print x, xi[i], "fem"
            # get base k matrix for element e
            # print "in a loop"
            for a in np.arange(0, p+1):
                # print LM
                if LM[a, e - 1] == 0:
                    continue
                for b in np.arange(0, p+1):
                    # print "in b loop"
                    if LM[b, e - 1] == 0:
                        continue
                    # k_basis[a,b] = ((-1)**(a+b))/he[e]
                    # k_basis[a, b] = Ndxdx[a]*E*I*Ndxdx[b]*dxdxi*w[i]

                    ke[int(a), int(b)] += Ndx[int(a)]*E*I*Ndx[int(b)]*w[int(i)]*dxdxi

                    # K[int(LM[a, e - 1] - 1), int(LM[b, e - 1] - 1)] += Ndx[a]*E*I*Ndx[b]*w[i]*dxdxi
                fe[a] += N[a] * ffunc(x, ffunc_args) * dxdxi * w[i]

        # assemble global stifness matrix and force vector
        for a in np.arange(0, p + 1):
            if LM[a, e - 1] == 0:
                continue
            for b in np.arange(0, p + 1):
                if LM[b, e - 1] == 0:
                    continue
                # k_basis[a,b] = ((-1)**(a+b))/he[e]
                # k_basis[a, b] = Ndxdx[a]*E*I*Ndxdx[b]*dxdxi*w[i]
                # k_e[a, b] += Ndx[a] * E * I * Ndx[b] * w[i] * dxdxi

                # global stiffness matrix
                K[int(LM[a, e - 1] - 1), int(LM[b, e - 1] - 1)] += ke[a, b]

            # global force vector
            F[int(LM[a, e - 1] - 1)] += fe[a]

    # solve for d
    d = solve_for_d(K, F)
    da = np.copy(d)
    if case == 0:
        dda = np.append(d, 0.)
    elif case == 1:
        da = np.append(5., d)
    elif case == 2:
        da = np.append(d, 0.)

    # determine the number of nodes
    Nnodes = Nell + p

    # print d, Nnodes
    # quit()

    # initialize solution array
    solution = np.zeros(Nnodes)

    # populate solution array
    for a in np.arange(0, Nnodes):
        if ID[a] == 0:
            continue
        else:
            solution[a] = d[int(ID[a])-1]
    # print 'solution: ', solution
    return K, F, d, solution, da

def solve_for_d(K, F):

    sK = sparse.csr_matrix(K)
    d = spsolve(sK, F)

    return d

def solve_for_displacements(d, Nell, he, g=0):

    u = np.zeros(Nell+1)
    x1 = 0.0

    u[0] = (1.-x1)*d[0]

    for e in np.arange(1, Nell):

        x1 += he[e]

        # u[e] = u[e-1] + (1.-x1)*d[e]
        # u[e] = (1.-x1)*d[e]
        u[e] = d[e]

    # u[-1] = u[-2] + g
    u[-1] = g

    return u

def node_locations_x(Nell, he):

    x_el = np.zeros(Nell + 1)

    for e in np.arange(1, Nell):
        x_el[e] = x_el[e-1] + he[e-1]

    x_el[Nell] = x_el[Nell-1] + he[Nell-1]

    return x_el

def quadrature_rule(Nint):

    if (Nint < 1 or Nint > 3) or type(Nint) != int:
        raise ValueError('Nint must be and integer and one of 1, 2, 3')

    gp = np.zeros(Nint)
    w = np.zeros(Nint)

    if Nint == 1:
        gp[0] = 0.
        w[0] = 2.

    elif Nint == 2:
        gp[0] = -1./np.sqrt(3.)
        gp[1] = 1./np.sqrt(3.)

        w[0] = 1.
        w[1] = 1.

    elif Nint == 3:
        gp[0] = -np.sqrt(3./5.)
        gp[1] = 0.
        gp[2] = np.sqrt(3./5.)

        w[0] = 5./9.
        w[1] = 8./9.
        w[2] = 5./9.

    return gp, w

def get_u_of_x_approx(sol, he, Nell, Nint, p):

    # get IEN array
    IEN = ien_array(Nell, p)

    # get quadrature points and weights in local coordinants
    xi_sample, w = quadrature_rule(Nint)

    # find number of samples
    Nsamples = xi_sample.size

    # initialize displacement vector
    u = np.zeros(Nell * Nint+1)

    # initialize error vector
    error = np.zeros(Nell * Nint)

    # initialize x vector
    X = np.zeros(Nell * Nint)

    # get node locations in global coordinates
    xe = node_locations_x(Nell, he)

    # get the knot vector
    S = knot_vector(Nell, xe, p)

    # find the Greville Abscissae
    ga = greville_abscissae(S, p)

    # set up resulting x location vector
    x_sample = np.zeros(Nell*Nsamples+1)

    # loop over elements
    print "start loop"
    for e in np.arange(0, Nell):
        # loop over samples
        # for i in np.arange(0, Nsamples):
        for i in np.arange(0, xi_sample.size):
            B, Bdxi, Bdxidxi = local_bernstein(xi_sample[i], p)
            N, Nedxi, Nedxidxi = local_bezier_extraction(p, e + 1, Nell, B, Bdxi, Bdxidxi)
            Ndx, Ndxdx, dxdxi, x = global_bezier_extraction(ga[e:e + p + 1], N, Nedxi, Nedxidxi)

            x_sample[e*Nsamples+i] = x

            # print x, xi_sample[i]
            for a in np.arange(0, p + 1):
                u[int(e * Nint + i)] += N[a] * sol[int(IEN[a, e]) - 1]
    return u, x_sample

def get_u_of_x_exact(x, q, ffunc_num):

    u_ex = 0.

    if ffunc_num == 0:
        u_ex = q*(1.-x**2)/2.
    elif ffunc_num == 1:
        u_ex = q*(1.-x**3)/6.
    elif ffunc_num == 2:
        u_ex = q * (1. - x ** 4) / 12.

    return u_ex

def knot_vector(Nell, Xe, p, open=True):
    """
    Construct knot vector
    :param Nell: number of elements
    :param he: array containing the length of each element
    :param p: order of basis functions
    :return knots: knot vector
    """

    # initialize knot vector
    knots = np.zeros([Nell+2*p+1])

    # populate knot vector
    if open:
        knots[0:p+1] = Xe[0]
        knots[-p-1:] = Xe[-1]

        for i in np.arange(1, Nell):
            knots[i+p] = Xe[i]

    return knots

def greville_abscissae(S, p):

    Nell = len(S) - 2*p - 1
    GA = np.zeros(Nell+p)

    for i in np.arange(0, Nell+p):
        GA[i] = (1./p)*(np.sum(S[i+1:i+p+1]))
        # print i, GA[i], S[i+1:i+p+1], np.sum(S[i+1:i+p+1]), p

    return GA

def get_id(case, Nell, p):

    ID = np.zeros(Nell+p)

    # cantilever L
    if case == 0:
        # print 'here in ', case
        ID[0:Nell] = np.arange(1,Nell+1)


    # cantilever R
    elif case == 1:
        # print 'here in ', case
        ID[p:] = np.arange(1,Nell+1)

    # coding two part one
    elif case == 2:
        ID[0:Nell+p-1] = np.arange(1, Nell+p)

    else:
        raise ValueError('invalid ID case')

    return ID

def ien_array(Nell, p):

    IEN = np.zeros([p+1, Nell])
    for i in np.arange(0, p+1):
        IEN[i,:] = np.arange(i+1, i+1+Nell)

    return IEN

def local_bernstein(xi, p):

    # check if xi is in the acceptable range
    if np.any(xi < -1) or np.any(xi >1):
        raise ValueError("the value of xi is $f, but must be in the range [-1, 1]" %xi)

    # check if p is in the acceptable range for this code
    if p > 3 or p < 2:
        raise ValueError("the value of p must be 2 or 3, but %i was given" % p)

    # initialize Bernstein polynomial vectors
    B = np.zeros(p+1)
    Bdxi = np.zeros(p+1)
    Bdxidxi = np.zeros(p+1)

    for a in np.arange(1., p + 2.):
        # compute common factor of B and it's derivatives
        eta = (1. / (2. ** p)) * (m.factorial(p) / (m.factorial(a - 1.) * m.factorial(p + 1. - a)))

        # calculate the value and derivatives of each element of the Bernstein polynomial vector
        # print eta*((1.-xi)**(p-(a-1.)))*((1+xi)**(a-1.))
        B[int(a - 1)] = eta * ((1. - xi) ** (p - (a - 1.))) * ((1. + xi) ** (a - 1.))


    if xi == -1.:
        if p == 2:
            Bdxi[0] = -1.
            Bdxi[1] = 1.
            Bdxi[2] = 0.

            Bdxidxi[0] = 0.5
            Bdxidxi[1] = -1.0
            Bdxidxi[2] = 0.5
        elif p == 3:
            Bdxi[0] = -1.5
            Bdxi[1] = 1.5
            Bdxi[2] = 0.
            Bdxi[3] = 0.

            Bdxidxi[0] = 1.5
            Bdxidxi[1] = -3.
            Bdxidxi[2] = 1.5
            Bdxidxi[3] = 0.

    elif xi == 1.:
        if p == 2:
            Bdxi[0] = 0.
            Bdxi[1] = -1.
            Bdxi[2] = 1.

            Bdxidxi[0] = 0.5
            Bdxidxi[1] = -1.0
            Bdxidxi[2] = 0.5
        if p == 3:
            Bdxi[0] = 0.
            Bdxi[1] = 0.
            Bdxi[2] = -1.5
            Bdxi[3] = 1.5

            Bdxidxi[0] = 0.
            Bdxidxi[1] = 1.5
            Bdxidxi[2] = -3.
            Bdxidxi[3] = 1.5

    else:
        # solve for the Bernstein polynomial vectors
        for a in np.arange(1, p+2):

            # compute common factor of B and it's derivatives
            eta = (1./(2.**p))*(m.factorial(p)/(m.factorial(a-1.)*m.factorial(p+1.-a)))

            # calculate the value and derivatives of each element of the Bernstein polynomial vector
            # print eta*((1.-xi)**(p-(a-1.)))*((1+xi)**(a-1.))
            # B[a-1] = eta*((1.-xi)**(p-(a-1.)))*((1+xi)**(a-1.))

            Bdxi[a-1] = eta*(((1.-xi)**(p-a+1.))*(a-1.)*((1.+xi)**(a-2.))-
                             ((1.+xi)**(a-1.))*(p-a+1.)*((1.-xi)**(p-a)))

            # set up terms for second derivative
            t1 = ((1.-xi)**(p-a+1))*(a-2.)*((1+xi)**(a-3.))
            t2 = -((1.+xi)**(a-2.))*(p-a+1.)*((1.-xi)**(p-a))
            t3 = -((1.+xi)**(a-1.))*(p-a)*((1.-xi)**(p-a-1.))
            t4 = ((1.-xi)**(p-a))*(a-1.)*((1.+xi)**(a-2.))

            Bdxidxi[a-1] = eta*((a-1.)*(t1+t2)-(p-a+1.)*(t3+t4))

    return B, Bdxi, Bdxidxi

def local_bezier_extraction(p, e, Nell, B, Bdxi, Bdxidxi):
    # if Nell = 1 C = Identity
    # determine the appropriate Bezier extraction matrix
    if Nell == 1:

        C = np.identity(p+1)

    elif p == 2:
        if e == 1:
            C = np.array([[1., 0., 0. ],
                          [0., 1., 0.5],
                          [0., 0., 0.5]])

        elif e >=2 and e <= Nell-1.:
            C = np.array([[0.5, 0., 0. ],
                          [0.5, 1., 0.5],
                          [0., 0.,  0.5]])

        elif e == Nell:
            C = np.array([[0.5, 0., 0.],
                          [0.5, 1., 0.],
                          [0.,  0., 1.]])

        else:
            raise ValueError('Invalid value of e. Must be in [1, %i], but %i was given' % (Nell,e))

    elif p == 3:
        if e == 1:
            C = np.array([[1., 0., 0.,  0.    ],
                          [0., 1., 0.5, 0.25  ],
                          [0., 0., 0.5, 7./12.],
                          [0., 0., 0.,  1./6.  ]])

        elif e == 2:
            C = np.array([[0.25,   0.,    0.,    0.   ],
                          [7./12., 2./3., 1./3., 1./6.],
                          [1./6.,  1./3., 2./3., 2./3.],
                          [0.,     0.,    0.,    1./6.]])

        elif e >= 3 and e <= Nell-2:
            C = np.array([[1./6.,  0.,    0.,    0.   ],
                          [2./3.,  2./3., 1./3., 1./6.],
                          [1./6.,  1./3., 2./3., 2./3.],
                          [0.,     0.,    0.,    1./6.]])

        elif e == Nell-1.:
            C = np.array([[1./6., 0.,    0.,    0.    ],
                          [2./3., 2./3., 1./3., 1./6. ],
                          [1./6., 1./3., 2./3., 7./12.],
                          [0.,    0.,    0.,    0.25  ]])

        elif e == Nell:
            C = np.array([[1./6.,  0.,  0., 0.],
                          [7./12., 0.5, 0., 0.],
                          [0.25,   0.5, 1., 0.],
                          [0.,     0.,  0., 1.]])
        else:
            raise ValueError('Invalid value of e. Must be in [1, %i], but %i was given' % (Nell, e))
    else:
        raise ValueError('p must be 2 or 3, but p=%f was given' % p)

    # solve for the value of the Bezier basis function and derivatives on the element (Ne)
    Ne = np.matmul(C, B)

    Nedxi = np.matmul(C, Bdxi)

    Nedxidxi = np.matmul(C, Bdxidxi)

    return Ne, Nedxi, Nedxidxi

def global_bezier_extraction(GA, Ne, Nedxi, Nedxidxi):

    # solve for xe and derivatives
    xe = np.sum(GA*Ne)
    # print GA, Nedxi
    dxedxi = np.sum(GA*Nedxi)

    dxedxedxidxi = np.sum(GA*Nedxidxi)

    # derivatives of the basis function in global coordinates
    Ndx = Nedxi/dxedxi
    Ndxdx = (Nedxidxi - Ndx*dxedxedxidxi)/(dxedxi**2)
    # print 'dxidxi', dxedxi
    return Ndx, Ndxdx, dxedxi, xe

def error_quadrature(solution, p, Nell, Nint, he):

    # get IEN array
    IEN = ien_array(Nell, p)

    # get quadrature points and weights in local coordinants
    xi_sample, w = quadrature_rule(Nint)

    # initialize displacement vector
    u = np.zeros(Nell * Nint)

    # initialize error vector
    error = np.zeros(Nell * Nint)

    # initialize x vector
    X = np.zeros(Nell * Nint)

    # get node locations in global coordinates
    xe = node_locations_x(Nell, he)

    # get the knot vector
    S = knot_vector(Nell, xe, p)

    # find the Greville Abscissae
    ga = greville_abscissae(S, p)

    # loop over elements
    print "start loop"
    for e in np.arange(0, Nell):
        # loop over samples
        # for i in np.arange(0, Nsamples):
        for i in np.arange(0, xi_sample.size):
            B, Bdxi, Bdxidxi = local_bernstein(xi_sample[i], p)
            N, Nedxi, Nedxidxi = local_bezier_extraction(p, e + 1, Nell, B, Bdxi, Bdxidxi)
            Ndx, Ndxdx, dxdxi, x = global_bezier_extraction(ga[e:e+p+1],N, Nedxi, Nedxidxi)

            # print x, xi_sample[i]
            for a in np.arange(0, p + 1):
                u[int(e * Nint + i)] += N[a] * solution[int(IEN[a, e]) - 1]

            u_ex = get_u_of_x_exact(x, 1, 2)

            error[e * Nint + i] += ((u_ex - u[int(e * Nint + i)])**2)*dxdxi*w[i]
            # print error, e, i, e * Nint + i, u, X
    # print "end loop", error
    error = np.sqrt(abs(np.sum(error)))

    # initialize location array
    # x = np.linspace(0., 1., Nell * Nsamples)
    # x_ex = np.linspace(0., 1., 500)
    # print x
    # print u
    # q = 1
    # u_ex = get_u_of_x_exact(x_ex, q, 2)
    # print error
    # quit()
    return error, X

def lm_array(Nell, p, ID, IEM):

    LM = np.zeros([p+1, Nell])

    for e in np.arange(0, Nell):
        for a in range(0, p+1):
            LM[a, e] = ID[int(IEM[a, e]-1)]

    return LM

def plot_error():

    E = I = 1.

    Nint = 3

    n = np.array([1, 10, 100, 1000])

    theoretical_error = np.zeros([2, n.size])
    real_error = np.zeros([2, n.size])
    # slope = np.zeros([2, n.size-1])

    q = 1

    h = np.zeros([2, n.size])

    nodes = np.zeros([2, n.size])

    r_slope = np.zeros(2)
    t_slope = np.zeros(2)

    # print h, n
    for p, i in zip(np.array([2, 3]), np.arange(0, 2)):
        for Nell, j in zip(n, np.arange(n.size)):
            print 'p = %i, N = %i' % (p, Nell)
            # run_quadratic(Nell, Nint, p)

            nodes[i,j] = Nell + p

            he = np.ones(Nell) / Nell

            h[i, j] = he[0]

            ID = get_id(2, Nell, p)

            K, F, d, sol, da = fem_solver(Nell, he, Nint, p, ID, E, I)

            # u = solve_for_displacements(d, Nell, he, g=0)

            real_error[i, j], x = error_quadrature(sol, p, Nell, Nint, he)

            # u_ap = get_u_of_x_approx(x, u, he)
            u_ex = get_u_of_x_exact(x, q, 2)
            # print u_ap, u_ex
            # error[i, j] = np.sum(n(u_ap - u_ex)**2)

            theoretical_error[i, j] = (abs(u_ex[0])*he[0]**(p+1))
            # print theoretical_error


            # print "ffunc: %i, Nell: %i, Error: %f" % (ffunc_num, Nell, error[i, j])

        r_slope[i] = np.log(real_error[i, 2]/real_error[i, 0])/np.log(n[2]/n[0])
        t_slope[i] = np.log(theoretical_error[i, -1]/theoretical_error[i, 0])/np.log(n[-1]/n[0])

    # print (np.log(error[1])-np.log(error[0]))/(x[1]-x[0])
    # print real_error.shape
    # quit()

    # np.savetxt('error.txt', np.c_[n, he, np.transpose(error)], header="Nell, h, E(f(x)=c), E(f(x)=x), E(f(x)=x^2)")
    # print h.shape, real_error.shape
    plt.loglog(h[0, :], theoretical_error[0,:], '--or', label='A priori, $p=2, slope=%.3f$' % t_slope[0])
    plt.loglog(h[0, :], real_error[0,:], '-or', label='Real, $p=2$, $slope=%.3f$' % r_slope[0])
    plt.loglog(h[1, :], theoretical_error[1,:], '--ob', label='A priori, $p=3, slope=%.3f$' % t_slope[1])
    plt.loglog(h[1, :], real_error[1,:], '-ob', label='Real, $p=3$, $slope=%.3f$' % r_slope[1])
    # plt.loglog(he, error[2,:], '-o', label='$f(x)=x^2$')
    leg = plt.legend(loc=4, frameon=False)
    leg.get_frame().set_alpha(0.0)
    plt.xlabel('$h$')
    plt.ylabel('$Error$')

    plt.savefig('error_he.pdf', tranparent=True)
    plt.show()

    plt.loglog(nodes[0, :], theoretical_error[0, :], '--or', label='A priori, $p=2$' % t_slope[0])
    plt.loglog(nodes[0, :], real_error[0, :], '-or', label='Real, $p=2$' % r_slope[0])
    plt.loglog(nodes[1, :], theoretical_error[1, :], '--ob', label='A priori, $p=3$' % t_slope[1])
    plt.loglog(nodes[1, :], real_error[1, :], '-ob', label='Real, $p=3$' % r_slope[1])
    # plt.loglog(he, error[2,:], '-o', label='$f(x)=x^2$')
    leg=plt.legend(loc=1, frameon=False)
    leg.get_frame().set_alpha(0.0)
    plt.xlabel('$Nodes$')
    plt.ylabel('$Error$')

    plt.savefig('error_nodes.pdf', transparent=True)
    plt.show()

    return

def plot_displacements(u, x, he, Nell, q=1, ffunc=ffunc_constant, ffunc_args=np.array([1])):

    plt.rcParams.update({'font.size': 22})

    x_ex = np.linspace(0, 1., 100)

    x_el = node_locations_x(Nell, he)

    u_ex = get_u_of_x_exact(x_ex, q, ffunc_num=len(ffunc_args)-1)

    u_a = get_u_of_x_approx(x, u, he)

    plt.figure()
    plt.plot(x_ex, u_ex, label="Exact sol.", linewidth=3)
    # plt.plot(x_el, u, '-s', label="Approx. sol. (nodes)")
    plt.plot(x, u_a, '--r', markerfacecolor='none', label="Approx. sol.", linewidth=3)

    plt.xlabel('X Position')
    plt.ylabel("Displacement")
    functions = ["$f(x)=c$", "$f(x)=x$", "$f(x)=x^2$"]
    # plt.title(functions[ffunc]+", $n=%i$" %Nell, y=1.02)
    plt.legend(loc=3, frameon=False)
    plt.tight_layout()
    # plt.savefig("displacement_func%i_Nell%i.pdf" %(ffunc, Nell))
    plt.show()
    plt.close()
    return

def run_constant(Nell, Nint, p, E=1, I=1):

    ffunc = ffunc_constant
    ffunc_args = np.array([1.])
    he = np.ones(Nell) / Nell
    x = np.linspace(0, 1, 4 * Nell + 1)

    ID = get_id(2, Nell, p)
    Xe = node_locations_x(Nell, he)

    tic = time.time()
    K, F, d = fem_solver(Nell, he, Nint, p, ID, E, I, ffunc, ffunc_args)
    toc = time.time()
    print he, Nell, K
    print "Time to run fem solver: %.3f (s)" % (toc - tic)

    tic = time.time()
    u = solve_for_displacements(d, Nell, he, g=0)
    toc = time.time()
    print u
    print np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384., 0])
    print "Time to solve for u(x): %.3f (s)" % (toc - tic)

    print "Finished"

    # error = quadrature(Xe, he, u, ffunc_args=ffunc_args)
    # print error
    plot_displacements(u, x, he, Nell, ffunc=ffunc, ffunc_args=ffunc_args)

def run_linear(Nell, Nint, p, E=1, I=1):

    ffunc = ffunc_linear
    ffunc_args = np.array([0, 1])
    he = np.ones(Nell) / Nell
    x = np.linspace(0, 1, 4 * Nell + 1)

    ID = get_id(2, Nell, p)
    Xe = node_locations_x(Nell, he)

    tic = time.time()
    K, F, d = fem_solver(Nell, he, Nint, p, ID, E, I, ffunc, ffunc_args)
    toc = time.time()
    print he, Nell, K
    print "Time to run fem solver: %.3f (s)" % (toc - tic)

    tic = time.time()
    u = solve_for_displacements(d, Nell, he, g=0)
    toc = time.time()
    print u
    print np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384., 0])
    print "Time to solve for u(x): %.3f (s)" % (toc - tic)

    print "Finished"

    # error = quadrature(Xe, he, u, ffunc_args=ffunc_args)
    # print error
    plot_displacements(u, x, he, Nell, ffunc=ffunc, ffunc_args=ffunc_args)

def run_quadratic(Nell, Nint, p, E=1, I=1, Nsamples=3):

    ffunc = ffunc_quadratic
    ffunc_args = np.array([0, 0, 1])
    he = np.ones(Nell) / Nell
    x = np.linspace(0, 1, 4 * Nell + 1)

    ID = get_id(2, Nell, p)
    Xe = node_locations_x(Nell, he)

    tic = time.time()
    K, F, d, sol = fem_solver(Nell, he, Nint, p, ID, E, I, ffunc)
    toc = time.time()
    print he, Nell, K
    print "Time to run fem solver: %.3f (s)" % (toc - tic)

    tic = time.time()
    u = solve_for_displacements(d, Nell, he, g=0)
    toc = time.time()
    print u
    print np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384., 0])
    print "Time to solve for u(x): %.3f (s)" % (toc - tic)

    print "Finished"

    # error = quadrature(Xe, he, u, ffunc_args=ffunc_args)
    # print error
    plot_displacements(u, x, he, Nell, ffunc=ffunc, ffunc_args=ffunc_args)
    plot_results(sol, p, Nell, Nsamples, ID)

    return sol

def plot_results(solution, p, Nell, Nsamples, ID):
    print "here"
    # create sample vector
    xi_sample = np.linspace(-1., 1., Nsamples)

    # get IEN array
    IEN = ien_array(Nell, p)

    # get quadrature points and weights in local coordinants
    xi_sample, w = quadrature_rule(Nint)

    Nsamples = Nint
    # initialize displacement array
    u = np.zeros(Nell*Nsamples)

    # loop over elements
    for e in np.arange(0, Nell):
        # loop over samples
        # for i in np.arange(0, Nsamples):
        for i in np.arange(0, Nsamples):
            B, Bdxi, Bdxidxi = local_bernstein(xi_sample[i], p)
            N, Nedxi, Nedxidxi = local_bezier_extraction(p, e+1, Nell, B, Bdxi, Bdxidxi)
            # print N, p, IEN, solution
            # print e, i
            for a in np.arange(0, p+1):

                u[int(e*Nsamples + i)] += N[a]*solution[int(IEN[a, e])-1]

    # initialize location array
    x = np.linspace(0., 1., Nell*Nsamples)
    x_ex = np.linspace(0., 1., 500)
    # print x
    # print u
    q = 1
    u_ex = get_u_of_x_exact(x_ex, q, 2)

    plt.plot(x_ex, u_ex)
    plt.plot(x, u, '--')
    plt.show()

def beam_solution_1():

    # problem parameters
    E = 1000000
    b = 0.005
    h = 0.005
    l = 1.
    Nint = 3

    # number of elements
    n = np.array([1, 10, 100])

    # forcing function
    ffunc = ffunc_beam

    # forcing function arguments
    ffunc_args = np.array([10., h])

    # get the moment of inertia of the cross section
    Ix, _, _ = moment_of_inertia_rectangle(b, h)
    # print Ix*E
    # quit()

    # set case to use
    case = 1

    figure, axes = plt.subplots(2, 3, sharex=True, sharey=True)
    max_deflection_fem = np.zeros([2, n.size])
    max_deflection_thoeretical = np.zeros([2, n.size])
    nodes = np.zeros([2, int(n.size)])

    for p, i in zip(np.array([2, 3]), np.arange(0, 2)):

        for Nell, j in zip(n, np.arange(0, n.size)):

            # vector of element lengths
            he = np.ones(Nell) / Nell

            # vector of element locations
            x = np.linspace(0, 1, 4 * Nell + 1)

            # ID array
            ID = get_id(case, Nell, p)

            nodes[i, j] = Nell+p

            tic = time.time()
            K, F, d, sol, da = fem_solver(Nell, he, Nint, p, ID, E, Ix, ffunc=ffunc, ffunc_args=ffunc_args, case=case)
            toc = time.time()
            print he, Nell, K
            print "Time to run fem solver: %.3f (s)" % (toc - tic)

            u, x = get_u_of_x_approx(sol, he, Nell, Nint, p)
            print np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384., 0])
            print "Time to solve for u(x): %.3f (s)" % (toc - tic)

            u_exact = (ffunc(x, ffunc_args)*x**2)*(x**2+6.*l-4.*l*x)/(24.*E*Ix)
            print "Finished"

            print d, u, x[::-1]
            # if Nell > 1:
            axes[i,j].plot(x[:-1], u[:-1])
            axes[i,j].plot(x[:-1], u_exact[:-1])

            max_deflection_fem[i,j] = max(u)
            max_deflection_thoeretical[i,j] = max(u_exact)

    axes[0,0].set_title('$N_{ell}=1$')
    axes[0,1].set_title('$N_{ell}=10$')
    axes[0,2].set_title('$N_{ell}=100$')
    axes[0,0].set_ylabel('Deflection, $p=2$')
    axes[1,0].set_ylabel('Deflection, $p=3$')
    axes[1,0].set_xlabel('X Position')
    axes[1,1].set_xlabel('X Position')
    axes[1,2].set_xlabel('X Position')

    plt.show()


    fig = plt.figure()

    plt.plot(nodes[0,:], max_deflection_thoeretical[0,:], label='theoretical, p=2')
    plt.plot(nodes[0,:], max_deflection_fem[0,:], label='fem, p=2')
    plt.plot(nodes[1,:], max_deflection_thoeretical[1,:], label='theoretical, p=3')
    plt.plot(nodes[1,:], max_deflection_fem[1,:], label='fem, p=3')
    plt.xlabel('Nodes')
    plt.ylabel('Max Deflection')
    plt.legend(loc = 0)

    plt.savefig('max_deflection_vs_n.pdf', transparent=True)
    plt.show()



    return

def beam_solution_2():

    # problem parameters
    E = 1000000
    b = 0.005
    h = 0.005
    l = 1.
    Nint = 3
    Nell = 10

    # number of elements
    H = np.array([0.1, 0.01, 0.005, 0.002, 0.001])

    # forcing function
    ffunc = ffunc_beam




    # print Ix*E
    # quit()

    # set case to use
    case = 1

    figure, axes = plt.subplots(2, H.size, sharex=True, sharey=True)
    max_deflection_fem = np.zeros([2, H.size])
    max_deflection_thoeretical = np.zeros([2, H.size])
    nodes = np.zeros([2, int(H.size)])

    for p, i in zip(np.array([2, 3]), np.arange(0, 2)):

        for h, j in zip(H, np.arange(0, H.size)):

            # forcing function arguments
            ffunc_args = np.array([10., h])

            # get the moment of inertia of the cross section
            Ix, _, _ = moment_of_inertia_rectangle(b, h)



            # vector of element lengths
            he = np.ones(Nell) / Nell

            # vector of element locations
            x = np.linspace(0, 1, 4 * Nell + 1)

            # ID array
            ID = get_id(case, Nell, p)

            nodes[i, j] = Nell+p

            tic = time.time()
            K, F, d, sol, da = fem_solver(Nell, he, Nint, p, ID, E, Ix, ffunc=ffunc, ffunc_args=ffunc_args, case=case)
            toc = time.time()
            # print he, Nell, K
            # print "Time to run fem solver: %.3f (s)" % (toc - tic)

            u, x = get_u_of_x_approx(sol, he, Nell, Nint, p)
            # print np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384., 0])
            # print "Time to solve for u(x): %.3f (s)" % (toc - tic)

            u_exact = (ffunc(x, ffunc_args)*x**2)*(x**2+6.*l-4.*l*x)/(24.*E*Ix)

            print "Finished"
            print "Ix: ", Ix, "h: ", h
            print "max def: ", np.max(u_exact)
            # print d, u, x[::-1]
            # if Nell > 1:
            axes[i,j].plot(x[:-1], u[:-1])
            axes[i,j].plot(x[:-1], u_exact[:-1])

            max_deflection_fem[i,j] = max(u)
            max_deflection_thoeretical[i,j] = max(u_exact)

    axes[0,0].set_title('$N_{ell}=1$')
    axes[0,1].set_title('$N_{ell}=10$')
    axes[0,2].set_title('$N_{ell}=100$')
    axes[0,0].set_ylabel('Deflection, $p=2$')
    axes[1,0].set_ylabel('Deflection, $p=3$')
    axes[1,0].set_xlabel('X Position')
    axes[1,1].set_xlabel('X Position')
    axes[1,2].set_xlabel('X Position')

    plt.show()


    fig = plt.figure()

    plt.plot(l/H, max_deflection_thoeretical[0,:], label='theory p=2')
    plt.plot(l/H, max_deflection_fem[0,:], label='fem p=2')
    plt.plot(l/H, max_deflection_thoeretical[1,:], '--', label='theory p=3')
    plt.plot(l/H, max_deflection_fem[1,:], label='fem p=3')
    plt.legend()

    plt.savefig('max_deflection_vs_h.pdf', transparent=True)
    plt.show()

    return

if __name__ == "__main__":

    # get_lm(10)
    # get_slope()
    # exit()
    # plot_error()
    # exit()
    #
    # input variables

    # p = 3                       # basis function order
    # Nell = 10                   # number of elements
    # Nint = 3                    # number of quadrature intervals
    # Nsamples = 3
    # run_constant(Nell, Nint, p)
    # run_linear(Nell, Nint, p)
    # sol = run_quadratic(Nell, Nint, p, Nsamples=Nsamples)
    # plot_error()
    # plot_results()

    beam_solution_1()
    beam_solution_2()