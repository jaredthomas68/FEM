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

def define_stiffness_matrix(Nell, he, Nint, p, ID, E, I):

    # initialize local and global stiffness matrices
    k_basis = np.zeros((p+1,p+1))
    K = np.zeros((Nell, Nell))

    # define LM matrix
    IEM = get_iem(Nell, p)
    LM = get_lm(Nell, p, ID, IEM)

    # get quadrature points and weights in local coordinants
    xi, w = get_quadrature_rule(Nint)

    # get node locations in global coordinates
    xe = get_node_locations_x(Nell, he)

    S = get_knot_vector(Nell, xe, p)

    ga = get_greville_abscissae(S, p)

    # loop over elements
    for e in np.arange(1, Nell+1):
        print ga, ga[e-1:e+p]
        # quit()
        # loop over each interval in element e
        for i in np.arange(0, Nint):
            B, Bdxi, Bdxidxi = local_bernstein(xi[i], p)
            N, Nedxi, Nedxidxi = local_bezier_extraction(p, e, Nell, B, Bdxi, Bdxidxi)
            Ndx, Ndxdx, dxdxi, x = global_bezier_extraction(ga[e-1:e+p],N[i], Nedxi, Nedxidxi)

            # get base k matrix for element e
            for a in np.arange(0, p+1):
                for b in np.arange(0, p+1):
                    # k_basis[a,b] = ((-1)**(a+b))/he[e]
                    k_basis[a, b] = Ndxdx[a]*E*I*Ndxdx[b]*dxdxi*w[i]

        # populate the stifness matrix for entries corresponding to element e
        for a in np.arange(0, p+1):
            if LM[a, e-1] == 0:
                continue
            for b in np.arange(0, p+1):
                if LM[b, e-1] == 0:
                    continue
                K[int(LM[a,e-1]-1), int(LM[b,e-1]-1)] += k_basis[a,b]

    return K

def define_forcing_vector(Nell, he, ffunc=ffunc_constant, ffunc_args=np.array([1])):

    f_e = np.zeros(2)
    F = np.zeros(Nell)
    x1 = 0.
    for e in np.arange(0, Nell):
        x2 = x1 + he[e]
        f_e = (he[e]/6.)*np.array([2.*ffunc(x1, ffunc_args) + ffunc(x2, ffunc_args),
                                  ffunc(x1, ffunc_args) + 2.*ffunc(x2, ffunc_args)])
        if e == 0:
            F[e] = f_e[0]
            F[e+1] = f_e[1]
        elif e < Nell - 1:
            F[e] += f_e[0]
            F[e+1] = f_e[1]
        else:
            F[e] += f_e[0]
        x1 += he[e]
    return F

def solve_for_d(K, F):

    # Kd = F
    # sK = sparse.csr_matrix(K)
    # sK = sparse.csr_matrix(F)

    # d = np.matmul(np.linalg.inv(K),F)
    # print d

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

def get_node_locations_x(Nell, he):

    x_el = np.zeros(Nell + 1)

    for e in np.arange(1, Nell):
        x_el[e] = x_el[e-1] + he[e-1]

    x_el[Nell] = x_el[Nell-1] + he[Nell-1]

    return x_el

def get_quadrature_rule(Nint):

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

def plot_displacements(u, x, he, Nell, q=1, ffunc=ffunc_constant, ffunc_args=np.array([1])):

    plt.rcParams.update({'font.size': 22})

    x_ex = np.linspace(0, 1., 100)

    x_el = get_node_locations_x(Nell, he)

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

def get_u_of_x_approx(Xp, u, he):
    Nell = he.size
    Xn = np.zeros(Nell+1)
    for e in np.arange(1, Nell+1):
        Xn[e] = Xn[e-1] + he[e-1]

    u_x = np.zeros_like(Xp)
    for x, i in zip(Xp, np.arange(0,Xp.size)):
        for e in np.arange(1, Nell+1):
            if x < Xn[e]:
                u_x[i] = ((u[e] - u[e-1])/(Xn[e] - Xn[e-1]))*(x-Xn[e-1]) + u[e-1]
                break

    return u_x

def get_u_of_x_exact(x, q, ffunc_num):

    u_ex = 0.

    if ffunc_num == 0:
        u_ex = q*(1.-x**2)/2.
    elif ffunc_num == 1:
        u_ex = q*(1.-x**3)/6.
    elif ffunc_num == 2:
        u_ex = q * (1. - x ** 4) / 12.

    return u_ex

def get_knot_vector(Nell, Xe, p, open=True):
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

def get_greville_abscissae(S, p):

    Nell = len(S) - 2*p - 1
    GA = np.zeros(Nell+p)

    for i in np.arange(0, Nell+p):
        GA[i] = (1./p)*(np.sum(S[i+1:i+p+1]))
        # print i, GA[i], S[i+1:i+p+1], np.sum(S[i+1:i+p+1]), p

    return GA

def get_id(case, Nell, p):

    ID = np.zeros(Nell+p)

    if case == 'cantilever L':
        # print 'here in ', case
        ID[0:Nell] = np.arange(1,Nell+1)

    if case == 'cantilever R':
        # print 'here in ', case
        ID[p:] = np.arange(1,Nell+1)

    # print ID
    return ID

def get_iem(Nell, p):

    IEM = np.zeros([p+1, Nell])
    for i in np.arange(0, p+1):
        IEM[i,:] = np.arange(i+1, i+1+Nell)

    return IEM

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

    for a in np.arange(1, p + 2):
        # compute common factor of B and it's derivatives
        eta = (1. / (2. ** p)) * (m.factorial(p) / (m.factorial(a - 1.) * m.factorial(p + 1. - a)))

        # calculate the value and derivatives of each element of the Bernstein polynomial vector
        # print eta*((1.-xi)**(p-(a-1.)))*((1+xi)**(a-1.))
        B[a - 1] = eta * ((1. - xi) ** (p - (a - 1.))) * ((1 + xi) ** (a - 1.))


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

    # determine the appropriate Bezier extraction matrix
    if p == 2:
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
                          [0., 0., 0.,  0.25  ]])

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
    print GA, Nedxi
    dxedxi = np.sum(GA*Nedxi)

    dxedxedxidxi = np.sum(GA*Nedxidxi)

    # derivatives of the basis function in global coordinates
    Ndx = Nedxi/dxedxi
    Ndxdx = (Nedxidxi - Ndx*dxedxedxidxi)/(dxedxi**2)

    return Ndx, Ndxdx, dxedxi, xe

def quadrature(Xe, he, ue, ffunc_args=1):

    # print Xe, he, ue, ffunc
    # quit()
    Nell = he.size
    xi = np.array([-np.sqrt(3./5.), 0., np.sqrt(3./5.)])
    w = np.array([5./9., 8./9., 5./9.])

    ffunc_num = len(ffunc_args)-1

    error_squared = 0.0
    for el in np.arange(0, Nell):
        x = x_of_xi(xi, Xe, he, el)

        dxdxi = he[el]/2

        ux = get_u_of_x_exact(x, q=1, ffunc_num=ffunc_num)
        ua = get_u_of_x_approx(x, ue, he)

        error_squared += np.sum(((ux-ua)**2)*dxdxi*w)
    error = np.sqrt(error_squared)

    return error

def x_of_xi(xi, Xe, he, el):
    # print xi, Xe.size, he, el
    # quit()
    x = (he[el]*xi+Xe[el]+Xe[el+1])/2.

    return x

def get_lm(Nell, p, ID, IEM):

    LM = np.zeros([p+1, Nell])

    for e in np.arange(0, Nell):
        for a in range(0, p+1):
            LM[a, e] = ID[int(IEM[a, e]-1)]

    return LM

def plot_error():

    Nint = 1

    n = np.array([10, 100, 1000, 10000])
    error = np.zeros([3, n.size])
    h = np.ones(n.size)/n

    ffunc_array = [ffunc_constant, ffunc_linear, ffunc_quadratic]
    ffunc_args_array = [1, np.array([0, 1]), np.array([0,0,1])]
    print h, n
    for ffunc_num, i in zip(np.array([0, 1, 2]), np.arange(0, 3)):
        for Nell, j in zip(n, np.arange(n.size)):

            # print Nell
            he = np.ones(Nell) / Nell
            Xe = get_node_locations_x(Nell, he)
            K = define_stiffness_matrix(Nell, he, Nint)
            F = define_forcing_vector(Nell, he, ffunc=ffunc_array[ffunc_num], ffunc_args=ffunc_args_array[ffunc_num])
            d = solve_for_d(K, F)
            u = solve_for_displacements(d, Nell, he, g=0)
            error[i,j] = quadrature(Xe, he, u, ffunc_args_array)
            print "ffunc: %i, Nell: %i, Error: %f" % (ffunc_num, Nell, error[i, j])

    np.savetxt('error.txt', np.c_[n, h, np.transpose(error)], header="Nell, h, E(f(x)=c), E(f(x)=x), E(f(x)=x^2)")
    plt.loglog(h, error[0,:], '-o', label='$f(x)=c$')
    plt.loglog(h, error[1,:], '-o', label='$f(x)=x$')
    plt.loglog(h, error[2,:], '-o', label='$f(x)=x^2$')
    plt.legend(loc=2)
    plt.xlabel('$h$')
    plt.ylabel('$Error$')
    plt.show()
    return

def get_slope():

    data = np.loadtxt('error.txt')
    fx0 = data[:, 2]
    fx1 = data[:, 3]
    fx2 = data[:, 4]
    h = data[:, 1]
    print h
    print fx0
    print fx1
    print fx2

    print (np.log(fx0[-1])-np.log(fx0[0]))/(np.log(h[-1])-(np.log(h[0])))
    print (np.log(fx1[-1])-np.log(fx1[0]))/(np.log(h[-1])-(np.log(h[0])))
    print (np.log(fx2[-1])-np.log(fx2[0]))/(np.log(h[-1])-(np.log(h[0])))
    # print (fx2[-1]-fx2[0])/(h[-1]-h[0])
    return

def run_linear(Nell, Nint):

    ffunc = ffunc_linear
    ffunc_args = np.array([0, 1])
    he = np.ones(Nell) / Nell
    x = np.linspace(0, 1, 4 * Nell + 1)

    Xe = get_node_locations_x(Nell, he)

    tic = time.time()
    K = define_stiffness_matrix(Nell, he, Nint)
    toc = time.time()
    print he, Nell, K
    print "Time to define stiffness matrix: %.3f (s)" % (toc - tic)

    tic = time.time()
    F = define_forcing_vector(Nell, he, ffunc=ffunc, ffunc_args=ffunc_args)
    toc = time.time()
    print F
    print np.array([1 / 96., 1. / 16., 1 / 8., 3. / 16.])
    print "Time to define forcing vector: %.3f (s)" % (toc - tic)

    tic = time.time()
    d = solve_for_d(K, F)
    toc = time.time()
    print d
    print np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384.])
    print "Time to solve for d: %.3f (s)" % (toc - tic)

    tic = time.time()
    u = solve_for_displacements(d, Nell, he, g=0)
    toc = time.time()
    print u
    print np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384., 0])
    print "Time to solve for u(x): %.3f (s)" % (toc - tic)

    print "Finished"

    error = quadrature(Xe, he, u, ffunc_args=ffunc_args)
    print error
    plot_displacements(u, x, he, Nell, ffunc=ffunc, ffunc_args=ffunc_args)

def run_constant(Nell, Nint, p, E=1, I=1):

    ffunc = ffunc_constant
    ffunc_args = np.array([1])
    he = np.ones(Nell) / Nell
    x = np.linspace(0, 1, 4 * Nell + 1)

    ID = get_id('cantilever L', Nell, p)
    Xe = get_node_locations_x(Nell, he)

    tic = time.time()
    K = define_stiffness_matrix(Nell, he, Nint, p, ID, E, I)
    toc = time.time()
    print he, Nell, K
    print "Time to define stiffness matrix: %.3f (s)" % (toc - tic)

    tic = time.time()
    F = define_forcing_vector(Nell, he, ffunc=ffunc, ffunc_args=ffunc_args)
    toc = time.time()
    print F
    # print np.array([1 / 96., 1. / 16., 1 / 8., 3. / 16.])
    print "Time to define forcing vector: %.3f (s)" % (toc - tic)

    tic = time.time()
    d = solve_for_d(K, F)
    toc = time.time()
    print d
    # print np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384.])
    print "Time to solve for d: %.3f (s)" % (toc - tic)

    tic = time.time()
    u = solve_for_displacements(d, Nell, he, g=0)
    toc = time.time()
    print u
    # print np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384., 0])
    print "Time to solve for u(x): %.3f (s)" % (toc - tic)

    print "Finished"

    error = quadrature(Xe, he, u, ffunc_args=ffunc_args)
    print error
    plot_displacements(u, x, he, Nell, ffunc=ffunc, ffunc_args=ffunc_args)

def run_quadratic(Nell, Nint, p, E=1, I=1):

    ffunc = ffunc_quadratic
    ffunc_args = np.array([0, 0, 1])
    he = np.ones(Nell) / Nell
    x = np.linspace(0, 1, 4 * Nell + 1)

    ID = get_id('cantilever L', Nell, p)
    Xe = get_node_locations_x(Nell, he)

    tic = time.time()
    K = define_stiffness_matrix(Nell, he, Nint, p, ID, E, I)
    toc = time.time()
    print he, Nell, K
    print "Time to define stiffness matrix: %.3f (s)" % (toc - tic)

    tic = time.time()
    F = define_forcing_vector(Nell, he, ffunc=ffunc, ffunc_args=ffunc_args)
    toc = time.time()
    print F
    print np.array([1 / 96., 1. / 16., 1 / 8., 3. / 16.])
    print "Time to define forcing vector: %.3f (s)" % (toc - tic)

    tic = time.time()
    d = solve_for_d(K, F)
    toc = time.time()
    print d
    print np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384.])
    print "Time to solve for d: %.3f (s)" % (toc - tic)

    tic = time.time()
    u = solve_for_displacements(d, Nell, he, g=0)
    toc = time.time()
    print u
    print np.array([1. / 6., 21. / 128., 7. / 48., 37. / 384., 0])
    print "Time to solve for u(x): %.3f (s)" % (toc - tic)

    print "Finished"

    error = quadrature(Xe, he, u, ffunc_args=ffunc_args)
    print error
    plot_displacements(u, x, he, Nell, ffunc=ffunc, ffunc_args=ffunc_args)



if __name__ == "__main__":

    # get_lm(10)
    # get_slope()
    # exit()
    # plot_error()
    # exit()
    #
    # input variables

    p = 2                       # basis function order
    Nell = 4                    # number of elements
    Nint = 1                    # number of quadrature intervals
    run_constant(Nell, Nint, p)
    # run_linear(Nell, Nint)
    run_quadratic(Nell, Nint, p)
