import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import time
import matplotlib.pylab as plt


def define_stiffness_matrix(Nell, he):
    k_basis = np.zeros((2,2))
    K = np.zeros((Nell, Nell))
    LM = get_lm(Nell)

    for e in np.arange(0, Nell):

        # get base k matrix for element e
        for a in np.arange(0, 2):
            for b in np.arange(0, 2):
                k_basis[a,b] = ((-1)**(a+b))/he[e]

        # populate the stifness matrix for entries corresponding to element e
        for a in np.arange(0, 2):
            if LM[a, e] == 0:
                continue
            for b in np.arange(0, 2):
                if LM[b, e] == 0:
                    continue
                K[int(LM[a,e]-1), int(LM[b,e]-1)] += k_basis[a,b]

    return K

def define_forcing_vector(Nell, he, ffunc=0):

    f_e = np.zeros(2)
    F = np.zeros(Nell)
    x1 = 0.
    for e in np.arange(0, Nell):
        x2 = x1 + he[e]
        f_e = (he[e]/6.)*np.array([2.*forcing_function(x1, ffunc) + forcing_function(x2, ffunc),
                                  forcing_function(x1, ffunc) + 2.*forcing_function(x2, ffunc)])
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

def forcing_function(x, ffunc=0):

    f = 0

    if ffunc == 0:
        f = 1.
    elif ffunc == 1:
        f = x
    elif ffunc == 2:
        f = x**2
    else:
        ValueError("ffunc must be one of [0, 1, 2]")
    return f

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

def plot_displaccements(u, x, he, Nell, q=1, ffunc=1):

    plt.rcParams.update({'font.size': 22})

    x_ex = np.linspace(0, 1., 100)

    x_el = get_node_locations_x(Nell, he)

    u_ex = get_u_of_x_exact(x_ex, q, ffunc)

    u_a = get_u_of_x_approx(x, u, he)
    plt.figure()
    plt.plot(x_ex, u_ex, label="Exact sol.", linewidth=3)
    # plt.plot(x_el, u, '-s', label="Approx. sol. (nodes)")
    plt.plot(x, u_a, '--r', markerfacecolor='none', label="Approx. sol.", linewidth=3)

    plt.xlabel('X Position')
    plt.ylabel("Displacement")
    functions = ["$f(x)=c$", "$f(x)=x$", "$f(x)=x^2$"]
    plt.title(functions[ffunc]+", $n=%i$" %Nell, y=1.02)
    plt.legend(loc=3, frameon=False)
    plt.tight_layout()
    plt.savefig("displacement_func%i_Nell%i.pdf" %(ffunc, Nell))
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

def get_u_of_x_exact(x, q, ffunc=1):

    u_ex = 0.

    if ffunc == 0:
        u_ex = q*(1.-x**2)/2.
    elif ffunc == 1:
        u_ex = q*(1.-x**3)/6.
    elif ffunc == 2:
        u_ex = q * (1. - x ** 4) / 12.

    return u_ex

def quadrature(Xe, he, ue, ffunc):

    # print Xe, he, ue, ffunc
    # quit()
    Nell = he.size
    xi = np.array([-np.sqrt(3./5.), 0., np.sqrt(3./5.)])
    w = np.array([5./9., 8./9., 5./9.])

    error_squared = 0.0
    for el in np.arange(0, Nell):
        x = x_of_xi(xi, Xe, he, el)

        dxdxi = he[el]/2

        ux = get_u_of_x_exact(x, q=1, ffunc=ffunc)
        ua = get_u_of_x_approx(x, ue, he)

        error_squared += np.sum(((ux-ua)**2)*dxdxi*w)
    error = np.sqrt(error_squared)

    return error

def x_of_xi(xi, Xe, he, el):
    # print xi, Xe.size, he, el
    # quit()
    x = (he[el]*xi+Xe[el]+Xe[el+1])/2.

    return x

def get_lm(Nell):
    """
    Populates the location matrix, LM, to track the location of data in the global stiffness matrix, K
    :param Nell: Number of elements
    :return LM: Location matrix for data in the stiffness matrix, K
    """

    LM = np.zeros([2, Nell])

    for e in np.arange(0, Nell):
        LM[0, e] = e + 1
        LM[1, e] = e + 2
    LM[-1, -1] = 0

    return LM

def plot_error():


    n = np.array([10, 100, 1000, 10000])
    error = np.zeros([3, n.size])
    h = np.ones(n.size)/n
    print h, n
    for ffunc, i in zip(np.array([0, 1, 2]), np.arange(0, 3)):
        for Nell, j in zip(n, np.arange(n.size)):

            # print Nell
            he = np.ones(Nell) / Nell
            Xe = get_node_locations_x(Nell, he)
            K = define_stiffness_matrix(Nell, he)
            F = define_forcing_vector(Nell, he, ffunc=ffunc)
            d = solve_for_d(K, F)
            u = solve_for_displacements(d, Nell, he, g=0)
            error[i,j] = quadrature(Xe, he, u, ffunc)
            print "ffunc: %i, Nell: %i, Error: %f" % (ffunc, Nell, error[i, j])

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

if __name__ == "__main__":

    # get_lm(10)
    # get_slope()
    # exit()
    # plot_error()
    # exit()
    #
    # input variables
    Nell = 10
    ffunc = 2
    # for ffunc in np.array([0, 1, 2]):
    #     for Nell in np.array([10, 100]):
    he = np.ones(Nell)/Nell
    x = np.linspace(0, 1, 4*Nell+1)

    Xe = get_node_locations_x(Nell, he)
    # error
    # er_4_el = 0.0051

    tic = time.time()
    K = define_stiffness_matrix(Nell, he)
    toc = time.time()
    print he, Nell, K
    print "Time to define stiffness matrix: %.3f (s)" % (toc-tic)

    tic = time.time()
    F = define_forcing_vector(Nell, he, ffunc=ffunc)
    toc = time.time()
    print F
    print np.array([1/96., 1./16., 1/8., 3./16.])
    print "Time to define forcing vector: %.3f (s)" % (toc - tic)

    tic = time.time()
    d = solve_for_d(K, F)
    toc = time.time()
    print d
    print np.array([1./6., 21./128., 7./48., 37./384.])
    print "Time to solve for d: %.3f (s)" % (toc - tic)

    tic = time.time()
    u = solve_for_displacements(d, Nell, he, g=0)
    toc = time.time()
    print u
    print np.array([1./6., 21./128., 7./48., 37./384., 0])
    print "Time to solve for u(x): %.3f (s)" % (toc - tic)

    print "Finished"

    error = quadrature(Xe, he, u, ffunc)
    print error
    plot_displaccements(u, x, he, Nell, ffunc=ffunc)


    # pre compute
    # Nnodes = Nell + 1

    # x = 2.0
    # node = 1
    # Nodes = np.array([0, 0.5, 1])
    # Lengths = Nodes/Nodes.size
    #
    # x = Nodes[0]
    #
    # print x, node, Nodes, Lengths
    # print basis_functions(x, node, Nodes, Lengths)
