import numpy as np
import scipy as sp
import time
import matplotlib.pylab as plt


# def basis_functions(x, node, Nodes, Lengths):
#     Na = 0.0
#     print x, node, Nodes, Lengths
#     if node == 1:
#         Na = (Nodes[1]- x)/Lengths[1]
#     elif node == Nodes.size:
#         Na = (x-Nodes[-1])/Lengths[-1]
#     elif (x >= Nodes[node-1]) and (x<=Nodes[node]):
#         Na = (x-Nodes[node-1])/Lengths[node-1]
#     elif (x >= Nodes[node]) and (x<=Nodes[node+1]):
#         Na = (x-Nodes[node-1])/Lengths[node-1]
#
#     return Na


def define_stiffness_matrix(Nell, he):
    k_basis = np.zeros((2,2))
    K = np.zeros((Nell, Nell))
    # get base k matrices
    for e in np.arange(0, Nell):
        for a in np.arange(0, 2):
            for b in np.arange(0, 2):
                k_basis[a,b] = ((-1)**(a+b))/he[e]
        if e == 0:
            K[e, e] = k_basis[0,0]
            K[e+1, e+1] = k_basis[1,1]
            K[e, e + 1] = k_basis[0, 1]
            K[e + 1, e] = k_basis[1, 0]
        elif e < Nell - 1:
            K[e, e] += k_basis[0, 0]
            K[e + 1, e + 1] = k_basis[1, 1]
            K[e, e + 1] = k_basis[0, 1]
            K[e + 1, e] = k_basis[1, 0]
        else:
            K[e, e] += k_basis[0, 0]
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
            print "here", f_e
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

    d = np.matmul(np.linalg.inv(K),F)

    return d

def solve_for_displacements(d, Nell, he, g=0):

    u = np.zeros(Nell+1)
    x1 = 0.0

    u[0] = (1.-x1)*d[0]

    for e in np.arange(1, Nell):
        x1 += he[e]
        u[e] = u[e-1] + (1.-x1)*d[e]
        # u[e] = (1.-x1)*d[e]

    u[-1] = u[-2] + g
    # u[-1] = g

    return u

def plot_displaccements(u, he, Nell, q=1):

    x_el = np.zeros(Nell+1)
    x_ex = np.linspace(0, 1., 100)

    for e in np.arange(1, Nell):
        x_el[e] = x_el[e-1] + he[e-1]
    x_el[Nell] = x_el[Nell-1] + he[Nell-1]

    u_ex = q*(1.-x_ex**3)/6.

    plt.plot(x_ex, u_ex)
    plt.plot(x_el, u, 's')
    plt.show()
    return

if __name__ == "__main__":

    # input variables
    Nell = 4
    he = np.ones(Nell)/Nell

    tic = time.time()
    K = define_stiffness_matrix(Nell, he)
    toc = time.time()
    print he, Nell, K
    print "Time to define stiffness matrix: %.3f (s)" % (toc-tic)

    tic = time.time()
    F = define_forcing_vector(Nell, he, ffunc=1)
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

    plot_displaccements(u, he, Nell)


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
