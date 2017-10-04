import numpy as np


def basis_functions(x, node, Nodes, Lengths):
    Na = 0.0
    print x, node, Nodes, Lengths
    if node == 1:
        Na = (Nodes[1]- x)/Lengths[1]
    elif node == Nodes.size:
        Na = (x-Nodes[-1])/Lengths[-1]
    elif (x >= Nodes[node-1]) and (x<=Nodes[node]):
        Na = (x-Nodes[node-1])/Lengths[node-1]
    elif (x >= Nodes[node]) and (x<=Nodes[node+1]):
        Na = (x-Nodes[node-1])/Lengths[node-1]

    return Na


def define_stiffness_matrix(Nell, h):
    k_basis = np.zeros((2,2))
    K = np.zeros((Nell, Nell))
    # get base k matrices
    for e in np.arange(0, Nell):
        for a in np.arange(0, 2):
            for b in np.arange(0, 2):
                k_basis[a,b] = ((-1)**(a+b))/h[e]
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

def define_forcing_vector():

    return

def solve_for_d():

    return

if __name__ == "__main__":

    # input variables
    Nell = 5
    h = np.ones(Nell)/Nell
    print h, Nell, define_stiffness_matrix(Nell, h)

    # pre compute
    Nnodes = Nell + 1

    # x = 2.0
    # node = 1
    # Nodes = np.array([0, 0.5, 1])
    # Lengths = Nodes/Nodes.size
    #
    # x = Nodes[0]
    #
    # print x, node, Nodes, Lengths
    # print basis_functions(x, node, Nodes, Lengths)
