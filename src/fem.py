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


def define_stiffness_matrix():

    return

def define_forcing_vector():

    return

def solve_for_d():

    return

if __name__ == "__main__":

    x = 2.0
    node = 1
    Nodes = np.array([0, 0.5, 1])
    Lengths = Nodes/Nodes.size

    x = Nodes[0]

    print x, node, Nodes, Lengths
    print basis_functions(x, node, Nodes, Lengths)
