import numpy as np

class Graph():
    """ The Graph to represent the skeleton hierarchy """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 21 # MediaPipe Hand Landmarks
            self.self_link = [(i, i) for i in range(self.num_node)]
            self.neighbor_link = [
                (0, 1), (1, 2), (2, 3), (3, 4), # thumb
                (0, 5), (5, 6), (6, 7), (7, 8), # index
                (0, 9), (9, 10), (10, 11), (11, 12), # middle
                (0, 13), (13, 14), (14, 15), (15, 16), # ring
                (0, 17), (17, 18), (18, 19), (19, 20) # pinky
            ]
            self.edge = self.self_link + self.neighbor_link
            self.center = 0
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = np.zeros((3, self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                if hop == 0:
                    A[0][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                    hop]
                else:
                    A[1][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                    hop]
                    A[2][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                    hop]
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[i, j] = 1
        A[j, i] = 1

    # compute hop distance
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis
