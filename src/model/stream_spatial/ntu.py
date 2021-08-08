from .util import get_spatial_graph
from ..config import cfg_ds
num_node = cfg_ds.num_joint
# num_node = 25
self_link = [(i, i) for i in range(num_node)]
# inward_ori_index = [
#     (1, 2),
#     (2, 21),
#     (3, 21),
#     (4, 3),
#     (5, 21),
#     (6, 5),
#     (7, 6),
#     (8, 7),
#     (9, 21),
#     (10, 9),
#     (11, 10),
#     (12, 11),
#     (13, 1),
#     (14, 13),
#     (15, 14),
#     (16, 15),
#     (17, 1),
#     (18, 17),
#     (19, 18),
#     (20, 19),
#     (22, 23),
#     (23, 8),
#     (24, 25),
#     (25, 12),
#     (26,1),
# ]
inward_ori_index = [
    (2, 1),
    (21, 2),
    (3, 21),
    (4, 3),
    (5, 21),
    (6, 5),
    (7, 6),
    (8, 7),
    (9, 21),
    (10, 9),
    (11, 10),
    (12, 11),
    (13, 1),
    (14, 13),
    (15, 14),
    (16, 15),
    (17, 1),
    (18, 17),
    (19, 18),
    (20, 19),
    (22, 23),
    (23, 8),
    (24, 25),
    (25, 12),
    (26,1),
]
if num_node ==25:
    inward_ori_index.pop(-1)
    
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class NtuGraph:
    def __init__(self, labeling_mode="spatial"):
        #A: [self-link, in, out]
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == "spatial":
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A

if __name__=="__main__":
    import numpy as np
    graph = NtuGraph()
    A = graph.get_adjacency_matrix()
    A = np.round(A, decimals=2)

    def inra(M):
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in M]))
        print("")
    # print("inward: ", A[0])
    # print("outward: ", A[1])
    # print("adj: ", A[2])
    inra(A[0])
    inra(A[1])
    inra(A[2])