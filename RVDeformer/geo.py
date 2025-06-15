import torch


def adj_list2mat(adj_list: torch.Tensor, n_points: int):
    adj_mat = torch.zeros(n_points, n_points, dtype=torch.int)
    for edge in adj_list:
        adj_mat[edge[0], edge[1]] = 1
        adj_mat[edge[1], edge[0]] = 1
    return adj_mat


def adj_mat2list(adj_mat: torch.Tensor):
    '''
    convert adj matrix to adj list(edge index), size of (2, n_edges),
    edges are directed
    '''
    
    return adj_mat.nonzero().t().contiguous()
