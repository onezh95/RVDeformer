# RVDeformer
The implementation of RVDeformer

## Usage

1. RVDeformNet
~~~ python
from RVDeformer.model.rv_deform_net import RVDeformNet

model_args = {
  'encoder_name': 'res_pointnet',
  'gnn_name': 'graph_sage'
}
adj_mat = ... # adj matrix of mesh edges, (N_v, N_v)
rvdeform_net = RVDeformNet(adj_mat, model_args)
~~~

2. Data
~~~ python
from torch.utils.data.dataloader import DataLoader
dataset = ...
dataloader = Dataloader(dataset)
~~~

3. Training or Inference
~~~ python
'''
mesh_points: (N_v, 3)
mesh_feature: (N_v, feats_dim_mesh)
pc_points: list of (Ni, 3)
pc_feats: list of (Ni, feats_dim_pc)
'''

for batch in dataloader:
  mesh_points, pc_points, mesh_feats, pc_feature = batch
  output = rvdeform_net(mesh_points, pc_points, mesh_feats, pc_feature)
~~~
