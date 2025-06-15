

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from torch_geometric.nn import GraphSAGE, SAGEConv

from RVDeformer.model.gconv import GNNLayer
from RVDeformer.model.pc_encoder.pointnet import ResPointNet
from RVDeformer.geo import adj_mat2list


class ListGrouping(nn.Module):
    def __init__(self, group_method='mean'):
        super(ListGrouping, self).__init__()
        
        self.group_method = group_method
        
    
    def forward(self, feats_list):
        '''
        group a list of feats.
        
        - feats_list: list of tensor, each element has size: (hi, c), hi are different and len(feats_list) == batch_size
        
        Return:
        -------
        - grouped_feats: (batch_size, c)
        '''
        
        if self.group_method == 'max':
            feats_list = [f.max(0)[0] for f in feats_list]
            
        elif self.group_method == 'mean':
            feats_list = [f.mean(0) for f in feats_list]
            
        out_feats = torch.stack(feats_list)
        
        return out_feats
        

class ALLFeatsMixer(nn.Module):
    def __init__(self):
        super(ALLFeatsMixer, self).__init__()
        pass

    def forward(self, feats1, feats2):
        '''
        Parameters:
        -----------
        - feats1: base features, (batch_size, C1, N)
        - feats2: features to be added, (batch_size, C2)

        Return:
        -------
        mixed_feature: (batch_size, C1+C2, N)
        '''
        
        B, C, L = feats1.size()
        feats2 = feats2.unsqueeze(-1).repeat(1,1,L)
        
        return torch.concat([feats1, feats2], 1)
    
      
   
class MixLayer(nn.Module):
    '''
    PC features and mesh features mix layer.
    
    Parameters:
    -----------
    mesh_feats: (batch_size, n_vertices, d_mesh)
    pc_feats: (batch_size, n_pc, d_pc)
    
    Return: (batch_size, n_vertices, d_mix)
    
    '''
    def __init__(self): 
        
        super(MixLayer, self).__init__()
        self.feat_mix = ALLFeatsMixer()
        self.mix_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
    
    def forward(self, mesh_feats, pc_feats):
        
        mesh_feats = mesh_feats.transpose(2,1).contiguous()
        mix_feats = self.feat_mix(mesh_feats, pc_feats) # (batch_size, d, N_v)
        mix_feats = mix_feats.transpose(2,1).contiguous() # (batch_size, N_v, d)
        mix_feats = self.mix_mlp(mix_feats) # (batch_size, N_v, d')
        
        return mix_feats
    
    
def make_pad_mask(lengths, max_len, device):

    mask = torch.ones(len(lengths), max_len, dtype=torch.bool, device=device)
    for i, length in enumerate(lengths):
        mask[i, :length] = False

    return mask


class AttentionMixer(nn.Module):
    
    def __init__(self, mesh_feats_dim, pc_feats_dim, num_heads=1):
        super(AttentionMixer, self).__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=mesh_feats_dim, 
            num_heads=num_heads, 
            kdim=pc_feats_dim, 
            vdim=pc_feats_dim,
            batch_first=True
        )
    
    
    def forward(self, mesh_feats, pc_feats_list):
        
        # make inputs
        lengths = [len(pc_feats) for pc_feats in pc_feats_list]
        pc_feats_padded = pad_sequence(pc_feats_list, batch_first=True)
        pc_feats_mask = make_pad_mask(lengths, max(lengths), device=pc_feats_padded.device)

        # cross attention
        mix_feats, attn_weights = self.attn(
            key=pc_feats_padded, 
            value=pc_feats_padded, 
            query=mesh_feats, 
            key_padding_mask=pc_feats_mask
        )
        mix_feats = torch.cat([mesh_feats, mix_feats], -1)
        
        return mix_feats
    
    
class TransformerMixer(nn.Module):
    
    def __init__(self, mesh_feats_dim, pc_feats_dim, num_heads=1):
        super(TransformerMixer, self).__init__()

        self.transformer_decoder = nn.TransformerDecoderLayer(
            d_model=mesh_feats_dim, 
            nhead=num_heads, 
            batch_first=True
        )
    
    
    def forward(self, mesh_feats, pc_feats_list):
        
        # make inputs
        lengths = [len(pc_feats) for pc_feats in pc_feats_list]
        pc_feats_padded = pad_sequence(pc_feats_list, batch_first=True)
        pc_feats_mask = make_pad_mask(lengths, max(lengths), device=pc_feats_padded.device)
        
        mix_feats = self.transformer_decoder(
            tgt=mesh_feats,
            memory=pc_feats_padded,
            memory_key_padding_mask=pc_feats_mask
        )
        mix_feats = torch.cat([mesh_feats, mix_feats], -1)

        return mix_feats
    
    
class PoolingMixer(nn.Module):
    
    def __init__(self, ):
        super(PoolingMixer, self).__init__()
        
        self.pc_group = ListGrouping('mean')
        self.mix = MixLayer()        
    
    
    def forward(self, mesh_feats, pc_feats_list):

        grouped_pc_feats = self.pc_group(pc_feats_list)
        mix_feats = self.mix(mesh_feats, grouped_pc_feats)

        return mix_feats
    
    
class GlobalLocalMixer(nn.Module):
    
    def __init__(self, mesh_feats_dim, pc_feats_dim, num_heads=1):
        super(GlobalLocalMixer, self).__init__()
        self.global_mix = PoolingMixer()
        self.local_mix = AttentionMixer(mesh_feats_dim, pc_feats_dim, num_heads)
        
        
    def forward(self, mesh_feats, pc_feats_list):

        mesh_feats_dim = mesh_feats.size()[-1]
        global_feats = self.global_mix(mesh_feats, pc_feats_list)
        local_feats = self.local_mix(mesh_feats, pc_feats_list)[:, :, :mesh_feats_dim]

        return torch.cat([global_feats, local_feats], -1)


class GNNModel(nn.Module):
    def __init__(
        self, layer_name, in_channels: int,
                hidden_channels: int,
                num_layers: int,
                adj_mat,
                out_channels: int,
                **kwargs
    ):
        super(GNNModel, self).__init__()
        
        if layer_name == 'graph_sage':
            self.gcn = GraphSAGE(in_channels, hidden_channels, num_layers-1, **kwargs)
            self.gconv = SAGEConv(hidden_channels, out_channels)
        else:
            raise NotImplementedError('Not implemented gnn model: ', layer_name)
        
        self.edge_index = adj_mat2list(adj_mat)
    
    
    def forward(self, x):
        '''
        x: (batch_size, n_points, dim_feats)
        '''
        
        hidden_batch, out_batch = [], []
        for batch in x:
            
            hidden = self.gcn(batch, self.edge_index)
            out = self.gconv(F.relu(hidden), self.edge_index)
            
            hidden_batch.append(hidden)
            out_batch.append(out)
        
        hidden_batch = torch.stack(hidden_batch)
        out_batch = torch.stack(out_batch) 
        
        return out_batch, hidden_batch

    
class GDeform(nn.Module):
    def __init__(
        self, points_feats_dim, 
        pc_feats_dim, 
        hidden_dim, 
        adj_mat, 
        mixer='pooling', 
        gnn_name='graph_sage',
        pre_gcn_num_layer=3,
        post_gcn_num_layer=3
    ):
        '''
        - Mixer: {'pooling', 'attention'} 
        '''
        super(GDeform, self).__init__()
        
        if mixer == 'pooling':            
            self.mix_layer = PoolingMixer()
            post_gcn_input_dim = hidden_dim + pc_feats_dim + 3
        elif mixer == 'attention':
            self.mix_layer = AttentionMixer(
                mesh_feats_dim=hidden_dim,
                pc_feats_dim=pc_feats_dim
            )
            post_gcn_input_dim = hidden_dim + hidden_dim + 3
        elif mixer == 'transformer':
            self.mix_layer = TransformerMixer(
                mesh_feats_dim=hidden_dim,
                pc_feats_dim=pc_feats_dim
            )
            post_gcn_input_dim = hidden_dim + hidden_dim + 3
        elif mixer == 'global_local':
            self.mix_layer = GlobalLocalMixer(
                mesh_feats_dim=hidden_dim,
                pc_feats_dim=pc_feats_dim       
            )
            post_gcn_input_dim = hidden_dim + hidden_dim + pc_feats_dim + 3
        else: 
            raise NotImplementedError(f'the mixer layer: {mixer} is not implemented.')
        
        if gnn_name in ['graph_sage']:
            self.pre_gcn = GNNModel(gnn_name, points_feats_dim, hidden_dim, pre_gcn_num_layer, adj_mat, 3)
            self.post_gcn = GNNModel(gnn_name, post_gcn_input_dim, 256, post_gcn_num_layer, adj_mat, 3)
        else:
            pass
        
        
    def forward(self, mesh_feats, pc_feats_list, mesh_points):
        
        # Mesh Encoding
        _, mesh_feats = self.pre_gcn(mesh_feats) # _, (batch_size, N_v, hidden_dim)

        # PC-M Fusion
        mix_feats = self.mix_layer(mesh_feats, pc_feats_list)
        mix_feats = torch.concat([mesh_points, mix_feats], -1)
        
        # Delta prediciton
        delta_points, mesh_feats = self.post_gcn(mix_feats) # _, (batch_size, N_v, hidden_dim)

        return delta_points, mesh_feats

    
class RVDeformNet(nn.Module):
    def __init__(self, adj_mat, model_args):
        super(RVDeformNet, self).__init__()
        # self.use_template_label = model_args.get('use_template_label', True)
        # self.use_pc_label = model_args.get('use_pc_label', True)
        
        self.feats_dim_mesh = model_args.get('feats_dim_mesh', 0)
        self.feats_dim_pc = model_args.get('feats_dim_pc', 0)
        self.learn_delta = model_args.get('learn_delta', True)
        self.use_coord_head = model_args.get('use_coord_head', True)
        pc_input_dim = self.feats_dim_pc + 3
        mesh_net_dim = self.feats_dim_mesh + 3
        
        # 1. PC Net      
        encoder_name = model_args.get('encoder_name', 'mlp')  
        if encoder_name == 'res_pointnet':
            self.pc_net = ResPointNet(dim=pc_input_dim)
        else:
            raise Exception(f'Unknown encoder: {encoder_name}')
        
        # 2. Mesh deform layer        
        use_self_loop = model_args.get('self_loop', False)
        if use_self_loop:
            adj_mat_list = [adj_mat + torch.eye(adj_mat.size()[0], device=adj_mat.device) for adj_mat in adj_mat_list]
        self.deform_list = nn.ModuleList()
        gnn_name = model_args.get('gnn_name', '')
        self.deform_list.append(GDeform(mesh_net_dim, 128, 128, adj_mat, model_args.get('mixer', 'pooling'), gnn_name))
        for _ in range(model_args['num_blocks'] - 1):
            self.deform_list.append(GDeform(256+3, 128, 128, adj_mat, model_args.get('mixer', 'pooling'), gnn_name))
        
        if self.use_coord_head:
            if gnn_name in ['graph_sage']:
                self.coord_head = GNNLayer(gnn_name, 256+3, 3, adj_mat=adj_mat)
                
        
                
    def forward(self, mesh_points, pc_points, mesh_feats, pc_feature):
        '''
        mesh_points: (N_v, 3)
        mesh_feature: (N_v, feats_dim_mesh)
        pc_points: list of (Ni, 3)
        pc_feats: list of (Ni, feats_dim_pc)
        '''
        
        # PC encoder
        if self.feats_dim_pc > 0:
            pc_feats_list = [torch.concat([f1, f2], 1) for f1, f2 in zip(pc_points, pc_feature)]
        else:
            pc_feats_list = pc_points
        pc_feats_list = self.pc_net(pc_feats_list) # list of (hi, d)
        
        # Mesh feats     
        if self.feats_dim_mesh == 0:
            mesh_feats = torch.tensor([], device=mesh_points.device)
            
        # Deformation
        for deform in self.deform_list: 
            mesh_feats = torch.cat([mesh_points, mesh_feats], -1) # (batch_size, N_v, 3+feats_dim_mesh)
            delta_points, mesh_feats = deform(mesh_feats, pc_feats_list, mesh_points)
            if self.learn_delta:
                mesh_points += delta_points
            else:
                mesh_points = delta_points
            
        if self.use_coord_head:
            mesh_feats = torch.concat([mesh_points, mesh_feats], -1)
            mesh_feats = F.relu(mesh_feats)
            mesh_points = self.coord_head(mesh_feats) #  (batch_size, N_v, 3)
            
        return mesh_points
        