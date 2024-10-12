import torch
import torch.nn as nn
import math


class GAGStateEncoder(nn.Module):
    """
    Gated Attentive GNN state encoder network.
    """
    EPSILON = 1e-6

    def __init__(self, cfg, agent):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.numerical_feature_encoder = self.create_numerical_feature_encoder(cfg)
        self.node_encoder = self.create_node_encoder(cfg, agent)
        self.num_gcn_layers = cfg['num_gcn_layers']
        self.num_edge_fc_layers = cfg['num_edge_fc_layers']

        self.edge_fc_layers1 = self.create_edge_fc_layers(cfg)
        self.edge_fc_layers2 = self.create_edge_fc_layers(cfg)

        self.max_num_nodes = cfg['max_num_nodes']
        self.max_num_edges = cfg['max_num_edges']

        self.attention_layer1 = nn.MultiheadAttention(cfg['gcn_node_dim'], cfg['num_attention_heads'])
        self.attention_query_layer1 = nn.Linear(2*cfg['gcn_node_dim'], cfg['gcn_node_dim'])
        self.attention_key_layer1 = nn.Linear(2*cfg['gcn_node_dim'], cfg['gcn_node_dim'])
        self.attention_value_layer1 = nn.Linear(2*cfg['gcn_node_dim'], cfg['gcn_node_dim'])

        self.attention_layer2 = nn.MultiheadAttention(cfg['gcn_node_dim'], cfg['num_attention_heads'])
        self.attention_query_layer2 = nn.Linear(2*cfg['gcn_node_dim'], cfg['gcn_node_dim'])
        self.attention_key_layer2 = nn.Linear(2*cfg['gcn_node_dim'], cfg['gcn_node_dim'])
        self.attention_value_layer2 = nn.Linear(2*cfg['gcn_node_dim'], cfg['gcn_node_dim'])

        self.layer_norm1 = nn.LayerNorm(cfg['gcn_node_dim'])
        self.layer_norm2 = nn.LayerNorm(cfg['gcn_node_dim'])
        self.dropout = nn.Dropout(0.5)

        self.output_policy_road_size = cfg['gcn_node_dim']
        self.output_value_size = cfg['gcn_node_dim'] + cfg['state_encoder_hidden_size'][-1] + 2

        self.positional_encodings = nn.Parameter(torch.zeros(1, 1177, cfg['gcn_node_dim']))
        self.gated_update = GatedUpdate(cfg['gcn_node_dim'])

    def create_numerical_feature_encoder(self, cfg):
        """Create the numerical feature encoder."""
        feature_encoder = nn.Sequential()
        for i in range(len(cfg['state_encoder_hidden_size'])):
            if i == 0:
                feature_encoder.add_module('flatten_{}'.format(i), nn.Flatten())
                feature_encoder.add_module('linear_{}'.format(i), nn.Linear(self.agent.numerical_feature_size, cfg['state_encoder_hidden_size'][i]))
            else:
                feature_encoder.add_module('linear_{}'.format(i), nn.Linear(cfg['state_encoder_hidden_size'][i - 1], cfg['state_encoder_hidden_size'][i]))
            feature_encoder.add_module('tanh_{}'.format(i), nn.Tanh())
        return feature_encoder

    def create_node_encoder(self, cfg, agent):
        feature_encoder = nn.Sequential()
        feature_encoder.add_module('linear_1', nn.Linear(agent.node_dim, cfg['gcn_node_dim']))
        feature_encoder.add_module('tanh_2', nn.Tanh())
        return feature_encoder

    def create_edge_fc_layers(self, cfg):
        """Create the edge fc layers."""
        def create_edge_fc():
            seq = nn.Sequential()
            for i in range(self.num_edge_fc_layers):
                seq.add_module('linear_{}'.format(i), nn.Linear(cfg['gcn_node_dim'], cfg['gcn_node_dim']))
                seq.add_module('tanh_{}'.format(i), nn.Tanh())
            return seq

        edge_fc_layers = nn.ModuleList()
        for _ in range(self.num_gcn_layers):
            edge_fc_layers.append(create_edge_fc())
        return edge_fc_layers

    def scatter_count(self, h_edges, indices, max_num_nodes):
        """Aggregate edge embeddings to nodes."""
        batch_size = h_edges.shape[0]
        num_latents = h_edges.shape[2]

        h_nodes = torch.zeros(batch_size, max_num_nodes, num_latents).to(h_edges.device)
        count_edge = torch.zeros_like(h_nodes)
        count = torch.ones_like(h_edges).float()

        idx = indices.unsqueeze(-1).expand(-1, -1, num_latents)
        h_nodes = h_nodes.scatter_add_(1, idx, h_edges)
        count_edge = count_edge.scatter_add_(1, idx, count)
        return h_nodes, count_edge

    def gather_to_edges(self, h_nodes, edge_index, edge_fc_layer):
        """Gather node embeddings to edges."""
        h_edges_12 = torch.gather(h_nodes, 1, edge_index[:, :, 0].unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        h_edges_12 = edge_fc_layer(h_edges_12.to(torch.float32))

        h_edges_21 = torch.gather(h_nodes, 1, edge_index[:, :, 1].unsqueeze(-1).expand(-1, -1, h_nodes.size(-1)))
        h_edges_21 = edge_fc_layer(h_edges_21.to(torch.float32))

        h_edges = (h_edges_12 + h_edges_21) / 2

        return h_edges, h_edges_12, h_edges_21

    def scatter_to_nodes(self, h_nodes, h_edges, edge_index, max_num_nodes):
        """Scatter edge embeddings to nodes."""
        _, h_edges_12, h_edges_21 = h_edges
        h_nodes_1, count_1 = self.scatter_count(h_edges_21, edge_index[:, :, 0], max_num_nodes)
        h_nodes_2, count_2 = self.scatter_count(h_edges_12, edge_index[:, :, 1], max_num_nodes)
        h_nodes_combined = (h_nodes_1 + h_nodes_2) / (count_1 + count_2 + self.EPSILON)

        h_nodes_updated = self.gated_update(h_nodes, h_nodes_combined)

        return h_nodes_updated

    def self_attention(self, h_nodes, attention_layer, query_layer, key_layer, value_layer, layer_norm):
        """Self attention with layer normalization."""
        query = query_layer(h_nodes).transpose(0, 1)
        keys = key_layer(h_nodes).transpose(0, 1)
        values = value_layer(h_nodes).transpose(0, 1)
        h_current_node_attended, _ = attention_layer(query, keys, values)
        h_current_node_attended = h_current_node_attended.transpose(0, 1)
        h_current_node_attended = layer_norm(h_current_node_attended + h_nodes)
        return self.dropout(h_current_node_attended)


    def forward(self, x):
        numerical, node_feature, edge_index_dis, edge_index_od, node_mask, stage = self.batch_data(x)
        h_numerical_features = self.numerical_feature_encoder(numerical.to(torch.float32))
        h_nodes = self.node_encoder(node_feature.to(torch.float32))

        pos_enc = self.positional_encodings[:, :h_nodes.size(1), :]
        pos_enc = pos_enc.to(h_nodes.device)
        h_nodes = h_nodes + pos_enc

        h_nodes_1 = torch.zeros_like(h_nodes)
        h_nodes_2 = torch.zeros_like(h_nodes)

        for edge_fc_layer1 in self.edge_fc_layers1:
            h_edges = self.gather_to_edges(h_nodes, edge_index_dis, edge_fc_layer1)
            h_nodes_1 = self.scatter_to_nodes(h_nodes, h_edges, edge_index_dis, node_feature.shape[1])

        for edge_fc_layer2 in self.edge_fc_layers2:
            h_edges = self.gather_to_edges(h_nodes, edge_index_od, edge_fc_layer2)
            h_nodes_2 = self.scatter_to_nodes(h_nodes, h_edges, edge_index_od, node_feature.shape[1])

        h_nodes = torch.cat([h_nodes_1, h_nodes_2], dim=-1)
        build_idx = torch.where(node_feature[:, :, -5][0] > 0)[0].tolist()
        h_node_attended = self.attention1(h_nodes, h_nodes[:, build_idx, :])

        h_nodes_mean = self.mean_features(h_node_attended)

        state_value = torch.cat([h_numerical_features, h_nodes_mean, stage], dim=-1)
        state_policy = torch.cat([h_node_attended.to(torch.float32)], dim=-1)
        return state_policy, state_value, node_mask, stage

    @staticmethod
    def batch_data(x):
        numerical, node_feature, edge_index_dis, edge_index_od, node_mask, stage = zip(*x)
        numerical = torch.stack(numerical)
        node_feature = torch.stack(node_feature)
        edge_index_dis = torch.stack(edge_index_dis)
        edge_index_od = torch.stack(edge_index_od)
        node_mask = torch.stack(node_mask)
        stage = torch.stack(stage)
        return numerical, node_feature, edge_index_dis, edge_index_od, node_mask, stage

    @staticmethod
    def mean_features(h, mask=None):
        if mask is not None:
            mean_h = (h * mask.unsqueeze(-1).float()).sum(dim=1) / mask.float().sum(dim=1, keepdim=True)
        else:
            mean_h = h.mean(dim=1)
        return mean_h

    def attention1(self, h_nodes_q, h_nodes_kv):
        """Attention mechanism for query and key-value nodes."""
        query = self.attention_query_layer1(h_nodes_q).transpose(0, 1)
        keys = self.attention_key_layer1(h_nodes_kv).transpose(0, 1)
        values = self.attention_value_layer1(h_nodes_kv).transpose(0, 1)
        h_current_node_attended, _ = self.attention_layer1(query, keys, values)
        h_current_node_attended = h_current_node_attended.transpose(0, 1).squeeze(1)
        return h_current_node_attended

class GatedUpdate(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.node_dim = node_dim

        self.gate_layer = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.Tanh()
        )

        self.update_layer = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.Tanh()
        )

    def forward(self, current_nodes, incoming_nodes):
        combined = torch.cat([current_nodes, incoming_nodes], dim=-1)
        gate = self.gate_layer(combined)
        update = self.update_layer(combined)

        new_nodes = gate * update + (1 - gate) * current_nodes

        return new_nodes


class RLStateEncoder(nn.Module):
    EPSILON = 1e-6

    def __init__(self, cfg, agent):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.numerical_feature_encoder = self.create_numerical_feature_encoder(cfg)
        self.node_encoder = self.create_node_encoder(cfg)
        self.num_gcn_layers = cfg['num_gcn_layers']

        self.max_num_nodes = cfg['max_num_nodes']
        self.max_num_edges = cfg['max_num_edges']

        self.attention_layer1 = nn.MultiheadAttention(cfg['gcn_node_dim'], cfg['num_attention_heads'])
        self.attention_query_layer1 = nn.Linear(cfg['gcn_node_dim'], cfg['gcn_node_dim'])
        self.attention_key_layer1 = nn.Linear(cfg['gcn_node_dim'], cfg['gcn_node_dim'])
        self.attention_value_layer1 = nn.Linear(cfg['gcn_node_dim'], cfg['gcn_node_dim'])

        self.output_policy_road_size = cfg['gcn_node_dim']
        self.output_value_size = cfg['gcn_node_dim'] + cfg['state_encoder_hidden_size'][-1] + 2

    def create_numerical_feature_encoder(self, cfg):
        """Create the numerical feature encoder."""
        feature_encoder = nn.Sequential()
        for i in range(len(cfg['state_encoder_hidden_size'])):
            if i == 0:
                feature_encoder.add_module('flatten_{}'.format(i),
                                           nn.Flatten())
                feature_encoder.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(self.agent.numerical_feature_size,
                              cfg['state_encoder_hidden_size'][i]))
            else:
                feature_encoder.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(cfg['state_encoder_hidden_size'][i - 1],
                              cfg['state_encoder_hidden_size'][i]))
            feature_encoder.add_module('tanh_{}'.format(i), nn.Tanh())
        return feature_encoder

    def create_node_encoder(self, cfg):
        feature_encoder = nn.Sequential()
        feature_encoder.add_module(
            'linear_{}'.format(1),
            nn.Linear(7, cfg['gcn_node_dim']))
        feature_encoder.add_module('tanh_{}'.format(2), nn.Tanh())

        return feature_encoder

    def self_attention1(self, h_nodes):
        """Self attention."""
        query = self.attention_query_layer1(h_nodes).transpose(0, 1)
        keys = self.attention_key_layer1(h_nodes).transpose(0, 1)
        values = self.attention_value_layer1(h_nodes).transpose(0, 1)
        h_current_node_attended, _ = self.attention_layer1(query, keys, values)
        h_current_node_attended = h_current_node_attended.transpose(0, 1).squeeze(1)
        return h_current_node_attended

    def attention1(self, h_nodes_q, h_nodes_kv):
        """Self attention."""
        query = self.attention_query_layer1(h_nodes_q).transpose(0, 1)
        keys = self.attention_key_layer1(h_nodes_kv).transpose(0, 1)
        values = self.attention_value_layer1(h_nodes_kv).transpose(0, 1)
        h_current_node_attended, _ = self.attention_layer1(query, keys, values)
        h_current_node_attended = h_current_node_attended.transpose(0, 1).squeeze(1)
        return h_current_node_attended

    @staticmethod
    def batch_data(x):
        numerical, node_feature, edge_index_dis, edge_index_od, node_mask, stage = zip(*x)
        numerical = torch.stack(numerical)
        node_feature = torch.stack(node_feature)
        edge_index_dis = torch.stack(edge_index_dis)
        edge_index_od = torch.stack(edge_index_od)
        node_mask = torch.stack(node_mask)
        stage = torch.stack(stage)

        return numerical, node_feature, edge_index_dis, edge_index_od, node_mask, stage

    @staticmethod
    def mean_features(h, mask=None):
        if mask is not None:
            mean_h = (h * mask.unsqueeze(-1).float()).sum(dim=1) / mask.float().sum(dim=1, keepdim=True)
        else:
            mean_h = (h).mean(dim=1)
        return mean_h

    def forward(self, x):
        numerical, node_feature, _, _, node_mask, stage = self.batch_data(x)
        node_feature = node_feature[:, :, [0, 1, -5, -4, -3, -2, -1]]

        h_numerical_features = self.numerical_feature_encoder(numerical.to(torch.float32))
        h_nodes = self.node_encoder(node_feature.to(torch.float32))

        build_idx = torch.where(node_feature[:, :, -5][0] > 0)[0].tolist()
        h_node_attended = self.attention1(h_nodes, h_nodes[:, build_idx, :])

        h_nodes_mean = self.mean_features(h_node_attended)

        state_value = torch.cat([h_numerical_features, h_nodes_mean, stage], dim=-1)
        state_policy = torch.cat([h_node_attended.to(torch.float32)], dim=-1)

        return state_policy, state_value, node_mask, stage










