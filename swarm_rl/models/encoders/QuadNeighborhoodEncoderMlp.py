from sample_factory.model.model_utils import fc_layer, nonlinearity
from torch import nn

from swarm_rl.models.encoders.QuadNeighborhoodEncoder import QuadNeighborhoodEncoder


class QuadNeighborhoodEncoderMlp(QuadNeighborhoodEncoder):
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, self_obs_dim, num_use_neighbor_obs):
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)

        self.self_obs_dim = self_obs_dim

        self.neighbor_mlp = nn.Sequential(
            fc_layer(neighbor_obs_dim * num_use_neighbor_obs, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
        )

    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size):
        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        final_neighborhood_embedding = self.neighbor_mlp(obs_neighbors)
        return final_neighborhood_embedding
