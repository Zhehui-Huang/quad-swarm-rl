import torch
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.model.model_utils import fc_layer, nonlinearity
from torch import nn

from gym_art.quadrotor_multi.quad_utils import QUADS_OBS_REPR, QUADS_NEIGHBOR_OBS_TYPE, QUADS_OBSTACLE_OBS_TYPE
from swarm_rl.models.encoders.QuadNeighborhoodEncoderAttention import QuadNeighborhoodEncoderAttention
from swarm_rl.models.encoders.QuadNeighborhoodEncoderDeepsets import QuadNeighborhoodEncoderDeepsets
from swarm_rl.models.encoders.QuadNeighborhoodEncoderMlp import QuadNeighborhoodEncoderMlp


class QuadMultiEncoder(Encoder):
    # Mean embedding encoder based on the DeepRL for Swarms Paper
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        self.self_obs_dim = QUADS_OBS_REPR[cfg.quads_obs_repr]
        self.use_obstacles = cfg.quads_use_obstacles

        # Neighbor
        neighbor_hidden_size = cfg.quads_neighbor_hidden_size
        neighbor_obs_dim = QUADS_NEIGHBOR_OBS_TYPE[cfg.quads_neighbor_obs_type]

        if cfg.quads_neighbor_obs_type == 'none':
            num_use_neighbor_obs = 0
        else:
            if cfg.quads_neighbor_visible_num == -1:
                num_use_neighbor_obs = cfg.quads_num_agents - 1
            else:
                num_use_neighbor_obs = cfg.quads_neighbor_visible_num

        self.all_neighbor_obs_size = neighbor_obs_dim * num_use_neighbor_obs

        # # Neighbor Encoder
        neighbor_encoder_out_size = 0
        self.neighbor_encoder = None

        if num_use_neighbor_obs > 0:
            neighbor_encoder_type = cfg.quads_neighbor_encoder_type
            if neighbor_encoder_type == 'mean_embed':
                self.neighbor_encoder = QuadNeighborhoodEncoderDeepsets(
                    cfg=cfg, neighbor_obs_dim=neighbor_obs_dim, neighbor_hidden_size=neighbor_hidden_size,
                    self_obs_dim=self.self_obs_dim, num_use_neighbor_obs=num_use_neighbor_obs)
            elif neighbor_encoder_type == 'attention':
                self.neighbor_encoder = QuadNeighborhoodEncoderAttention(
                    cfg=cfg, neighbor_obs_dim=neighbor_obs_dim, neighbor_hidden_size=neighbor_hidden_size,
                    self_obs_dim=self.self_obs_dim, num_use_neighbor_obs=num_use_neighbor_obs)
            elif neighbor_encoder_type == 'mlp':
                self.neighbor_encoder = QuadNeighborhoodEncoderMlp(
                    cfg=cfg, neighbor_obs_dim=neighbor_obs_dim, neighbor_hidden_size=neighbor_hidden_size,
                    self_obs_dim=self.self_obs_dim, num_use_neighbor_obs=num_use_neighbor_obs)
            elif neighbor_encoder_type == 'no_encoder':
                # Blind agent
                self.neighbor_encoder = None
            else:
                raise NotImplementedError

        if self.neighbor_encoder:
            neighbor_encoder_out_size = neighbor_hidden_size

        fc_encoder_layer = cfg.rnn_size
        # Encode Self Obs
        self.self_encoder = nn.Sequential(
            fc_layer(self.self_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg)
        )
        self_encoder_out_size = calc_num_elements(self.self_encoder, (self.self_obs_dim,))

        # Encode Obstacle Obs
        obstacle_encoder_out_size = 0
        if self.use_obstacles:
            if cfg.quads_obstacle_obs_type == 'ToFs':
                if cfg.quads_obstacle_tof_resolution == 4:
                    obstacle_obs_dim = QUADS_OBSTACLE_OBS_TYPE['ToFs_4']
                elif cfg.quads_obstacle_tof_resolution == 8:
                    obstacle_obs_dim = QUADS_OBSTACLE_OBS_TYPE['ToFs_8']
                else:
                    raise NotImplementedError(
                        f'Obstacle TOF resolution {cfg.quads_obstacle_tof_resolution} not supported!')
            elif cfg.quads_obstacle_obs_type == 'octomap':
                obstacle_obs_dim = QUADS_OBSTACLE_OBS_TYPE['octomap']
            else:
                raise NotImplementedError(f'Obstacle observation type {cfg.quads_obstacle_obs_type} not supported!')

            obstacle_hidden_size = cfg.quads_obst_hidden_size
            self.obstacle_encoder = nn.Sequential(
                fc_layer(obstacle_obs_dim, obstacle_hidden_size),
                nonlinearity(cfg),
                fc_layer(obstacle_hidden_size, obstacle_hidden_size),
                nonlinearity(cfg),
            )
            obstacle_encoder_out_size = calc_num_elements(self.obstacle_encoder, (obstacle_obs_dim,))

        total_encoder_out_size = self_encoder_out_size + neighbor_encoder_out_size + obstacle_encoder_out_size

        # This is followed by another fully connected layer in the action parameterization, so we add a nonlinearity
        # here
        self.feed_forward = nn.Sequential(
            fc_layer(total_encoder_out_size, 2 * cfg.rnn_size),
            nn.Tanh(),
        )

        self.encoder_out_size = 2 * cfg.rnn_size

    def get_out_size(self):
        return self.encoder_output_size

    def forward(self, obs_dict):
        obs = obs_dict['obs']
        obs_self = obs[:, :self.self_obs_dim]
        self_embed = self.self_encoder(obs_self)
        embeddings = self_embed
        batch_size = obs_self.shape[0]
        # Relative xyz and vxyz for the Entire Minibatch (batch dimension is batch_size * num_neighbors)
        if self.neighbor_encoder:
            neighborhood_embedding = self.neighbor_encoder(obs_self, obs, self.all_neighbor_obs_size, batch_size)
            embeddings = torch.cat((embeddings, neighborhood_embedding), dim=1)

        if self.use_obstacles:
            obs_obstacles = obs[:, self.self_obs_dim + self.all_neighbor_obs_size:]
            obstacle_embeds = self.obstacle_encoder(obs_obstacles)
            embeddings = torch.cat((embeddings, obstacle_embeds), dim=1)

        out = self.feed_forward(embeddings)
        return out
