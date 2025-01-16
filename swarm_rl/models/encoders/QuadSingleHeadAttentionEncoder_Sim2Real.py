import torch
from sample_factory.model.model_utils import fc_layer, nonlinearity
from torch import nn

from gym_art.quadrotor_multi.quad_utils import QUADS_OBS_REPR, QUADS_NEIGHBOR_OBS_TYPE, QUADS_OBSTACLE_OBS_TYPE
from swarm_rl.models.attention_layer import OneHeadAttention
from swarm_rl.models.encoders.QuadMultiHeadAttentionEncoder import QuadMultiHeadAttentionEncoder


class QuadSingleHeadAttentionEncoder_Sim2Real(QuadMultiHeadAttentionEncoder):
    def __init__(self, cfg, obs_space, is_critic):
        super().__init__(cfg, obs_space, is_critic)

        # Internal params
        self.is_critic = is_critic
        self.use_separate_critic = (cfg.quads_critic_rnn_size > 0)
        self.obst_obs_diff_ac = (cfg.quads_obstacle_obs_type != cfg.quads_critic_obs)

        if cfg.quads_obs_repr in QUADS_OBS_REPR:
            self.self_obs_dim = QUADS_OBS_REPR[cfg.quads_obs_repr]
        else:
            raise NotImplementedError(f'Layer {cfg.quads_obs_repr} not supported!')

        self.neighbor_hidden_size = cfg.quads_neighbor_hidden_size
        self.use_obstacles = cfg.quads_use_obstacles

        if cfg.quads_neighbor_visible_num == -1:
            self.num_use_neighbor_obs = cfg.quads_num_agents - 1
        else:
            self.num_use_neighbor_obs = cfg.quads_neighbor_visible_num

        self.neighbor_obs_dim = QUADS_NEIGHBOR_OBS_TYPE[cfg.quads_neighbor_obs_type]

        self.all_neighbor_obs_dim = self.neighbor_obs_dim * self.num_use_neighbor_obs

        # Embedding Layer
        fc_encoder_layer = cfg.rnn_size
        self.self_embed_layer = nn.Sequential(
            fc_layer(self.self_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
        )
        self.neighbor_embed_layer = nn.Sequential(
            fc_layer(self.all_neighbor_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
        )

        if self.use_separate_critic:
            if cfg.quads_obstacle_obs_type == 'ToFs':
                if cfg.quads_obstacle_tof_resolution == 4:
                    self.obst_actor_obs_dim = QUADS_OBSTACLE_OBS_TYPE['ToFs_4']
                elif cfg.quads_obstacle_tof_resolution == 8:
                    self.obst_actor_obs_dim = QUADS_OBSTACLE_OBS_TYPE['ToFs_8']
                else:
                    raise NotImplementedError(f'Obstacle TOF resolution {cfg.quads_obstacle_tof_resolution} not supported!')
            elif cfg.quads_obstacle_obs_type == 'octomap':
                self.obst_actor_obs_dim = QUADS_OBSTACLE_OBS_TYPE['octomap']
            else:
                raise NotImplementedError(f'Obstacle observation type {cfg.quads_obstacle_obs_type} not supported!')

            if self.obst_obs_diff_ac:
                if cfg.quads_critic_obs == 'ToFs':
                    if cfg.quads_obstacle_tof_resolution == 4:
                        self.obst_critic_obs_dim = QUADS_OBSTACLE_OBS_TYPE['ToFs_4']
                    elif cfg.quads_obstacle_tof_resolution == 8:
                        self.obst_critic_obs_dim = QUADS_OBSTACLE_OBS_TYPE['ToFs_8']
                    else:
                        raise NotImplementedError(
                            f'Obstacle TOF resolution {cfg.quads_obstacle_tof_resolution} not supported!')
                elif cfg.quads_critic_obs == 'octomap':
                    self.obst_critic_obs_dim = QUADS_OBSTACLE_OBS_TYPE['octomap']
                else:
                    raise NotImplementedError(f'Obstacle observation type {cfg.quads_obstacle_obs_type} not supported!')
            else:
                self.obst_critic_obs_dim = self.obst_actor_obs_dim

            if self.is_critic:
                self.obstacle_obs_dim = self.obst_critic_obs_dim
            else:
                self.obstacle_obs_dim = self.obst_actor_obs_dim

        else:
            if cfg.quads_obstacle_obs_type == 'ToFs':
                if cfg.quads_obstacle_tof_resolution == 4:
                    self.obstacle_obs_dim = QUADS_OBSTACLE_OBS_TYPE['ToFs_4']
                elif cfg.quads_obstacle_tof_resolution == 8:
                    self.obstacle_obs_dim = QUADS_OBSTACLE_OBS_TYPE['ToFs_8']
                else:
                    raise NotImplementedError(f'Obstacle TOF resolution {cfg.quads_obstacle_tof_resolution} not supported!')
            elif cfg.quads_obstacle_obs_type == 'octomap':
                self.obstacle_obs_dim = QUADS_OBSTACLE_OBS_TYPE['octomap']
            else:
                raise NotImplementedError(f'Obstacle observation type {cfg.quads_obstacle_obs_type} not supported!')

            self.obst_critic_obs_dim = self.obst_actor_obs_dim

        self.obstacle_embed_layer = nn.Sequential(
            fc_layer(self.obstacle_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
        )

        # Attention Layer
        self.attention_layer = OneHeadAttention(cfg.rnn_size)

        # MLP Layer
        self.feed_forward = nn.Sequential(
            fc_layer(3 * cfg.rnn_size, cfg.rnn_size), nn.Tanh()
        )
        self.encoder_output_size = cfg.rnn_size

    def get_out_size(self):
        return self.encoder_output_size

    def forward(self, obs_dict):
        obs = obs_dict['obs']
        batch_size = obs.shape[0]
        obs_self = obs[:, :self.self_obs_dim]
        obs_neighbor = obs[:, self.self_obs_dim: self.self_obs_dim + self.all_neighbor_obs_dim]

        if self.use_separate_critic and self.obst_obs_diff_ac:
            if self.is_critic:
                obs_critic_obst_start = self.self_obs_dim + self.all_neighbor_obs_dim + self.obst_actor_obs_dim
                obs_obstacle = obs[:, obs_critic_obst_start:]
            else:
                obs_actor_obst_start = self.self_obs_dim + self.all_neighbor_obs_dim
                obs_actor_obst_end = obs_actor_obst_start + self.obst_actor_obs_dim
                obs_obstacle = obs[:, obs_actor_obst_start: obs_actor_obst_end]
        else:
            obs_obstacle = obs[:, self.self_obs_dim + self.all_neighbor_obs_dim:]

        # Attention
        self_embed = self.self_embed_layer(obs_self)
        neighbor_embed = self.neighbor_embed_layer(obs_neighbor)
        obstacle_embed = self.obstacle_embed_layer(obs_obstacle)

        self_embed_view = self_embed.view(batch_size, 1, -1)
        neighbor_embed = neighbor_embed.view(batch_size, 1, -1)
        obstacle_embed = obstacle_embed.view(batch_size, 1, -1)

        attn_embed = torch.cat((self_embed_view, neighbor_embed, obstacle_embed), dim=1)

        attn_embed, attn_score = self.attention_layer(attn_embed, attn_embed, attn_embed)
        attn_embed = attn_embed.view(batch_size, -1)

        # Concat
        # embeddings = torch.cat((self_embed, attn_embed), dim=1)
        out = self.feed_forward(attn_embed)

        return out
