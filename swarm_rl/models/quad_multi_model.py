from sample_factory.algo.utils.context import global_model_factory
from sample_factory.model.encoder import Encoder

from swarm_rl.models.encoders.QuadMultiEncoder import QuadMultiEncoder
from swarm_rl.models.encoders.QuadMultiHeadAttentionEncoder import QuadMultiHeadAttentionEncoder
from swarm_rl.models.encoders.QuadSingleHeadAttentionEncoder_Sim2Real import QuadSingleHeadAttentionEncoder_Sim2Real


def make_quadmulti_encoder(cfg, obs_space) -> Encoder:
    if cfg.quads_encoder_type == "attention":
        if cfg.is_critic:
            model = QuadMultiHeadAttentionEncoder(cfg=cfg, obs_space=obs_space, is_critic=True)
        else:
            if cfg.quads_sim2real:
                model = QuadSingleHeadAttentionEncoder_Sim2Real(cfg=cfg, obs_space=obs_space, is_critic=False)
            else:
                model = QuadMultiHeadAttentionEncoder(cfg=cfg, obs_space=obs_space, is_critic=False)
    else:
        model = QuadMultiEncoder(cfg, obs_space)
    return model

def register_models():
    global_model_factory().register_encoder_factory(make_quadmulti_encoder)
