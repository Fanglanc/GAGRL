import torch
import torch.nn as nn
from transit.models.state_encoder import GAGStateEncoder, RLStateEncoder
from transit.models.policy import TransitPolicy
from transit.models.value import TransitValue


def create_GAGRL_model(cfg, agent):
    """Create policy and value network from a config file.
    Args:
        cfg: A config object.
        agent: An agent.
    Returns:
        A tuple containing the policy network and the value network.
    """
    shared_net = GAGStateEncoder(cfg.state_encoder_specs, agent)
    policy_net = TransitPolicy(cfg.policy_specs, agent, shared_net)
    value_net = TransitValue(cfg.value_specs, agent, shared_net)
    return policy_net, value_net


def create_RL_model(cfg, agent):
    """Create policy and value network from a config file.
    Args:
        cfg: A config object.
        agent: An agent.
    Returns:
        A tuple containing the policy network and the value network.
    """
    shared_net = RLStateEncoder(cfg.state_encoder_specs, agent)
    policy_net = TransitPolicy(cfg.policy_specs, agent, shared_net)
    value_net = TransitValue(cfg.value_specs, agent, shared_net)
    return policy_net, value_net


class ActorCritic(nn.Module):
    """
    An Actor-Critic network for parsing parameters.

    Args:
        actor_net (nn.Module): actor network.
        value_net (nn.Module): value network.
    """
    def __init__(self, actor_net, value_net):
        super().__init__()
        self.actor_net = actor_net
        self.value_net = value_net
