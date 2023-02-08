import math

import numpy as np
import torch
from torch import nn, Tensor
from torch.distributions import Categorical


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, positions: Tensor) -> Tensor:
        return self.pe[positions]


class Actor(nn.Module):

    def __init__(self, pos_encoder):
        super(Actor, self).__init__()
        self.activation = nn.Tanh()
        self.project = nn.Linear(4, 8)
        nn.init.xavier_uniform_(self.project.weight, gain=1.0)
        nn.init.constant_(self.project.bias, 0)
        self.pos_encoder = pos_encoder

        self.embedding_fixed = nn.Embedding(2, 1)
        self.embedding_legal_op = nn.Embedding(2, 1)

        self.tokens_start_end = nn.Embedding(3, 4)

        # self.conv_transform = nn.Conv1d(5, 1, 1)
        # nn.init.kaiming_normal_(self.conv_transform.weight, mode="fan_out", nonlinearity="relu")
        # nn.init.constant_(self.conv_transform.bias, 0)

        self.enc1 = nn.TransformerEncoderLayer(8, 1, dim_feedforward=8 * 4, dropout=0.0, batch_first=True,
                                               norm_first=True)
        self.enc2 = nn.TransformerEncoderLayer(8, 1, dim_feedforward=8 * 4, dropout=0.0, batch_first=True,
                                               norm_first=True)

        self.final_tmp = nn.Sequential(
            layer_init_tanh(nn.Linear(8, 32)),
            nn.Tanh(),
            layer_init_tanh(nn.Linear(32, 1), std=0.01)
        )
        self.no_op = nn.Sequential(
            layer_init_tanh(nn.Linear(8, 32)),
            nn.Tanh(),
            layer_init_tanh(nn.Linear(32, 1), std=0.01)
        )

    def forward(self, obs, attention_interval_mask, job_resource, mask, indexes_inter, tokens_start_end):
        embedded_obs = torch.cat((self.embedding_fixed(obs[:, :, :, 0].long()), obs[:, :, :, 1:3],
                                  self.embedding_legal_op(obs[:, :, :, 3].long())), dim=3)
        non_zero_tokens = tokens_start_end != 0
        t = tokens_start_end[non_zero_tokens].long()
        embedded_obs[non_zero_tokens] = self.tokens_start_end(t)
        pos_encoder = self.pos_encoder(indexes_inter.long())
        pos_encoder[non_zero_tokens] = 0
        obs = self.project(embedded_obs) + pos_encoder

        transformed_obs = obs.view(-1, obs.shape[2], obs.shape[3])
        attention_interval_mask = attention_interval_mask.view(-1, attention_interval_mask.shape[-1])
        transformed_obs = self.enc1(transformed_obs, src_key_padding_mask=attention_interval_mask == 1)
        transformed_obs = transformed_obs.view(obs.shape)
        obs = transformed_obs.mean(dim=2)

        job_resource = job_resource[:, :-1, :-1] == 0

        obs_action = self.enc2(obs, src_mask=job_resource) + obs

        logits = torch.cat((self.final_tmp(obs_action).squeeze(2), self.no_op(obs_action).mean(dim=1)), dim=1)
        return logits.masked_fill(mask == 0, -3.4028234663852886e+38)


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.pos_encoder = PositionalEncoding(8)
        self.actor = Actor(self.pos_encoder)

    def forward(self, data, attention_interval_mask, job_resource_masks, mask, indexes_inter, tokens_start_end,
                action=None):
        logits = self.actor(data, attention_interval_mask, job_resource_masks, mask, indexes_inter, tokens_start_end)
        probs = Categorical(logits=logits)
        if action is None:
            probabilities = probs.probs
            actions = torch.multinomial(probabilities, probabilities.shape[1])
            return actions, torch.log(probabilities), probs.entropy()
        else:
            return logits, probs.log_prob(action), probs.entropy()

    def get_action_only(self, data, attention_interval_mask, job_resource_masks, mask, indexes_inter, tokens_start_end):
        logits = self.actor(data, attention_interval_mask, job_resource_masks, mask, indexes_inter, tokens_start_end)
        probs = Categorical(logits=logits)
        return probs.sample()

    def get_logits_only(self,data, attention_interval_mask, job_resource_masks, mask, indexes_inter, tokens_start_end):
        logits = self.actor(data, attention_interval_mask, job_resource_masks, mask, indexes_inter, tokens_start_end)
        return logits


def layer_init_tanh(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer
