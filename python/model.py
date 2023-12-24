import torch, numpy as np
from torch import nn, autocast
from torch.distributions import Categorical

import tetris
from ev_var import kEvMatrix, kDevMatrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'
kBoardShape, kMetaShape, kMovesShape, kMoveMetaShape, _ = tetris.Tetris.StateShapes()
kH, kW = kBoardShape[1:]


class ConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.main = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding = 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(True),
                nn.Conv2d(ch, ch, 3, padding = 1),
                nn.BatchNorm2d(ch),
                )
        self.final = nn.ReLU(True)
    def forward(self, x):
        return self.final(self.main(x) + x)


class InitialEmbed(nn.Module):
    def __init__(self, feats, meta_feats, channels):
        super().__init__()
        self.embed_1 = nn.Conv2d(feats, channels, 5, padding=2)
        self.embed_2 = nn.Conv2d(feats, channels, (kH, 1))
        self.embed_3 = nn.Conv2d(feats, channels, (1, kW))
        self.meta = nn.Linear(meta_feats, channels)
        self.finish = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

    @autocast(device_type=device)
    def forward(self, obs, meta):
        x_meta = self.meta(meta)
        x = self.embed_1(obs) + self.embed_2(obs) + self.embed_3(obs) + x_meta.view(*x_meta.shape, 1, 1)
        return self.finish(x)


class PiValueHead(nn.Module):
    def __init__(self, in_feat):
        super().__init__()
        self.linear = nn.Linear(in_feat, 1)

    @autocast(device_type=device, enabled=False)
    def forward(self, pi, value, invalid):
        pi = pi.float()
        pi[invalid] = -float('inf')
        v = self.linear(value.float())
        return pi, v.transpose(0, 1)

class Model(nn.Module):
    def __init__(self, start_blocks, end_blocks, channels):
        super().__init__()
        self.board_embed = InitialEmbed(kBoardShape[0], kMetaShape[0], channels)
        self.moves_embed = InitialEmbed(kMovesShape[0], kMoveMetaShape[0], channels)
        self.main_start = nn.Sequential(*[ConvBlock(channels) for i in range(start_blocks)])
        self.main_end = nn.Sequential(*[ConvBlock(channels) for i in range(end_blocks)])
        self.pi_logits_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1),
            nn.BatchNorm2d(8),
            nn.Flatten(),
            nn.ReLU(True),
            nn.Linear(8 * kH * kW, 4 * kH * kW)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Flatten(),
            nn.ReLU(True),
            nn.Linear(1 * kH * kW, 256),
            nn.ReLU(True),
        )
        self.pi_value_final = PiValueHead(256)

    @autocast(device_type=device)
    def forward(self, obs, categorical=False):
        board, board_meta, moves, moves_meta, meta_int = obs
        batch = board.shape[0]
        pi = None
        v = torch.zeros((1, batch), dtype=torch.float32, device=board.device)
        evdev = torch.zeros((2, batch), dtype=torch.float32, device=board.device)

        invalid = moves[:,2:6].view(batch, -1) == 0
        x = self.board_embed(board, board_meta)
        x = self.main_start(x)
        x = x + self.moves_embed(moves, moves_meta)
        x = self.main_end(x)
        pi, v = self.pi_value_final(
                self.pi_logits_head(x),
                self.value_head(x),
                invalid)
        if categorical: pi = Categorical(logits=pi)
        return pi, torch.concat([v, evdev])

def obs_to_torch(obs, device=None):
    if device is None:
        # local
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if len(obs[0].shape) == 3:
        return [torch.tensor(i, device=device).unsqueeze(0) for i in obs]
    else:
        return [torch.tensor(i, device=device) for i in obs]
