# -*- coding:utf-8 -*-
# @FileName  :embedding.py
# @Time      :2023/4/28 21:00
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import torch


class StreamSinusoidalPositionEncoder(torch.nn.Module):
    '''

    '''

    def __int__(self):
        pass

    def encode(self, positions: torch.Tensor = None, depth: int = None, dtype: torch.dtype = torch.float32):
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        log_timescale_increment = torch.log(torch.tensor([10000], dtype=dtype)) / (depth / 2 - 1)
        inv_timescales = torch.exp(torch.arange(depth / 2).type(dtype) * (-log_timescale_increment))
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(inv_timescales, [1, 1, -1])
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return encoding.type(dtype)

    def forward(self, x, start_idx=0, cache=None):
        batch_size, timesteps, input_dim = x.size()
        start_idx = 0
        if cache is not None:
            start_idx = cache["start_idx"]
            cache["start_idx"] += timesteps
        positions = torch.arange(1, timesteps + start_idx + 1)[None, :]
        position_encoding = self.encode(positions, input_dim, x.dtype).to(x.device)
        return x + position_encoding[:, start_idx: start_idx + timesteps]
