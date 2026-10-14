"""Deep-learning network shared by the Colab notebook (AR_DL_Colab.ipynb) and the live scorer.

KEEP THIS FILE SELF-CONTAINED (only numpy and torch): the notebook embeds it verbatim, so the
network trained in Colab is rebuilt here, from its weights, with exactly the same code.

A prediction reads one window: the last ``steps`` rows of a recloser, ``step_hours`` apart, each row
holding the inputs of ml_features.build_features after scale_inputs.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NET_VERSION = 1          # change when ARNet changes so older exported weights are refused, not misread
ARCHS = ("gru", "tcn", "transformer")


class TCNBlock(nn.Module):
    """Two causal, dilated 1-D convolutions with a shortcut."""

    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.pad = 2 * dilation
        self.conv1 = nn.Conv1d(width, width, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, dilation=dilation)
        self.norm = nn.BatchNorm1d(width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                       # batch x width x steps
        y = self.drop(F.gelu(self.conv1(F.pad(x, (self.pad, 0)))))
        y = self.drop(self.conv2(F.pad(y, (self.pad, 0))))
        return F.gelu(self.norm(x + y))


class ARNet(nn.Module):
    """Sequence encoder (GRU, TCN or Transformer) + attention pooling + three heads:
    failure within 30 days, failure within 7 days (logits) and which part fails (logits per part)."""

    def __init__(self, arch, n_inputs, n_components, width=64, steps=168, dropout=0.2):
        super().__init__()
        self.arch = arch
        if arch == "gru":
            self.inp = nn.Conv1d(n_inputs, width, 5, padding=2)
            self.gru = nn.GRU(width, width, num_layers=2, batch_first=True, dropout=dropout)
        elif arch == "tcn":
            self.inp = nn.Conv1d(n_inputs, width, 1)
            self.blocks = nn.ModuleList(TCNBlock(width, 2 ** k, dropout) for k in range(7))  # reaches 509 steps back
        elif arch == "transformer":
            self.inp = nn.Linear(n_inputs, width)
            self.pos = nn.Parameter(0.02 * torch.randn(1, steps, width))
            layer = nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=2 * width, dropout=dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        else:
            raise ValueError(f"unknown network {arch!r}: use one of {', '.join(ARCHS)}")
        self.attn = nn.Linear(width, 1)
        self.mix = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, width), nn.GELU(), nn.Dropout(dropout))
        self.head30 = nn.Linear(width, 1)
        self.head7 = nn.Linear(width, 1)
        self.head_part = nn.Linear(width, max(1, n_components))

    def encode(self, x):                                        # batch x steps x inputs -> batch x steps x width
        if self.arch == "gru":
            return self.gru(F.gelu(self.inp(x.transpose(1, 2))).transpose(1, 2))[0]
        if self.arch == "tcn":
            h = self.inp(x.transpose(1, 2))
            for block in self.blocks:
                h = block(h)
            return h.transpose(1, 2)
        return self.enc(self.inp(x) + self.pos[:, -x.shape[1]:])

    def forward(self, x):
        h = self.encode(x)
        w = torch.softmax(self.attn(h).squeeze(-1).float(), dim=1)          # how much each step matters
        pooled = (w.unsqueeze(-1) * h.float()).sum(dim=1).to(h.dtype)
        z = self.mix(torch.cat([h[:, -1], pooled], dim=-1))                 # latest step + weighted summary
        return self.head30(z).squeeze(-1), self.head7(z).squeeze(-1), self.head_part(z)


def scale_inputs(raw, log_idx, mean, std, clip):
    """Raw inputs (rows x features, from build_features) -> what the network reads.

    Long-tailed counts (columns log_idx) are log-compressed, then every input is standardised with the
    fit-month mean/std. Missing values become 0 (the average); values are limited to +-clip.
    """
    v = np.array(raw, dtype=np.float32)
    v[:, log_idx] = np.log1p(np.clip(v[:, log_idx], 0, None))
    v -= np.asarray(mean, dtype=np.float32)
    v /= np.asarray(std, dtype=np.float32)
    np.nan_to_num(v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(v, -clip, clip, out=v)


def window_rows(end_rows, first_rows, steps, stride):
    """Row numbers of each window, oldest first: ``steps`` rows ``stride`` apart that end at ``end_rows``.

    Rows are one recloser's regular time series; positions before its first row (``first_rows``) repeat
    that first row, as for a recloser with less history than the window.
    """
    offsets = np.arange(-(steps - 1) * stride, 1, stride)
    return np.maximum(np.asarray(end_rows)[:, None] + offsets[None, :], np.asarray(first_rows)[:, None])
