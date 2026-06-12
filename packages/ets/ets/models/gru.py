"""GRU sequence model."""

from __future__ import annotations

import torch.nn as nn

from ets.models.base import BaseSequenceModel


class GRUModel(BaseSequenceModel):
    """GRU-based sequence model."""

    def _build_rnn(self) -> nn.Module:
        return nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
