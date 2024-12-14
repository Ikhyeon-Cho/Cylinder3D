"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: base_block.py
Date: 2024/12/16 10:00

Base block for all network blocks
"""

import torch.nn as nn
import spconv


class BaseBlock(nn.Module):
    """Base class for all blocks with common functionality"""

    def __init__(self):
        super().__init__()
        self.weight_initialization()

    def weight_initialization(self):
        """Initialize weights for all layer types"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                # BatchNorm initialization
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, spconv.SubMConv3d):
                # Sparse convolution initialization
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, spconv.SparseInverseConv3d):
                # Transpose convolution initialization
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                # Linear layer initialization
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
