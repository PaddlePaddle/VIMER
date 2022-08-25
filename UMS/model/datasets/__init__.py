"""
add datasets
"""
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .product1m_dataset import Product1MDataset
from .datasets import give_dataloaders
from .ctc_dataset import CTCDataset

__all__ = ["Product1MDataset", "give_dataloaders", "CTCDataset"]
