# https://github.com/rishikksh20/MLP-Mixer-pytorch/blob/master/mlp-mixer.py
# X \in B x 3 x H x W 
# patch -> X \in B x 3 x S x P x P
# transpose -> X \in B x S x 3 x P x P
# flatten -> X \in B x S x 3P^2
# project -> X \in B x S x C

# X \in B x 11 x 220
# patch -> X \in B x 11 x S x P
# transpose -> X \in B x S x 11 x P
# flatten -> X \in B x S x 11P
# project -> X \in B x S x C

import torch
import numpy as np
from torch import nn
from lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Create patching class that includes projection layer
# Input (batch x channels x timesteps)
# Output (batch x num_patches x hidden_dim)
class PatchingClass(nn.Module):
    def __init__(self, patch_size=20, hidden_dim=100, padded_length=220, patch_class="sequential1d", channels=11): # Hard-coded from phoneme data
        super().__init__()
        assert padded_length % patch_size == 0, "Padded length must be evenly divisible by patch size" 
        num_patches = int(padded_length/patch_size)

        # X \in B x S x 11P -> X \in B x S x C
        self.projection = nn.Linear(patch_size*channels, hidden_dim)
        self.patch_class = patch_class 
        self.array_length = padded_length
        self.patch_size = patch_size
        # X \in B x 11 x 220 ->  X \in B x 11 x S x P
        # X \in B x S x 11 x P ->  X \in B x S x 11P
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def cyclically_patch(self, x):
        def cyclic_sort(tensor, patch_size):
            # Reshape the original tensor into a matrix with patch_size columns
            reshaped_tensor = tensor.reshape(-1, patch_size)
            
            # Transpose the matrix and flatten it to get the cyclically sorted tensor
            sorted_tensor = reshaped_tensor.t().flatten()
            
            return sorted_tensor

        batch_size, num_channels, length = x.shape
        x_patched = torch.empty_like(x)
        for batch_idx in range(batch_size):
            for channel_idx in range(num_channels):
                channel_data = x[batch_idx, channel_idx, :]
                sorted_channel = cyclic_sort(channel_data, self.patch_size)
                x_patched[batch_idx, channel_idx, :] = sorted_channel.clone().detach().reshape(1, 1, -1)

        return x_patched
            
    def forward(self, x):
        if self.patch_class == 'random1d':
            idx = torch.randperm(x.shape[2])
            x = x[:, :, idx]

        elif self.patch_class == 'cyclical1d':
            x = self.cyclically_patch(x) 

        elif self.patch_class == 'sequential1d': pass
        else: raise ValueError()
            
        # X \in B x 11 x 220
        x = x.unfold(2, self.patch_size, self.patch_size).unsqueeze(-2).squeeze(dim=3)
        # X \in B x 11 x S x P
        # X \in B x S x P x 11
        x = torch.transpose(x, 1, 2)
        # X \in B x S x 11 x P
        x = self.flatten(x)
        # X \in B x S x 11P
        x = self.projection(x.float())
        # X \in B x S x C
        return x

class TokenMixingBlock(nn.Module):
    # applied to transposed X, maps R^s -> R^s, shared across all columns
    # 2 fully-connected layers and a GeLU layer
    # X \in B x S x C
    def __init__(self, num_patch, mlp_dimension):
        super().__init__()
        self.D_s = mlp_dimension
        self.S = num_patch
        self.dense_1 = nn.Linear(self.S, self.D_s)
        self.dense_2 = nn.Linear(self.D_s, self.S)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x_prime = self.gelu(self.dense_1(x))
        return self.dense_2(x_prime)

class ChannelMixingBlock(nn.Module):
    # applied to X, maps R^C -> R^C, shared across all rows
    # 2 fully-connected layers and a GeLU layer
    # X \in B x S x C
    def __init__(self, channel_dimension, mlp_dimension):
        super().__init__()
        self.D_c = mlp_dimension
        self.C = channel_dimension
        self.dense_1 = nn.Linear(self.C, self.D_c)
        self.dense_2 = nn.Linear(self.D_c, self.C)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        # in paper -> column, row indexing
        x_prime = self.gelu(self.dense_1(x))
        return self.dense_2(x_prime)

# Mixer consists of multiple layers of identical size, and each layer consists of two MLP blocks
class MixerLayer(nn.Module):
    # X \in B x S x C
    def __init__(self, channel_dim, num_patch, mlp_dimension_C, mlp_dimension_S, dropout = 0.):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm([channel_dim])
        self.layer_norm_2 = nn.LayerNorm([channel_dim])
        
        self.token_mixing = TokenMixingBlock(num_patch, mlp_dimension_S)
        self.channel_mixing = ChannelMixingBlock(channel_dim, mlp_dimension_C)

    def forward(self, x):
        normed_x = self.layer_norm_1(x)
        transposed_x = torch.transpose(normed_x, 1, 2)
        token_mixed_x = self.token_mixing(transposed_x)
        back_transpose_x = torch.transpose(token_mixed_x, 1, 2)
        x = x + back_transpose_x

        normed_x = self.layer_norm_2(x)
        channel_mixed_x = self.channel_mixing(normed_x)
        x = x + channel_mixed_x
        return x

class MLPMixer(LightningModule):
    def __init__(self, 
                 num_classes, 
                 num_blocks, 
                 patch_size, 
                 hidden_dim, 
                 patch_class, 
                 tokens_mlp_dim, 
                 channels_mlp_dim, 
                 padded_length=220, 
                 optimizer='adam', 
                 scheduler='reducelronplateau', 
                 lr=1e-3, 
                 momentum=0.9,
                 p_dropout=0.5):
        super().__init__()

        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.momentum = momentum
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.num_patch = int(padded_length/patch_size)

        # Create patching layer  
        self.patching = PatchingClass(patch_size, hidden_dim, padded_length, patch_class)

        self.mixer_blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.mixer_blocks.append(MixerLayer(hidden_dim, self.num_patch, tokens_mlp_dim, channels_mlp_dim))

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        # create S non-overlapping image patches X \in SxC 
        # and project each patch into hidden dimension C
        x = self.patching(x)

        # takes in X \in SxC and outputs X' \in SxC
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
            x = self.dropout(x)

        x = self.layer_norm(x)

        # takes in X \in SxC and outputs X \in C
        x = torch.mean(x, dim=1)

        # takes in X \in C and outputs X \in num_classes
        return self.mlp_head(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)

        # Log the loss
        loss = self.loss(preds, y) # No need for softmax, as it is included in nn.CrossEntropyLoss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log the accuracy
        _, predicted = torch.max(preds, 1)
        correct = (predicted == y).sum().item()
        total = y.size(0)
        acc = correct / total
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # collapse flag 
        collapse_flg = torch.unique(preds).size(dim=0)
        self.log("collapse_flg_train", collapse_flg, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)

        # Log the loss
        loss = self.loss(preds, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log the accuracy
        _, predicted = torch.max(preds, 1)
        correct = (predicted == y).sum().item()
        total = y.size(0)
        acc = correct / total
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # collapse flag 
        collapse_flg = torch.unique(preds).size(dim=0)
        self.log("collapse_flg_val", collapse_flg, sync_dist=True, on_epoch=True, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)

        # Log the loss
        loss = self.loss(preds, y)
        self.log("test_loss", loss)

        # Log the accuracy
        _, predicted = torch.max(preds, 1)
        correct = (predicted == y).sum().item()
        total = y.size(0)
        acc = correct / total
        self.log("test_acc", acc)

        return loss
    
    def configure_optimizers(self):
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.scheduler == "reducelronplateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10, min_lr=1e-6, cooldown=10)
        
        return [optimizer], [{"scheduler": scheduler, "monitor": "train_loss"}]




