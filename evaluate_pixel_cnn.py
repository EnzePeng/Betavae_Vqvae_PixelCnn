import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm

from vq_vae import VQVAE
from dataset import DefectDataset

# ------------------ MaskedConv2d ------------------
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', torch.ones_like(self.weight))
        _, _, kH, kW = self.weight.size()
        yc, xc = kH // 2, kW // 2
        self.mask[:, :, yc, xc + (mask_type == 'B'):] = 0
        self.mask[:, :, yc + 1:] = 0

    def forward(self, x):
        weight = self.weight * self.mask
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# ------------------ PixelCNN Model ------------------
class PixelCNNPrior(nn.Module):
    def __init__(self, num_embeddings, hidden_dim=64, num_layers=9):
        super().__init__()
        fm = hidden_dim
        layers = [
            MaskedConv2d('A', 64, fm, kernel_size=7, padding=3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        ]
        for _ in range(num_layers - 2):
            layers += [
                MaskedConv2d('B', fm, fm, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm2d(fm),
                nn.ReLU(True)
            ]
        layers.append(nn.Conv2d(fm, num_embeddings, kernel_size=1))
        self.net = nn.Sequential(*layers)
        self.embedding = nn.Embedding(num_embeddings, 64)  # embed to 1 channel to match input channel=1

    def forward(self, x):
        # x: [B,H,W], indices
        x = self.embedding(x)       # [B,H,W,1]
        x = x.permute(0, 3, 1, 2)   # [B,1,H,W]
        return self.net(x)          # [B,num_embeddings,H,W]

# ------------------ Sample ------------------
@torch.no_grad()
def sample_prior(prior_model, vqvae_model, img_size=64, num_embeddings=512,
                 device=None, num_samples=64, save_path='logs/prior_samples.png',
                 temperature=1.0, use_argmax=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prior_model.to(device).eval()
    vqvae_model.to(device).eval()

    H = W = img_size // 4
    samples = torch.zeros((num_samples, H, W), dtype=torch.long, device=device)

    top_k = 20
    for i in range(H):
        for j in range(W):
            logits = prior_model(samples)  # [B, num_embeddings, H, W]
            logits_ij = logits[:, :, i, j] / temperature  # [B, num_embeddings]

            if use_argmax:
                chosen = torch.argmax(logits_ij, dim=-1)  # [B]
                samples[:, i, j] = chosen
            else:
                probs = F.softmax(logits_ij, dim=-1)  # [B, num_embeddings]

                if top_k is not None and top_k < probs.size(-1):
                    # ---- Top-k  ----
                    topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=-1)  # [B, top_k]
                    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
                    choice_in_topk = torch.multinomial(topk_probs, num_samples=1).squeeze(-1)  # [B]
                    chosen = topk_idx.gather(-1, choice_in_topk.unsqueeze(-1)).squeeze(-1)
                    samples[:, i, j] = chosen
                else:
                    chosen = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
                    samples[:, i, j] = chosen


    embedding = vqvae_model.vq_layer.embedding.weight
    quantized = embedding[samples.view(-1)].view(num_samples, H, W, -1)
    quantized = quantized.permute(0, 3, 1, 2).contiguous()
    reconstructions = vqvae_model.decode(quantized)
    reconstructions = (reconstructions+1)/2
    reconstructions = reconstructions.clamp(0, 1)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    vutils.save_image(reconstructions.cpu(), save_path, nrow=int(num_samples ** 0.5))
    print(f"Sample saved to {save_path}")

# ------------------ Main ------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vqvae_model = VQVAE(in_channels=3, embedding_dim=64, num_embeddings=512, img_size=64, beta=0.25)
    vqvae_model.load_state_dict(torch.load('logs/vq_VAE.pth', map_location=device))

    dataset = DefectDataset('./img_align_celeba', size=64)

    prior_model = PixelCNNPrior(num_embeddings=512, hidden_dim=64, num_layers=9)

    prior_model.load_state_dict(torch.load('logs/prior_model.pth', map_location=device))

    sample_prior(prior_model, vqvae_model,
                 img_size=64, num_embeddings=512,
                 device=device, num_samples=36,
                 save_path='logs/prior_samples.png',
                 temperature=1, use_argmax=False)
