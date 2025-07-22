import os
import torch
import torchvision.utils as vutils
from dataset import DefectDataset
from beta_vae import BetaVAE
from vq_vae import VQVAE

def evaluate(model, model_path, img_dir='./img_align_celeba', batch_size=32, device=None,
             save_dir='logs/reconstructions', name="", num_samples=64):
    """
    Evaluate a given model: reconstruct input images and generate random samples.

    Args:
        model: a PyTorch model instance with methods `generate` and `sample`.
        model_path: path to the saved model weights (.pth).
        img_dir: directory of evaluation images.
        batch_size: batch size for loading images.
        device: torch device, if None automatically use CUDA if available.
        save_dir: directory to save reconstruction and sample images.
        num_samples: number of random samples to generate.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and dataloader
    dataset = DefectDataset(img_dir, size=64)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    # Reconstruction
    with torch.no_grad():
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device)
            # Assume model.generate takes input imgs and outputs reconstruction
            recons = model.generate(imgs)

            # Clamp images between 0 and 1 for visualization
            # imgs_show = imgs.clamp(0, 1)
            # recons_show = recons.clamp(0, 1)

            # imgs_show = (imgs + 1) / 2
            # recons_show = (recons + 1) / 2
            # imgs_show = imgs_show.clamp(0, 1)
            # recons_show = recons_show.clamp(0, 1)

            min_val = imgs.min()
            max_val = imgs.max()
            imgs_show = (imgs - min_val) / (max_val - min_val + 1e-8)

            min_val_r = recons.min()
            max_val_r = recons.max()
            recons_show = (recons - min_val_r) / (max_val_r - min_val_r + 1e-8)

            imgs_show = imgs_show.clamp(0, 1)
            recons_show = recons_show.clamp(0, 1)

            # Concatenate original and reconstructed images for comparison
            comparison = torch.cat([imgs_show.cpu(), recons_show.cpu()])
            vutils.save_image(comparison, os.path.join(save_dir, name+f'_reconstruction_{i}.png'), nrow=imgs.size(0))
            break  # save only one batch reconstruction for quick check
    print(f"Reconstruction images saved at {save_dir}")

    # Random sampling
    with torch.no_grad():
        samples = model.sample(num_samples=num_samples, current_device=device)
        samples = (samples +1) /2
        samples_show = samples.clamp(0, 1)
        vutils.save_image(samples_show, os.path.join(save_dir, name+'_random_samples.png'), nrow=int(num_samples ** 0.5))
    print(f"Randomly generated samples saved at {save_dir}")


if __name__ == '__main__':
    beta_vae_model = BetaVAE(in_channels=3, latent_dim=256)
    evaluate(beta_vae_model, model_path='logs/beta_VAE.pth',name="beta")

    vq_vae_model = VQVAE(in_channels=3, embedding_dim=64, num_embeddings=512, img_size=64, beta=0.25)
    evaluate(vq_vae_model, model_path='logs/vq_VAE.pth',name="vq")
