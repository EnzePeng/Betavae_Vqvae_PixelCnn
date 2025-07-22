import torch
from torch.utils.data import DataLoader, random_split
from dataset import DefectDataset
from utils import EarlyStopping, plot_loss
from tqdm import tqdm
import os
from torch.utils.data import Subset

from beta_vae import BetaVAE
from vq_vae import VQVAE

def train(img_dir, batch_size, epochs, lr, patience, save_path, img_size,
          subset_prop, val_prop, model):

    # 1. Load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load dataset
    dataset = DefectDataset(img_dir, size=img_size)

    # 3. Use subset data
    if subset_prop != 1:
        total_len = len(dataset)
        indices = torch.randperm(total_len)[:int(total_len * subset_prop)]
        dataset = Subset(dataset, indices)

    # 4. Split validation set
    train_prop = 1 - val_prop
    train_len = int(len(dataset) * train_prop)
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    # 5. Define dataloader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # 6. Define model and optimizer
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 7. Init early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # 8. Record losses
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # 9. Start training
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit='batch')
        for imgs in train_bar:
            imgs = imgs.to(device)
            optimizer.zero_grad()

            output = model(imgs)
            if isinstance(output, list):
                if len(output) == 4:
                    # BetaVAE: recons, input, mu, log_var
                    recons, input_data, mu, log_var = output
                    loss_dict = model.loss_function(recons, input_data, mu, log_var,
                                                    M_N=imgs.size(0)/len(train_loader.dataset))
                elif len(output) == 3:
                    # VQVAE: recons, input, vq_loss
                    recons, input_data, vq_loss = output
                    loss_dict = model.loss_function(recons, input_data, vq_loss)
                else:
                    raise ValueError("Unknown model output format.")
            else:
                raise ValueError("Model forward should return a tuple.")

            # recons, input, mu, log_var = model(imgs)
            # loss_dict = model.loss_function(recons, input, mu, log_var, M_N=imgs.size(0)/len(train_loader.dataset))
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * imgs.size(0)
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", unit='batch')
        with torch.no_grad():
            for imgs in val_bar:
                imgs = imgs.to(device)

                output = model(imgs)
                if isinstance(output, list):
                    if len(output) == 4:
                        recons, input_data, mu, log_var = output
                        loss_dict = model.loss_function(recons, input_data, mu, log_var,
                                                        M_N=imgs.size(0)/len(val_loader.dataset))
                    elif len(output) == 3:
                        recons, input_data, vq_loss = output
                        loss_dict = model.loss_function(recons, input_data, vq_loss)
                    else:
                        raise ValueError("Unknown model output format.")
                else:
                    raise ValueError("Model forward should return a tuple.")

                # recons, input, mu, log_var = model(imgs)
                # loss_dict = model.loss_function(recons, input, mu, log_var, M_N=imgs.size(0)/len(val_loader.dataset))

                running_val_loss += loss_dict['loss'].item() * imgs.size(0)
                val_bar.set_postfix(val_loss=loss_dict['loss'].item())

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # torch.save(model.state_dict(), save_path)

        # early stopping
        early_stopping(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch+1} with val loss {best_val_loss:.4f}")

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        # plot losses every epoch
        plot_loss(train_losses, val_losses, save_path='logs/loss_curve.png')

if __name__ == '__main__':
    #=================  CONFIG ====================
    img_dir = './img_align_celeba'
    batch_size = 128
    epochs = 20
    lr = 1e-3
    patience = 3

    os.makedirs('logs', exist_ok=True)

    img_size = 64
    subset_prop = 1
    val_prop = 0.1

    # =================  MODEL AND TRAIN ====================
    latent_dim = 256  # 128
    betaVAE_model = BetaVAE(in_channels=3, latent_dim=latent_dim, gamma=10, max_capacity=25, Capacity_max_iter=10000, loss_type="B")
    save_path = 'logs/beta_VAE.pth'

    # train(img_dir, batch_size, epochs, lr, patience, save_path, img_size, subset_prop,
    #       val_prop, betaVAE_model)

    vqVAE_model = VQVAE(
        in_channels=3,
        embedding_dim=64,
        num_embeddings=512,
        img_size=64,
        beta=0.25
    )

    save_path = 'logs/vq_VAE.pth'

    train(img_dir, batch_size, epochs, lr, patience, save_path, img_size, subset_prop,
          val_prop, vqVAE_model)