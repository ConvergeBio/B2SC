import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path
from termcolor import colored

def zinb_loss(y_true, y_pred, prob_extra_zero_batch, over_disp_batch, eps=1e-10):
    y_true = y_true.float()
    y_pred = y_pred.float()
    
    # Ensure prob_extra_zero_batch and over_disp_batch are correctly shaped for broadcasting
    # These should be (batch_size, 1) or (batch_size,) to work with (batch_size, num_features)
    if prob_extra_zero_batch.ndim == 1:
        prob_extra_zero_batch = prob_extra_zero_batch.unsqueeze(1)
    if over_disp_batch.ndim == 1:
        over_disp_batch = over_disp_batch.unsqueeze(1)

    # Negative binomial part
    # pi in NB is the success probability, which is (1 - prob_extra_zero_batch) in ZINB context if interpreting prob_extra_zero as dropout prob
    # Or, if y_pred is mean, then pi = r / (r + mu) where mu is y_pred
    # Let's stick to the original formulation where y_pred is mu and pi is a separate parameter from the model for NB part.
    # The 'pi' in the original zinb_loss seems to be 'theta' or 'p' parameter of NB, not the zero-inflation probability.
    # Let's assume y_pred is the mean (mu) of the NB part.
    # Then, the probability parameter of NB, let's call it nb_p, is r / (r + mu).
    # And (1 - nb_p) is mu / (r + mu).
    
    # The original code had: r * torch.log(pi + eps) + y_true * torch.log(1.0 - (pi + eps))
    # This implies 'pi' was the success probability 'p' of the NB distribution.
    # If y_pred is the mean (mu), then p = r / (r + mu).
    # So, r * log(r / (r + y_pred)) + y_true * log(y_pred / (r + y_pred))
    
    mu = y_pred # mu is the mean of the NB component
    r = over_disp_batch # r is the dispersion parameter

    # Log likelihood of NB: y_true * log(mu / (mu + r)) + r * log(r / (mu + r))
    # Simplified by taking log( (mu/ (mu+r))^y_true * (r/(mu+r))^r )
    # = y_true * (log(mu) - log(mu+r)) + r * (log(r) - log(mu+r))
    # = y_true * log(mu) - y_true * log(mu+r) + r * log(r) - r * log(mu+r)
    # = y_true * log(mu) + r * log(r) - (y_true+r) * log(mu+r)

    # Using the parameterization from Wikipedia (p = r / (r+mu))
    # log_nb_part = torch.lgamma(y_true + r) - torch.lgamma(y_true + 1) - torch.lgamma(r) + \
    #               r * torch.log(r / (r + mu + eps)) + y_true * torch.log(mu / (r + mu + eps) + eps)
    
    # The original code used `pi` as a direct input, let's clarify its role.
    # If `pi` from `zinb_loss(y_true, y_pred, pi, r)` was `p` of NB (success prob),
    # and `y_pred` was `mu` (mean), then it's a bit mixed.
    # If `y_pred` is the predicted rate (lambda) of Poisson, which becomes `mu` for NB,
    # and `pi` in `zinb_loss` is the NB probability `p = mu / (mu + r)` or `r / (mu + r)`.
    # The original had: r * torch.log(pi + eps) + y_true * torch.log(1.0 - (pi + eps))
    # This means `pi` was `p_success`. So `p_success` is `r / (mu + r)`. `1-p_success` is `mu / (mu+r)`
    # nb_case = -torch.lgamma(r + eps) + torch.lgamma(y_true + r + eps) - torch.lgamma(y_true + 1.0) \
    #           + r * torch.log(r / (r + mu + eps) + eps) + y_true * torch.log(mu / (r + mu + eps) + eps)
    
    # Let's use the original structure for nb_case but ensure `pi_for_nb` is defined correctly.
    # The `y_pred` is the mean `mu`. The NB success probability `p` is `r / (r + mu)`.
    # So, `pi` in the original formula should be `r / (r + y_pred)`.
    pi_for_nb = r / (r + y_pred + eps) # success probability p for NB

    nb_case = (torch.lgamma(y_true + r) - torch.lgamma(y_true + 1.0) - torch.lgamma(r) +
               r * torch.log(pi_for_nb + eps) + y_true * torch.log(1.0 - pi_for_nb + eps))
    
    # Zero-inflated part
    # `prob_extra_zero_batch` is the probability of an extra zero (dropout probability, often denoted as psi or pi_zinb)
    # The likelihood for y=0 is: psi + (1-psi) * P(Y_NB = 0)
    # P(Y_NB=0) = (r / (r+mu))^r = pi_for_nb^r
    log_zero_part = torch.log(prob_extra_zero_batch + (1.0 - prob_extra_zero_batch) * torch.pow(pi_for_nb, r) + eps)
    log_non_zero_part = torch.log(1.0 - prob_extra_zero_batch + eps) + nb_case

    res = torch.where(y_true < eps, log_zero_part, log_non_zero_part)
    
    return -torch.mean(res)


def train_GMVAE(model, epoch, dataloader, optimizer, proportion_tensor, kl_weight, mapping_dict, color_map, max_epochs, gammas,device='cuda', base_dir=None, plot_umap=False):
    assert base_dir is not None
    assert(isinstance(base_dir, str) or isinstance(base_dir, Path))
    model.train()
    total_loss = 0
    model = model.to(device)

    for idx, (data, labels) in enumerate(dataloader):
        # mask = data != 0
        # nonzero_idx = mask.nonzero(as_tuple=False)
        # assert nonzero_idx.size(0) > 0, "ERROR: got an all-zero data batch!"
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        reconstructed, mus, logvars, pis, zs = model(data, labels)
        if epoch == 0 and idx == 0:
            print("************************************* First batch of data: *************************************************")
            print("reconstructed.shape:", reconstructed.shape)
            print("data.shape:", data.shape)
            print("mus.shape:", mus.shape)
            print("logvars.shape:", logvars.shape)
            print("pis.shape:", pis.shape)
            print("zs.shape:", zs.shape)
        
        assert(reconstructed.shape == data.shape)
        
        proportion_tensor_reshaped = proportion_tensor.to(pis.device)
        # import pdb; pdb.set_trace()

        fraction_loss =  F.mse_loss(pis.mean(0), proportion_tensor_reshaped)
        loss_recon = F.mse_loss(reconstructed, data)


        # print(data.shape, reconstructed.shape)
        # print(model.module.prob_extra_zero.shape)
        # print(model.module.over_disp.shape)

        # Select cell-type specific ZINB parameters for the batch
        batch_labels = labels.squeeze() # Ensure labels are 1D
        
        # Ensure model parameters are on the correct device
        raw_prob_extra_zero_all = model.module.prob_extra_zero.to(device)
        raw_over_disp_all = model.module.over_disp.to(device)

        # Apply activations to constrain parameter ranges
        prob_extra_zero_all_activated = torch.sigmoid(raw_prob_extra_zero_all)
        over_disp_all_activated = F.softplus(raw_over_disp_all) + 1e-6 # Add epsilon for numerical stability, ensuring it's strictly positive

        # Gather the parameters for each item in the batch
        # Ensure batch_labels are long type for indexing
        prob_extra_zero_batch = prob_extra_zero_all_activated[batch_labels.long()]
        over_disp_batch = over_disp_all_activated[batch_labels.long()]
        
        # Clamp over_disp to be positive to avoid issues with lgamma and log - No longer strictly needed due to softplus, but kept as a safeguard or can be removed.
        # over_disp_batch = torch.clamp(over_disp_batch, min=1e-4)

        zinb_loss_val = zinb_loss(data, reconstructed, prob_extra_zero_batch, over_disp_batch)

        # Calculate KL divergence (ensure it's batch-averaged or reasonably scaled)
        # Original: loss_kl = (0.5 * torch.sum(-1 - logvars + mus.pow(2) + logvars.exp())/1e9)
        # Sum over latent dimensions, then mean over batch and components if mus/logvars are per component before reparam.
        # Assuming mus and logvars here are for the reparameterized zs, shape (batch_size, latent_dim)
        # The VAE KL divergence is typically E_q(z|x)[log q(z|x) - log p(z)]
        # For a standard normal prior p(z) = N(0,I), this is 0.5 * sum(mus^2 + exp(logvars) - logvars - 1) per sample.
        # mus, logvars in your code are (batch_size, n_components, latent_dim)
        # zs is (batch_size, latent_dim) after reparameterize_with_labels
        # The KL should be calculated for the specific z sampled for each data point.
        # The current `mus` and `logvars` in scope here are the full (batch_size, n_components, latent_dim)
        # Let's re-evaluate how KL should be computed based on how `reparameterize_with_labels` works.

        # For the VAE, KL is calculated between q(z_k|x) and p(z_k) for the chosen component k.
        # mus_k = mus[torch.arange(data.shape[0]), labels.squeeze().long(), :]
        # logvars_k = logvars[torch.arange(data.shape[0]), labels.squeeze().long(), :]
        # kl_divergence_per_sample = 0.5 * torch.sum(-1 - logvars_k + mus_k.pow(2) + logvars_k.exp(), dim=1)
        # loss_kl_unscaled = torch.mean(kl_divergence_per_sample) # Mean over batch

        # Given the current structure, `mus` and `logvars` are output by encode and are (batch_size, n_components, latent_dim).
        # The `reparameterize_with_labels` selects one component's mu/logvar per sample and then takes their mean before reparameterizing.
        # This means the `zs` (latent variables) are derived from mean_mus and mean_logvars per component. 
        # This is a bit non-standard. Typically, you'd sample z ~ q(z|x,c) for each x and its label c.
        # Let's assume the KL should be based on the parameters of the specific component chosen for each sample.
        
        selected_mus = mus[torch.arange(mus.size(0)), labels.squeeze().long()] # Shape: (batch_size, latent_dim)
        selected_logvars = logvars[torch.arange(logvars.size(0)), labels.squeeze().long()] # Shape: (batch_size, latent_dim)

        loss_kl_unscaled = 0.5 * torch.sum(-1 - selected_logvars + selected_mus.pow(2) + selected_logvars.exp(), dim=1).mean()

        loss_kl_weighted = loss_kl_unscaled * kl_weight
        # The clipping might still be useful if KL can occasionally explode, but ensure kl_weight is appropriate.
        # loss_kl_weighted = 1 if loss_kl_weighted > 1 else loss_kl_weighted 
        # Let's remove the aggressive clipping for now, and rely on appropriate kl_weight.
        
        loss = loss_recon*gammas['recon'] + \
               zinb_loss_val*gammas['zinb'] + \
               fraction_loss*gammas['fraction'] + \
               loss_kl_weighted*gammas['kl']
        
        loss.backward()

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm().item()
        #         print(f"Grad | {name:30}: {grad_norm:.4e}")

        optimizer.step()
        print(colored(f"Loss: {loss.item():.4f}", 'magenta'))
        total_loss += loss.item()
    
    print(colored(f"fraction_loss: {fraction_loss:.4f}", 'cyan'))
    print(colored(f"loss_kl  unweighted: {loss_kl_unscaled:.4f}", 'blue'))
    print(colored(f"loss_kl  weighted: {loss_kl_weighted:.4f}", 'blue'))
    print(colored(f"kl_weight: {kl_weight:.4f}", 'blue'))
    print(colored(f"zinb_loss_val: {zinb_loss_val:.4f}", 'blue'))
    print(colored(f"loss_recon: {loss_recon:.4f}", 'blue'))
    print("Epoch finished")

    if ((epoch+1) % 10 == 0) and (epoch != max_epochs - 1):
        print("Saving intermediate results to folder:", base_dir)
        print(f'Epoch: {epoch+1} KL Loss: {loss_kl_unscaled:.4f}\n Recon Loss: {loss_recon:.4f}\n Total Loss: {total_loss:.4f}\n Fraction Loss: {fraction_loss:.4f}\n ZINB Loss: {zinb_loss_val:.4f}')

        # Save reconstructed.
        print("Saving reconstructed to folder:", base_dir)
        torch.save(reconstructed, base_dir + '/GMVAE_reconstructed.pt')

        mus = mus.mean(0)
        logvars = logvars.mean(0)
        pis = pis.mean(0)
        
        # Save the mean, logvar, and pi.
        print("Saving mus, logvars, and pis to folder:", base_dir)
        torch.save(mus, base_dir + '/GMVAE_mus.pt')
        torch.save(logvars, base_dir + '/GMVAE_logvars.pt')
        torch.save(pis, base_dir + '/GMVAE_pis.pt')
        print("GMVAE mu & var & pi saved.")

        model.eval()
        torch.save(model.state_dict(), base_dir + 'GMVAE_model.pt')
        print("GMVAE Model saved.")

        if plot_umap:
            print("Plotting UMAP...")
            k = labels.cpu().detach().numpy()
            
            # Generate QQ plot for reconstructed data.
            reconstructed = reconstructed.cpu().detach().numpy()

            z = zs.cpu().detach().numpy()

            # Convert all_labels to colors using the color_map
            label_map = {str(v): k for k, v in mapping_dict.items()}
            mean_colors = [color_map[label_map[str(label.item())]] for label in k]
            z_colors = [color_map[label_map[str(label.item())]] for label in k]

            # UMAP transformation of recon
            reducer = umap.UMAP()
            embedding_z = reducer.fit_transform(z)
            embedding_recon = reducer.fit_transform(reconstructed)
    

            plt.figure(figsize=(12, 10))
            plt.scatter(embedding_z[:, 0], embedding_z[:, 1], c=z_colors, s=5)
            # Remove ticks
            plt.xticks([])
            plt.yticks([])
            # Name the axes.
            plt.xlabel('UMAP1')
            plt.ylabel('UMAP2')
            plt.title('UMAP of reparameterized z')
            plt.savefig(base_dir + 'umap_latent.png')
            plt.close()

            plt.figure(figsize=(12, 10))
            plt.scatter(embedding_recon[:, 0], embedding_recon[:, 1], c=mean_colors, s=5)
            # Remove ticks
            plt.xticks([])
            plt.yticks([])
            # Name the axes.
            plt.xlabel('UMAP1')
            plt.ylabel('UMAP2')
            plt.title('UMAP of Reconstructed Data')
            plt.savefig(base_dir + 'umap_recon.png')
            plt.close()

    elif epoch == max_epochs - 1:
        print(colored(f"Saving final results to folder: {base_dir}", 'green'))
        

        print(f'Epoch: {epoch+1} KL Loss: {loss_kl_unscaled:.4f}\n Recon Loss: {loss_recon:.4f}\n Total Loss: {total_loss:.4f}\n Fraction Loss: {fraction_loss:.4f}\n ZINB Loss: {zinb_loss_val:.4f}')

        # Save reconstructed.
        print("Saving reconstructed to folder:", base_dir)
        torch.save(reconstructed, base_dir + 'GMVAE_reconstructed.pt')

        mus = mus.mean(0)
        logvars = logvars.mean(0)
        pis = pis.mean(0)

        # Save the mean, logvar, and pi.
        print("Saving mus, logvars, and pis to folder:", base_dir)
        torch.save(mus, base_dir + '/GMVAE_mus.pt')
        torch.save(logvars, base_dir + '/GMVAE_logvars.pt')
        torch.save(pis, base_dir + '/GMVAE_pis.pt')
        print("GMVAE mu & var & pi saved.")

        model.eval()
        torch.save(model.state_dict(), base_dir + '/GMVAE_model.pt')
        print("GMVAE Model saved.")
    
    return total_loss