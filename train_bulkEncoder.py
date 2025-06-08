import torch
import torch.nn.functional as F
from termcolor import colored

def train_BulkEncoder(epoch, model, GMVAE_model, max_epochs, optimizer, dataloader, scMus, scLogVars, scPis, device='cuda', base_dir='saved_files/'):

    model.train()
    model = model.to(device)
    GMVAE_model.eval()
    GMVAE_model = GMVAE_model.to(device)

    for _, (data, labels) in enumerate(dataloader):
        data = data.to(device)

        # You can use scMu and scLogVar from GMVAE_model to train bulkEncoder_model or
        # run GMVAE_model on the data and use the output to train bulkEncoder_model.
        bulk_data = data.sum(dim=0)
        # print(f"Sum expression per sample: {bulk_data.sum()}")
        bulk_data = bulk_data.unsqueeze(0)

        mus, logvars, pis = model(bulk_data) # Predict mus (means), logvars (variances), pis (proportions)

        mus = mus.squeeze()
        logvars = logvars.squeeze()
        pis = pis.squeeze()

        assert(mus.shape == scMus.shape)
        mus_loss = F.mse_loss(mus, scMus)
        assert(logvars.shape == scLogVars.shape)
        logvars_loss = F.mse_loss(logvars, scLogVars)
        assert(pis.shape == scPis.shape)
        pis_loss = F.mse_loss(pis, scPis)

        # C = pis.shape[0]
        # uniform = torch.full_like(pis, 1.0 / C)
        # uniform_penalty = F.cosine_similarity(pis.unsqueeze(0), uniform.unsqueeze(0)).squeeze()

        loss = mus_loss + logvars_loss + (pis_loss * ((epoch+1) // 10))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(colored(f"Loss: {loss.item():.4f}", 'magenta'))

    # print(colored(f"Uniform penalty: {uniform_penalty}", 'cyan'))
    print(colored(f"pis_loss: {pis_loss}", 'cyan'))
    print(colored(f"mus_loss: {mus_loss}", 'cyan'))
    print(colored(f"logvars_loss: {logvars_loss}", 'cyan'))

    if (epoch+1)%10==0:
        print("Epoch[{}/{}]: mus_loss:{:.3f}, vars_loss:{:.3f}, pis_loss:{:.3f}".format(epoch+1,
                                                                                        max_epochs,
                                                                                        mus_loss.item(),
                                                                                        logvars_loss.item(),
                                                                                        # h0_loss.item(),
                                                                                        pis_loss.item()))

    if (epoch+1) % 20== 0 and epoch != max_epochs - 1:
        print(colored(f"Saving an intermediate checkpoint of bulkEncoder model to {base_dir}...", "yellow"))
        torch.save(model.state_dict(), base_dir + '/bulkEncoder_model.pt')
    
    if epoch == max_epochs - 1:
        print(colored("Saving last checkpoint of bulkEncoder model...", "yellow"))
        torch.save(model.state_dict(), base_dir + '/bulkEncoder_model.pt')

