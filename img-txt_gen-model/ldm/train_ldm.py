import torch
import copy
import torchvision
from torchvision import transforms 
from torchvision.datasets import ImageFolder
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from latent_diffusion import LatentDiffusion
from model.autoencoder import Autoencoder, Encoder, Decoder
from model.clip_embedder import CLIPTextEmbedder
from model.unet import UNetModel
from sampler.ddim import DDIMSampler
from sampler.ddpm import DDPMSampler

if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device="cuda"
    
    unet_model = UNetModel(in_channels=4,
                            out_channels=4,
                            channels=192,
                            n_res_blocks=2,
                            attention_levels=[1,2,4,8],
                            channel_multipliers=[1,2,2,4,4],
                            n_heads=8,
                            ).to(device)
    encoder = Encoder(channels=128,channel_multipliers=[1,2,4,4],n_resnet_blocks=2,in_channels=3,z_channels=4).to(device)
    
    decoder = Decoder(channels=128,channel_multipliers=[1,2,4,4],n_resnet_blocks=2,out_channels=3,z_channels=4).to(device)
    
    autoencoder  = Autoencoder(encoder,decoder,emb_channels=4,z_channels=4).to(device) 
    autoencoder.load_state_dict(torch.load("ckpt/ae.pt"))
    clip_embedder = CLIPTextEmbedder().to(device)
    model = LatentDiffusion(unet_model,autoencoder,clip_embedder,8,1000,0.0015,0.155)
    # ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    sampler = DDPMSampler(model)
    train_transform = transforms.Compose(
        [transforms.Resize(256),
        transforms.ToTensor()]
    )
    # train_data = torchvision.datasets.CIFAR10(root = "/home/liqi/datasets", train = True, download = True, transform=train_transform)
    # test_data = torchvision.datasets.CIFAR10(root = "/home/liqi/datasets", train = False, download = True, transform=train_transform)
    train_data = ImageFolder(root = "/home/liqi/datasets/manga", transform=train_transform)
    batch_size = 1

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    epochs = 10
    print_step = 100
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-5, weight_decay=0.001)
    for epoch in range(epochs):
        for batch, (x, y) in enumerate(train_dataloader):
            print(f"training epoch/epochs: [{epoch}]/10")
            x = x.to(device)
            x.requires_grad_(True)
            
            x = model.autoencoder_encode(x)
            
            y = [str(i) for i in y.tolist()]
            
            c = model.get_text_conditioning(y)
            
            t = torch.randint(low=1, high=model.n_steps, size=(batch_size,)).to(device)
            
            x_prev, x0, e_t = sampler.p_sample(x, c, t, int(t[0].item()))
          
            loss = loss_fn(x_prev,e_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if bacth % print_step == 0:
                print(f"training epoch/epochs: [{epoch}]/10, loss is {loss}")
        
        torch.save(model.model.state_dict(), "ckpt/ldm.pt")

