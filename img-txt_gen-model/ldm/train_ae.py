from model.autoencoder import Autoencoder, Encoder, Decoder
import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision
from torchvision import transforms 
from loss.loss import LPIPSWithDiscriminator

if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device="cuda"
    print(device)
    encoder = Encoder(channels=128,channel_multipliers=[1,2,4,4],n_resnet_blocks=2,in_channels=3,z_channels=4).to(device)
    
    decoder = Decoder(channels=128,channel_multipliers=[1,2,4,4],n_resnet_blocks=2,out_channels=3,z_channels=4).to(device)
    
    model  = Autoencoder(encoder,decoder,emb_channels=4,z_channels=4).to(device)
    loss_fn = LPIPSWithDiscriminator(disc_start=50001).to(device)
    lr = 4.5e-6
    opt_ae = torch.optim.Adam(list(model.encoder.parameters())+
                              list(model.decoder.parameters())+
                              list(model.quant_conv.parameters())+
                              list(model.post_quant_conv.parameters()),
                              lr=lr, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(loss_fn.discriminator.parameters(),
                                lr=lr, betas=(0.5, 0.9)) 
    # train_data = torchvision.datasets.CIFAR10(root = "/home/liqi/datasets", train = True, download = True, transform=transforms.ToTensor())
    # test_data = torchvision.datasets.CIFAR10(root = "/home/liqi/datasets", train = False, download = True, transform=transforms.ToTensor())
    train_transform = transforms.Compose(
        [transforms.Resize(256),
        transforms.ToTensor()]
    )
    batch_size = 16
    train_data = ImageFolder(root = "/home/liqi/datasets/manga", transform=train_transform)
    # test_data = ImageFolder(root = "/home/liqi/datasets/manga",  transform=transforms.ToTensor())

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    step_per_epoch = len(train_data)//batch_size
    model.train()
    epochs = 10
    print_step = 100
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(train_dataloader):
            print(f"training epoch/epochs: [{epoch}]/10")
            X, y = X.to(device), y.to(device)

            
            posterior = model.encode(X)
            
            z = posterior.sample()
            
            reconstructions = model.decode(z)
            
            ae_loss, _ = loss_fn(X, reconstructions, posterior, 0, epoch*step_per_epoch+batch,
                                 last_layer=model.decoder.conv_out.weight, split="train")
            
            opt_ae.zero_grad()
            ae_loss.backward()
            opt_ae.step()

            disc_loss, _ = loss_fn(X, reconstructions, posterior, 1, epoch*step_per_epoch+batch,
                                  last_layer=model.decoder.conv_out.weight, split="train")

            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()
            if bacth % print_step == 0:
                print(f"training epoch/epochs: [{epoch}]/10, ae_loss is {ae_loss}, disc_loss is {disc_loss}.")
        

        torch.save(model.state_dict(), "ckpt/ae.pt")
            
