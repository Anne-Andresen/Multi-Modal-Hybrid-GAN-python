import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from attention import *
from tensorboardX import SummaryWriter
from dataloader import *
from Model import *
from earlyStopping import *

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    G = UNet3D(1, 1).to(device)
    D = Discriminator(1).to(device)
    G = DDP(G, device_ids=[rank])
    D = DDP(D, device_ids=[rank])
    gan_model = GAN(G, D).to(device)  # Renamed variable to avoid conflict
    
    criterion = nn.MSELoss()
    optimizerG = optim.Adam(G.parameters(), lr=0.0002)
    optimizerD = optim.Adam(D.parameters(), lr=0.0002)
    writer = SummaryWriter('runs')
    best_loss = float('inf')
    patience = 10
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    num_epochs = 10000
    batch_size = 4
    embed_dim = 128
    num_heads = 8
    
    dataset = CustomDataset('/with_structs/')
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for real_data, real_labels, real_struct in dataloader:
            real_data, real_labels, real_struct = real_data.to(device), real_labels.to(device), real_struct.to(device)
            
            if rank == 0:
                print(f"Data shapes - real_data: {real_data.shape}, real_labels: {real_labels.shape}, real_struct: {real_struct.shape}")
            
            # Train Discriminator
            D.zero_grad()
            real_output = D(real_labels)
            label_fake = torch.zeros_like(real_output)
            real_loss = criterion(real_output, label_fake)
            real_loss.backward()
            
            fake_data = G(real_data, real_struct)
            fake_output = D(fake_data)
            d_real = D(real_labels)
            fake_loss = criterion(fake_output, d_real)
            fake_loss.backward()
            optimizerD.step()
            
            # Train Generator
            D.zero_grad()
            gen_output = D(G(real_data, real_struct))
            label_fake_gen = torch.zeros_like(gen_output)
            gen_loss = criterion(gen_output, label_fake_gen)
            gen_loss.backward()
            optimizerG.step()
        
            if rank == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Discriminator Loss: {real_loss.item() + fake_loss.item():.4f}, Generator Loss: {gen_loss.item():.4f}")
                writer.add_scalar('Discriminator Loss', real_loss.item() + fake_loss.item(), epoch)
                writer.add_scalar('Generator Loss', gen_loss.item(), epoch)
        
            current_loss = real_loss.item() + fake_loss.item()
            early_stopping(current_loss, G.module, D.module, epoch)
        
            if (epoch + 1) % 10 == 0 and rank == 0:
                torch.save(G.module.state_dict(), f"./models/G_epoch_{epoch + 1}.pth")
                torch.save(D.module.state_dict(), f"./models/D_epoch_{epoch + 1}.pth")
                print(f"Models saved at epoch {epoch + 1}")
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
