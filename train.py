from attention import *
from tensorboardX import SummaryWriter
from dataloader import *
from Model import *
from dataloader import *
# Define the GAN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from earlyStopping import *
# Define the GAN model
# Define the GAN model
G = UNet3D(1, 1).to(device)
D = Discriminator(1).to(device)
GAN = GAN(G, D).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizerG = optim.Adam(G.parameters(), lr=0.00001)
optimizerD = optim.Adam(D.parameters(), lr=0.00001)
writer = SummaryWriter('runs')
best_loss = float('inf')
patience = 10
early_stopping = EarlyStopping(patience=patience, verbose=True)

# Define the number of epochs and batch size
num_epochs = 10000
batch_size = 4
#num_heads = 2
embed_dim = 128
#embed_dim = 128
num_heads = 8
#cross_attention = CrossAttention(embed_dim, num_heads)
#output = cross_attention(torch_tensor_img, torch_tensor_struct, torch_tensor_struct)
# Define the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = CustomDataset('/processing/annand/with_structs/')
dataloader = DataLoader(dataset, batch_size=batch_size)
# Define the training loop
for epoch in range(num_epochs):
    for real_data, real_labels, real_struct in dataloader:
        #merged_tensor = merge_tensors_3d(real_data, real_struct)
        #print('shapingg', real_struct.shape, real_data.shape)
        input_size = 1
        #cross_attention_merge = CrossAttentionMerge(input_size)
        #merged_tensor = merge_tensors_3d(real_data, real_struct)
        # Creating the cross-attention module
        #cross_attention = CrossAttention(embed_dim, num_heads)
        #embed_dim = 128
        #num_heads = 8
        #cross_attention = CrossAttention(embed_dim, num_heads)
        #output = cross_attention(real_data, real_struct, real_struct)

# Applying cross-attention to merge the images
        #output = cross_attention(real_struct, real_data)
        #print('merged_tensor shape: ', outout.shape)
        #merged_tensor = output.numpy()
        #merged_tensor = torch.from_numpy(merged_tensor).to(torch.float32)
        #print('merged_tensor: ', merged_tensor.shape)
        #merged_tensor = torch.tensor(merged_tensor)
        real_data, real_labels, merged_tensor = real_data.to(device), real_labels.to(device), real_struct.to(device)
        #print('real data shape', real_data.shape, real_labels.shape)

        # Train the discriminator on real data
        D.zero_grad()
        real_output = D(real_labels)
        real_labels = real_labels.to(device)
        #print('reals output: ', real_output)
        label_fake =torch.zeros_like(real_output)
        #D_fake = D()
        real_loss = criterion(real_output, label_fake)
        real_loss.requires_grad = True
        real_loss.backward()
        #print('here now')

        # Train the discriminator on fake data
        fake_data = G(real_data, merged_tensor)
        fake_output = D(fake_data)
        d_real = D(real_labels)
        #fake_labels = torch.zeros(fake_data.size(0)).to(device)
        fake_loss = criterion(fake_output, d_real)
        #print('ehre here now: ')
        fake_loss.backward()

        # Update the discriminator weights
        optimizerD.step()
        #print('disc over moving to gen')

        # Train the generator
        D.zero_grad()
        gen_output = D(G(real_data, merged_tensor))
        label_fake_gen =torch.zeros_like(gen_output)
        #gen_labels = torch.ones(real_data.size(0)).to(device)
        #print('output gen shape: ', gen_output.shape, real_labels.shape)
        gen_loss = criterion(gen_output,label_fake_gen)
        gen_loss.backward()

        # Update the generator weights
        optimizerG.step()

    # Print the loss for each epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Discriminator Loss: {real_loss.item() + fake_loss.item():.4f}, Generator Loss: {gen_loss.item():.4f}")
    # Log the losses to TensorBoard
        writer.add_scalar('Discriminator Loss', real_loss.item() + fake_loss.item(), epoch)
        writer.add_scalar('Generator Loss', gen_loss.item(), epoch)

    # Check for early stopping
        current_loss = real_loss.item() + fake_loss.item()
        early_stopping(current_loss, G, D, epoch)
    # Save the models every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(G.state_dict(), f"./models/G_epoch_{epoch + 1}.pth")
        torch.save(D.state_dict(), f"./models/D_epoch_{epoch + 1}.pth")
        print(f"Models saved at epoch {epoch + 1}")
