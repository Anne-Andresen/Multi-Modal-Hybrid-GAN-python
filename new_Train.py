import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from attention import CrossAttention
from dataloader import CustomDataset
from earlyStopping import EarlyStopping
from Model import Discriminator, GAN
from Model import UNet
from attention import *
# Define the GAN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = UNet(1, 1).to(device)
D = Discriminator(1).to(device)

# Define the loss function and optimizer
criterion = nn.BCELoss()
criterion_1 = nn.MSELoss()
optimizerG = optim.Adam(G.parameters(), lr=0.00001)
optimizerD = optim.Adam(D.parameters(), lr=0.00001)

# Set up TensorBoard
writer = SummaryWriter('runs')

# Set up early stopping
best_loss = float('inf')
patience = 10
early_stopping = EarlyStopping(patience=patience, verbose=True)
last_generator_loss = float('inf')  # initialize last generator loss to infinity
best_model_state = None  # initialize best model state
# Define the number of epochs and batch size
num_epochs = 10000
batch_size = 1
'''
with open('loss.txt', 'w') as f:
    f.write('Epoch,Discriminator Loss,Generator Loss\n')
'''
# Load the dataset
dataset = CustomDataset('/processing/annand/with_structs/')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the training loop
for epoch in range(num_epochs):
    #print('device; ', device)
    for real_data, real_labels, real_struct in dataloader:
        real_data, real_labels, real_struct = real_data.to(device), real_labels.to(device), real_struct.to(device)
        encoded = encoding('glioma', '49')
        #print('encodedddd: ', encoded.shape)
        encoded = encoded.to(device)
        # Train the discriminator on real data
        D.zero_grad()
        real_output = D(real_labels)
        real_loss = criterion(real_output, torch.ones_like(real_output))
        real_loss.backward()

        # Train the discriminator on fake data
        fake_data = G(real_data, real_struct, encoded)
        fake_output = D(fake_data.detach())
        fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
        fake_loss.backward()

        # Update the discriminator weights
        optimizerD.step()

        # Train the generator
        G.zero_grad()
        gen_output = D(fake_data)
        #print('shape generator output: ', fake_data.shape,'real output shape', real_labels.shape)
        gen_loss = criterion_1(fake_data, real_labels)
        gen_loss.backward()
        if gen_loss < last_generator_loss:
            last_generator_loss = gen_loss
            best_model_state = G.state_dict()  # save current generator model state
            torch.save(best_model_state, './gen_models/best_generator_'+str(last_generator_loss.item()) + '.pth')  # save generator model to file
            with open('./gen_models/best_loss.txt', 'w') as f:
                f.write(f'Best Generator Loss: {gen_loss.item():.4f}\n')
                f.write(f'Epoch: {epoch + 1}\n')

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


# Close the TensorBoard writer
writer.close()

