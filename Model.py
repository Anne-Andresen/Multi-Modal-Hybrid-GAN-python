import torch
import torch.nn as nn
from attention import *
import torch
import torch.nn as nn



num_heads = 8
embed_dim = 128
'''
#class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=128, num_heads=8):
        super(UNet, self).__init__()
        self.attention = CrossAttention(embed_dim, num_heads)
        self.attention_block_1 = attention_block(32,8)
        self.attention_block_2 = attention_block(8, num_heads)

        # Encoder layers
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 64 x 64 x 64

        self.encoder2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 32 x 32 x 32

        self.encoder3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)  # 16 x 16 x 16
        # introduce attention block

        self.encoder4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)  # 8 x 8 x 8

        self.bottleneck = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)  # 16 x 16 x 16
        self.decoder4 = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)  # 32 x 32 x 32
        self.decoder3 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # introduce attention block

        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)  # 64 x 64 x 64
        self.decoder2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)  # 128 x 128 x 128
        self.decoder1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=1)
        )

    def forward(self, x, struct):
        x = self.attention(x, struct, struct)

        enc1 = self.encoder1(x)
        x = self.pool1(enc1)

        enc2 = self.encoder2(x)
        x = self.pool2(enc2)
        print('x shape before att: ', x.size())
        #self.attention_block = attention_block(x)
        x_att = self.attention_block_1(x)
        print('x after att: ', x_att.size())
        enc3 = self.encoder3(x_att)
        x = self.pool3(enc3)

        enc4 = self.encoder4(x)
        x = self.pool4(enc4)
        x_att0 = self.attention_block_2(x)

        x = self.upconv4(x_att0)
        x = torch.cat([x, enc4], dim=1)
        x = self.decoder4(x)
        x_att1 = self.attention_block(x)

        x = self.upconv3(x_att1)
        x = torch.cat([x, enc3], dim=1)
        x = self.decoder3(x)

        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.decoder2(x)

        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.decoder1(x)
        #print('shape: x', x.shape)
        return x




num_heads = 8
embed_dim = 128

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=128, num_heads=8):
        super(UNet, self).__init__()
        self.attention = CrossAttention(embed_dim, num_heads)
        #self.attention_block = attention_block(num_heads)
        self.attention_block_1 = attention_block(32,8)
        self.attention_block_2 = attention_block(8, num_heads)

        # Encoder layers
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 64 x 64 x 64

        self.encoder2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 32 x 32 x 32

        self.encoder3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)  # 16 x 16 x 16
        # introduce attention block

        self.encoder4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)  # 8 x 8 x 8

        self.bottleneck = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)  # 16 x 16 x 16
        self.decoder4 = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)  # 32 x 32 x 32
        self.decoder3 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # introduce attention block

        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)  # 64 x 64 x 64
        self.decoder2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)  # 128 x 128 x 128
        self.decoder1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=1)
        )

    def forward(self, x, struct):
        x = self.attention(x, struct, struct)

        enc1 = self.encoder1(x)
        x = self.pool1(enc1)

        enc2 = self.encoder2(x)
        x = self.pool2(enc2)
        #x = self.attention_block_1(x)

        enc3 = self.encoder3(x)
        x = self.pool3(enc3)

        enc4 = self.encoder4(x)
        x = self.pool4(enc4)

        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.decoder4(x)
        x= self.attention_block(x)

        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.decoder3(x)

        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.decoder2(x)

        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.decoder1(x)
        #print('shape: x', x.shape)
        return x
'''

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=128, num_heads=8):
        super(UNet, self).__init__()
        self.attention = CrossAttention(embed_dim, num_heads)
        #self.attention1 = CrossAttention(embed_dim, 2)
        self.attention_block_1 = attention_block(32,8)
        self.attention_block_2 = attention_block(8, num_heads)
        self.attention_block_3 = attention_block(16, num_heads)
        self.attention_block_4 = attention_block(64, num_heads)
        #self.BPE = encoding

        # Encoder layers
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 64 x 64 x 64

        self.encoder2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 32 x 32 x 32

        self.encoder3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)  # 16 x 16 x 16

        self.encoder4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)  # 8 x 8 x 8

        self.bottleneck = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)  # 16 x 16 x 16
        self.decoder4 = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)  # 32 x 32 x 32
        self.decoder3 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)  # 64 x 64 x 64
        self.decoder2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)  # 128 x 128 x 128
        self.decoder1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=1)
        )

    def forward(self, x, struct, dose):
        #encoded = self.BPE('glioma', '49')
        #print('encodedddd: ', encoded.shape)
        #encoded = encoded.to('cuda:0')
        x = self.attention(x, struct, struct)
        x = x + dose
        # print('out cross att: ', x.shape)
        #x = self.attention1(x, x, encoded)
        #x = torch.cat((x, encoded), dim=0)
        #x = x + encoded

        enc1 = self.encoder1(x)
        x = self.pool1(enc1)

        enc2 = self.encoder2(x)
        x = self.pool2(enc2)
        x = self.attention_block_1(x)

        enc3 = self.encoder3(x)
        x = self.pool3(enc3)

        enc4 = self.encoder4(x)
        x = self.pool4(enc4)
        #x = self.attention_block_2(x)

        x = self.bottleneck(x)

        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.decoder4(x)
        #x = self.attention_block_3(x)

        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.decoder3(x)

        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.decoder2(x)
        x = self.attention_block_4(x)

        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.decoder1(x)
        #print('shape: x', x.shape)
        return x



class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv3d(input_size, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(256)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(512)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.fc = nn.Linear(512 * 16 * 16 * 16, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu4(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc()
        x = torch.sigmoid(x)
        return x
class GAN(nn.Module):
    """GAN model"""
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        for param in self.discriminator.parameters():
            param.requires_grad = False

    def forward(self, x, labels):
        gen_img = self.generator(x)
        return self.discriminator(gen_img, labels)
