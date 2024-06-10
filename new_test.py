import os.path
import torch.nn as nn
from attention import *
import torch
import torchvision.transforms as transforms
import SimpleITK as sitk
import numpy as np
import glob
from Model import UNet
# Define your model architecture
num_heads = 8
embed_dim = 128
'''
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.attention = CrossAttention(embed_dim, num_heads)

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(1024, 1024, kernel_size=2, stride=2),
            nn.Conv3d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, out_channels, kernel_size=1)
        )

        self.final_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x, struct):
        x = self.attention(x, struct, struct)
        x = self.encoder(x)
        x = self.decoder(x)
        return self.final_conv(x)
'''
def maybe_mkdir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

# Load the trained model
def load_model(model_path):
    model = UNet(1, 1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Test the model on a single image
def test_single_image(model, data_path, output_path):
    # Read the image using SimpleITK
    image_path = os.path.join(data_path, 'CTs/re_sized_test_CTs/')
    struc_path = os.path.join(data_path, 'combined_structs/re_sized_test_structs/')
    image = sitk.ReadImage(image_path)
    struct = sitk.ReadImage(struc_path)

    # Convert the SimpleITK image to a numpy array
    image_array = sitk.GetArrayFromImage(image)
    struct_array = sitk.GetArrayFromImage(struct)
    # Preprocess the image
    input_tensor = torch.from_numpy(image_array).to(torch.float32)
    struct_tensor = torch.from_numpy(struct_array).to(torch.float32)
    #input_tensor = preprocess_image(image_array)
    # Convert the tensor to a torch Variable
    #input_tensor = input_tensor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    #struct_tensor = struct_tensor.to(torch.device('cuda' if  torch.cuda.is_available() else 'cpu'))

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor, struct_tensor)
    # Post-process the output if needed
    # For example, convert tensor to numpy array
    output_array = output.numpy()
    # Perform any necessary post-processing here

    # Save the output image
    output_image = sitk.GetImageFromArray(output_array.squeeze())
    output_image.CopyInformation(image)
    sitk.WriteImage(output_image, output_path)

    return output_array

def single_image(model, image_path, struct_path, output_path):
    # Read the image using SimpleITK
    #struc_path = os.path.join(data_path, 'Structs')
    image = sitk.ReadImage(image_path)
    struct = sitk.ReadImage(struct_path)

    # Convert the SimpleITK image to a numpy array
    image_array = sitk.GetArrayFromImage(image)
    struct_array = sitk.GetArrayFromImage(struct)
    # Preprocess the image
    input_tensor = torch.from_numpy(image_array).to(torch.float32).unsqueeze(0).unsqueeze(0)
    struct_tensor = torch.from_numpy(struct_array).to(torch.float32).unsqueeze(0).unsqueeze(0)
    #input_tensor = preprocess_image(image_array)
    # Convert the tensor to a torch Variable
    print('input tensor: ', input_tensor.shape)
    #input_tensor = input_tensor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    #struct_tensor = struct_tensor.to(torch.device('cuda' if  torch.cuda.is_available() else 'cpu'))

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor, struct_tensor)
    # Post-process the output if needed
    # For example, convert tensor to numpy array
    output_array = output.cpu().numpy()
    # Perform any necessary post-processing here

    # Save the output image
    output_image = sitk.GetImageFromArray(output_array.squeeze())
    output_image.CopyInformation(image)
    print('output path: ', output_path)
    sitk.WriteImage(output_image, output_path)

    return output_array

# Test multiple images us

# Test multiple images using the model
def test_multiple_images(model, data_path, output_dir):
    results = []
    image_paths = data_path + 'CTs/re_sized_test_CTs/*.nii.gz'
    output_path=  './predicted_output/'
    print('here: ', data_path)
    image_paths = glob.glob(image_paths)
    struct_path = data_path +'/combined_structs/re_sized_test_structs/'
    for image_path in image_paths:
        patient = os.path.basename(image_path)
        print('patient: ', patient)
        structs = struct_path + '/' + patient
        #output_path = output_dir + "/" + image_path.split("/")[-1]# Output path for the predicted image
        print('namem: ', output_path, image_path, structs)
        #struct = structs[0]
        #output_path = os.path.dirname(os.path.dirname(output_dir))
        print('ewfcf ', output_path)
        output_path_s = output_path + '/' +patient
        result = single_image(model, image_path, structs, output_path_s)
        results.append(result)
    return results

if __name__ == "__main__":
    # Path to the trained model
    model_path = "/home/annand/dose_attention/gen_models/best_generator_0.004636445082724094.pth"
    # Load the model
    model = load_model(model_path)
    generic_image_path = '/processing/annand/with_structs/'
    #image_paths = glob.glob(os.path.join(generic_image_path, '/CTs/*.nii.gz'))
    # Path to the test images

    #image_paths = ["test_image1.nii", "test_image2.nii", "test_image3.nii"]  # Update with your test image paths

    # Directory to save the predicted output images
    output_dir = "/home/annand/dose_attention/predicted_output/"
    maybe_mkdir(output_dir)
    test_multiple_images(model, generic_image_path, output_dir)

    # Test the model on multiple images
    #results = test_multiple_images(model, image_paths, output_dir)
    # Process the results as needed
    print("Output images saved in:", output_dir)

