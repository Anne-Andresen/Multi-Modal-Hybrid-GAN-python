import torch
import torchvision.transforms as transforms
import SimpleITK as sitk
import numpy as np
import glob
# Define your model architecture
class YourModel(torch.nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # Define your model layers here

    def forward(self, x):
        # Define the forward pass of your model
        return x

# Load the trained model
def load_model(model_path):
    model = YourModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Preprocess the image before feeding it to the model
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add any necessary transformations here
    ])
    return transform(image).unsqueeze(0)

# Test the model on a single image
def test_single_image(model, image_path, output_path):
    # Read the image using SimpleITK
    image = sitk.ReadImage(image_path)
    # Convert the SimpleITK image to a numpy array
    image_array = sitk.GetArrayFromImage(image)
    # Preprocess the image
    input_tensor = preprocess_image(image_array)
    # Convert the tensor to a torch Variable
    input_tensor = input_tensor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
    # Post-process the output if needed
    # For example, convert tensor to numpy array
    output_array = output.cpu().numpy()
    # Perform any necessary post-processing here

    # Save the output image
    output_image = sitk.GetImageFromArray(output_array.squeeze())
    output_image.CopyInformation(image)
    sitk.WriteImage(output_image, output_path)

    return output_array

# Test multiple images us

# Test multiple images using the model
def test_multiple_images(model, image_paths, output_dir):
    results = []
    for image_path in image_paths:
        output_path = output_dir + "/" + image_path.split("/")[-1]  # Output path for the predicted image
        result = test_single_image(model, image_path, output_path)
        results.append(result)
    return results

if __name__ == "__main__":
    # Path to the trained model
    model_path = "best.pth"
    # Load the model
    model = load_model(model_path)
    generic_image_path = ''
    image_paths = glob.glob(generic_image_path)
    # Path to the test images
    #image_paths = ["test_image1.nii", "test_image2.nii", "test_image3.nii"]  # Update with your test image paths

    # Directory to save the predicted output images
    output_dir = "predicted_output"

    # Test the model on multiple images
    results = test_multiple_images(model, image_paths, output_dir)
    # Process the results as needed
    print("Output images saved in:", output_dir)
