import torch
import glob
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import SimpleITK as sitk
import os
import torchvision.transforms as transforms
'''
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        # Initialize dataset
        self.data_dir = data_dir
        self.transform = transform
        #label_file = os.path.join(self.data_dir, 'label.xlsx')
        #self.labels = label_file
        #label_df = pd.read_excel(self.labels, ',')

        # Load and preprocess data (replace this with your own dataset loading and >        self.data = []  # List to store data samples
        self.lab = []
        patient_lst = glob.glob(data_dir+'/CTs/*.nii.gz')
        for i in patient_lst:  # Assume 1000 samples
            # Example: Load 3D and 2D images (replace this with your own data loadi>
            name = os.path.basename(i)
            image_3d_path = f"{data_dir}/CTs/{name}"
            label_3d_path = f"{data_dir}/Doses/{name}"
            print('image, label: ', image_3d_path, label_3d_path)
            #image_2d_path = f"{data_dir}/2D/{i}.jpg"
            #print('loaded 3d labels')
            label_img = sitk.ReadImage(label_3d_path)
            label_img = sitk.GetArrayFromImage(label_img)
            #for x, y,z in zip(label_img[1], label_img[0], label_img[2]):
            #    print(f"as value {label_img[y, x, z]}")
            scale_value = 1e-9
            label_img = np.round(np.array(label_img*scale_value))
            print('here: ',label_img.shape)
            image_3d = sitk.ReadImage(image_3d_path)
            image_3d = sitk.GetArrayFromImage(image_3d)
            #print('dose values', label_img)

            label_3d = label_img.astype(np.float32)
            label_3d = torch.from_numpy(label_3d).to(torch.float32)
            print('ges', label_3d.shape)
            image_3d = image_3d.astype(np.float32)
            image_3d = torch.from_numpy(image_3d).to(torch.float32)
            #print('reading 3d images', label_3d.shape)

            # for x, y,z in zip(label_img[1], label_img[0], label_img[2]):
            #    print(f"Pixel at ({x}, {y}, {z}) has value {label_img[y, x, z]}")

            # Example: Apply transformations (replace this with your own transforma>            if self.transform:

            # image_2d = self.transform(image_2d)

            # self.data.append((image_3d, label_3d))
        self.images = image_3d
        self.labels = label_3d

        # label = label_df.loc[i, 'Label']  # Assuming the label column in Excel>            #self.lab.append(label)
        # print('data loaded')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images
        label = self.labels
        print('image ship', image.shape)
        return image, label

    # Define transformations (replace with your own transformations)

    transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.5,), (0.5,))
    ])

'''
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Load paths to CT and Dose files
        self.ct_paths = glob.glob(os.path.join(data_dir, 'CTs/re_sized/', '*.nii.gz'))
        self.dose_paths = glob.glob(os.path.join(data_dir, 'Doses/re_sized/', '*.nii.gz'))
        self.struct_paths = glob.glob(os.path.join(data_dir, 'combined_structs/re_sized/', '*.nii.gz'))


    def __len__(self):
        return len(self.ct_paths)

    def __getitem__(self, idx):
        ct_path = self.ct_paths[idx]
        dose_path = self.dose_paths[idx]
        struct_path = self.struct_paths[idx]

        # Read CT and Dose images
        ct_image = sitk.ReadImage(ct_path)
        dose_image = sitk.ReadImage(dose_path)
        struct_img = sitk.ReadImage(struct_path)

        # Convert images to arrays
        ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float32)
        dose_array = sitk.GetArrayFromImage(dose_image).astype(np.float32)
        struct_array = sitk.GetArrayFromImage(struct_img).astype(np.float32)
        ct_array = torch.from_numpy(ct_array).to(torch.float32)
        dose_array = torch.from_numpy(dose_array).to(torch.float32)
        struct_array = torch.from_numpy(struct_array).to(torch.float32)

        # Apply transformations
        if self.transform:
            ct_array = self.transform(ct_array)
            dose_array = self.transform(dose_array)
            struct_array = self.transform(struct_array)

        # Add batch and channel dimensions
        ct_array = ct_array.unsqueeze(0)  # Add batch dimension
        dose_array = dose_array.unsqueeze(0)  # Add batch dimension
        struct_array = struct_array.unsqueeze(0)
        #print('dataloaded: ')

        return ct_array, dose_array, struct_array
