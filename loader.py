from torch.utils.data import Dataset
import os
import numpy as np
from skimage import io
import torch
from torchvision import transforms

class GMKDataset(Dataset):
    def __init__(self, img_dir, device, classes, transform=None):
        
        self.classes = classes
        self.device = device
        
        img_paths = []
        for dir in img_dir:
            for file in os.listdir(dir):
                if ("mask" not in file) & (file.endswith(".tif")):
                    img_paths.append(os.path.join(dir, file))
                
        self.img_paths = img_paths
        
        self.transform = transform
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx): 
        
        img_path = self.img_paths[idx]
        mask_path = img_path.replace(".tif", "_mask.tif")
        
        img_arr = np.atleast_3d(io.imread(img_path)) #use only the DTM
                
        gmk_arr = np.atleast_3d(io.imread(mask_path))       
        gmk_arr -= 1    #labels range from 1 to 20; they need to start with 0
        gmk_arr = np.clip(gmk_arr,a_min=0, a_max=self.classes-1).astype(np.uint8)
               
        # put it from HWC to CHW format
        # img_tensor = np.transpose(img_arr, (2, 0, 1))
        img_tensor = self.transform(img_arr)
        mask_tensor = np.transpose(gmk_arr, (2, 0, 1))
        
        assert mask_tensor.dtype == np.uint8, "Mask must be np.uint8"
        assert img_tensor.dtype == torch.float32, "Image must be np.float32 not %s" % str(img_tensor.dtype)
        # assert torch.max(img_tensor) < 1, "Image exceeds range [0, 1]"
        
        #https://github.com/qubvel-org/segmentation_models.pytorch/discussions/454
        return img_tensor.to(self.device), torch.from_numpy(mask_tensor).long().to(self.device)