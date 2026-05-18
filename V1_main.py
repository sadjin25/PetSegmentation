import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import cv2
import glob

IMG_PATH = "./Datas/images/images/"
MASK_PATH = "./Datas/annotations/trimaps/"
IMG_SIZE = (800, 800)

def GetPaddedImage(img):
    targetH,targetW = 800, 800
    h,w = img.shape[:2]

    padH = targetH - h
    padW = targetW - w

    if padH < 0 or padW < 0:
        raise ValueError(f"image size is over {targetH}!!")
    
    paddedImg = cv2.copyMakeBorder(
        img,
        top=0,
        left=0,
        bottom=padH,
        right=padW,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    return paddedImg

class PetImages(Dataset):
    def __init__(self):
        self.img_path = sorted(glob.glob(IMG_PATH + "*.jpg"))
        self.mask_path = sorted(glob.glob(MASK_PATH + "*.jpg"))

    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        img_each_path = self.img_path[idx]

        img = cv2.imread(img_each_path, cv2.IMREAD_COLOR_RGB)
        if img is None:
            raise RuntimeError(f"Failed to read img: {img_each_path}")
        
        img = GetPaddedImage(img)
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0 # (H,W,C) -> (C,H,W), uint8 to float 0~1

        mask_each_path = self.mask_path[idx]

        mask = cv2.imread(mask_each_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_each_path}")

        mask = (mask == 1)
        mask = GetPaddedImage(mask)
        mask = torch.from_numpy(mask).long()

        return img, mask

