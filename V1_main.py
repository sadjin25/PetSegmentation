import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import cv2
import glob
import os

IMG_PATH = "./Datas/images/images/"
MASK_PATH = "./Datas/annotations/trimaps/"
IMG_SIZE = (384, 384)
EPOCH_SIZE = 5
TRAIN_BATCH_SIZE = 2
TRAINED_DATA_PATH = "./Datas/trained_data/pet_seg.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def GetResizedImage(img, isMask=False):
    if(isMask is True) :
        paddedImg = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    else:
        paddedImg = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)

    return paddedImg

class PetImages_Dataset(Dataset):
    def __init__(self):
        self.img_path = sorted(glob.glob(IMG_PATH + "*.jpg"))
        self.mask_path = sorted(
            each for each in glob.glob(MASK_PATH + "*.png")
            if not os.path.basename(each).startswith("._")
        )
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        img_each_path = self.img_path[idx]

        img = cv2.imread(img_each_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read img: {img_each_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # (H,W,C)
        
        img = GetResizedImage(img)
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0 # (H,W,C) -> (C,H,W), uint8 to float 0~1

        mask_each_path = self.mask_path[idx]

        mask = cv2.imread(mask_each_path, cv2.IMREAD_UNCHANGED) # (H,W)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_each_path}")

        mask = (mask != 2)
        mask = GetResizedImage(mask, isMask=True)
        mask = torch.from_numpy(mask).unsqueeze(0).float() #img와 형 일치

        return img, mask
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                                kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        self.shortcut_bn = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = self.shortcut_bn(self.shortcut(x)) # outchannels * 32 * 32
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        feat = self.res1(x)
        down = self.pool(feat)
        return feat, down

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res1 = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # x.shape = [B,C,H,W], correction of convtranspose err 
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        
        x = torch.cat((x, skip), dim=1)
        x = self.res1(x)
        return x

class PetSegModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()

        # Encoder
        self.down1 = DownBlock(in_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        # Bottleneck
        self.res1 = ResidualBlock(512, 1024)
        self.dropout_res = nn.Dropout2d(p=0.3)

        # Decoder
        self.up4 = UpBlock(1024, 512, 512)
        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)

        # Output
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        feat1, x = self.down1(x)
        feat2, x = self.down2(x)
        feat3, x = self.down3(x)
        feat4, x = self.down4(x)

        x = self.res1(x)
        x = self.dropout_res(x)

        x = self.up4(x, feat4)
        x = self.up3(x, feat3)
        x = self.up2(x, feat2)
        x = self.up1(x, feat1)

        x = self.out_conv(x)

        return x
    

if __name__ == "__main__":
    model = PetSegModel()
    model.to(DEVICE)

    train_dataset = PetImages_Dataset()
    train_dataloader = DataLoader(train_dataset, batch_size = TRAIN_BATCH_SIZE, shuffle=False, num_workers=0)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(EPOCH_SIZE):
        for imgs, masks in train_dataloader:
            imgs = imgs.to(DEVICE) # [B 3 H W]
            masks = masks.to(DEVICE) # [B 1 H W]
            optimizer.zero_grad()

            output = model(imgs)
            loss = criterion(output, masks)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), TRAINED_DATA_PATH)