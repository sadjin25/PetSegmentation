import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import cv2
import glob
import os
from tqdm import tqdm

IMG_PATH = "./Datas/images/images/"
MASK_PATH = "./Datas/annotations/trimaps/"
IMG_SIZE = (384, 384)
EPOCH_SIZE = 5
TRAIN_BATCH_SIZE = 4
TRAINED_DATA_PATH = "./Datas/trained_data/pet_seg.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

TEST_BATCH_SIZE = 4
TEST_SEED = 245


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

        mask = GetResizedImage(mask, isMask=True)
        mask = (mask != 2)
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

def DoTrain(model):
    model.train()

    train_dataset = PetImages_Dataset()
    train_dataloader = DataLoader(train_dataset, batch_size = TRAIN_BATCH_SIZE, 
                                  shuffle=True, num_workers=4, persistent_workers=True,
                                  pin_memory=(DEVICE.type == "cuda"))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(EPOCH_SIZE):
        running_loss = 0.0
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{EPOCH_SIZE}",
            unit="batch"
        )

        for step, (imgs, masks) in enumerate(progress_bar, start=1):
            imgs = imgs.to(DEVICE, non_blocking=True) # [B 3 H W]
            masks = masks.to(DEVICE, non_blocking=True) # [B 1 H W]
            optimizer.zero_grad()

            output = model(imgs)
            loss = criterion(output, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{running_loss / step:.4f}"
            )

    torch.save(model.state_dict(), TRAINED_DATA_PATH)

@torch.no_grad()
def DoTest(model):
    torch.manual_seed(TEST_SEED)
    torch.cuda.manual_seed_all(TEST_SEED)
    g = torch.Generator()
    g.manual_seed(TEST_SEED)

    model.load_state_dict(torch.load(TRAINED_DATA_PATH, map_location=DEVICE))
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    test_dataset = PetImages_Dataset()
    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, 
                                    shuffle=False, num_workers=4, persistent_workers=True,
                                    pin_memory=(DEVICE.type == "cuda"))

    total_loss = 0.0
    pet_miou = 0.0
    dice = 0.0
    total_imgs = 0

    for imgs, masks in tqdm(test_dataloader, desc="Test", unit="batch"):
        imgs = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        total_imgs += imgs.size(0)

        logits = model(imgs)
        loss = criterion(logits, masks)
        total_loss += loss.item() * imgs.size(0) # BCEWithLogitsLoss() returns meanloss over batch.

        preds = (torch.sigmoid(logits) >= 0.5).long()
        masks = masks.long()

        intersection_per_img = (preds & masks).sum(dim=(1,2,3)) # [B]
        union_per_img = (preds | masks).sum(dim=(1,2,3)) # [B]
        
        iou_per_img = intersection_per_img / union_per_img.clamp_min(1) # [B]
        pet_miou += iou_per_img.sum().item()
    
        # shape for dice cal : [B]
        dice_intersection = (preds & masks).sum(dim=(1,2,3)) # [B,1,H,W]
        dice_denominator = preds.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3)) # [B,1,H,W]
        dice_batch = ((2 * dice_intersection + 1e-5) / (dice_denominator + 1e-5))

        dice += dice_batch.sum().item()

    print(f"Test Loss: {total_loss / total_imgs:.4f}")
    print(f"Mean Pet IoU : {pet_miou / total_imgs:.4f}")
    print(f"Mean Dice : {dice / total_imgs:.4f}")

# MAIN---------------------

if __name__ == "__main__":
    model = PetSegModel()
    model.to(DEVICE)

# TRAIN
    # DoTrain(model)

# TEST
    DoTest(model)



