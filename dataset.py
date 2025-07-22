import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class DefectDataset(Dataset):
    def __init__(self, img_dir, size=64):
        self.files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img)
