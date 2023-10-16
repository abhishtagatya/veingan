import glob
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

EVALUATE_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to a common size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])


class AnomalyImageDataset(Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    @classmethod
    def load_from_dir(cls, data_dir, transform=None):
        dir_list = glob.glob(data_dir)
        return cls(data=dir_list, transform=transform)

    def to_dataloader(self, batch_size: int = 1, shuffle: bool = True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
