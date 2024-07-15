import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class VQADataset(Dataset):
    def __init__(self, data_file, transform=None):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image'])
        if self.transform:
            image = self.transform(image)
        question = item['question']
        answer = item['answer']
        return image, question, answer
