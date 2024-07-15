import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
sys.path.append('C:\\Users\\dell\\OneDrive\\Desktop\\VQA')
from model.vqa_model import VQAModel
from dataset import VQADataset

class SelfSupervisedTask(nn.Module):
    def __init__(self, cnn):
        super(SelfSupervisedTask, self).__init__()
        self.cnn = cnn
        self.fc = nn.Linear(32 * 16 * 16, 2)  # Output for color and shape

    def forward(self, image):
        features = self.cnn(image)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        return output

def pretrain(model, dataloader, num_epochs=8):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    
    for epoch in range(num_epochs):
        for images, _, _ in dataloader:
            colors = torch.randint(0, 3, (images.size(0),))
            shapes = torch.randint(0, 2, (images.size(0),))
            labels = torch.stack((colors, shapes), dim=1).float()

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    dataset = VQADataset('data/dataset.json', transform=transform)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    model = VQAModel()
    pretrain_model = SelfSupervisedTask(model.cnn)
    pretrain(pretrain_model, dataloader)
