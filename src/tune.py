import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
sys.path.append('C:\\Users\\dell\\OneDrive\\Desktop\\VQA')
from model.vqa_model import VQAModel
from dataset import VQADataset

def fine_tune(model, dataloader, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for images, questions, answers in dataloader:
            questions = torch.zeros(images.size(0), 10, 500)
            answers = torch.randint(0, 100, (images.size(0),))

            outputs = model(images, questions)
            loss = criterion(outputs, answers)
            
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
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    
    model = VQAModel()
    fine_tune(model, dataloader)
