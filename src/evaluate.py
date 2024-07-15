import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
sys.path.append('C:\\Users\\dell\\OneDrive\\Desktop\\VQA')
from model.vqa_model import VQAModel
from dataset import VQADataset

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, questions, answers in dataloader:
            questions = torch.zeros(images.size(0), 10, 50)
            answers = torch.randint(0, 10, (images.size(0),))

            outputs = model(images, questions)
            _, predicted = torch.max(outputs.data, 1)
            total += answers.size(0)
            correct += (predicted == answers).sum().item()
    
    print(f'Accuracy: {100 * correct / total}')

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    dataset = VQADataset('data/dataset.json', transform=transform)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    
    model = VQAModel()
    evaluate(model, dataloader)
