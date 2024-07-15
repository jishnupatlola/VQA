import torch
import torch.nn as nn

class VQAModel(nn.Module):
    def __init__(self):
        super(VQAModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(input_size=50, hidden_size=100, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32 * 16 * 16 + 100, 10)

    def forward(self, image, question):
        image_features = self.cnn(image)
        image_features = image_features.view(image_features.size(0), -1)
        question_features, _ = self.lstm(question)
        question_features = question_features[:, -1, :]
        combined_features = torch.cat((image_features, question_features), dim=1)
        output = self.fc(combined_features)
        return output
