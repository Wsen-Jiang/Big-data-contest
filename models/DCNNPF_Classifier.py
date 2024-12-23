

import torch
import torch.nn as nn
import torch.nn.functional as F

class DCNNPF_Classifier(nn.Module):
    def __init__(self, channels=2, num_classes=10, dropout=0.5):
        super(DCNNPF_Classifier, self).__init__()

        self.input_shape = [256]
        self.classes = num_classes
        self.dr = dropout  # Dropout rate
        
        # Conv1D layers for input1
        self.conv1_1 = nn.Conv1d(1, 32, 3, padding=1)  # First Conv1d layer for input1
        self.conv1_2 = nn.Conv1d(32, 64, 3, padding=1)  # Second Conv1d layer for input1
        self.conv1_3 = nn.Conv1d(64, 64, 3, padding=1)  # Third Conv1d layer for input1


        # Conv1D layers for input2
        self.conv2_1 = nn.Conv1d(1, 32, 3, padding=1)  # First Conv1d layer for input2
        self.conv2_2 = nn.Conv1d(32, 64, 3, padding=1)  # Second Conv1d layer for input2
        self.conv2_3 = nn.Conv1d(64, 64, 3, padding=1)  # Third Conv1d layer for input2


        # Conv1D layers for the concatenated input
        self.conv3_1 = nn.Conv1d(128, 64, 3, padding=1)  # Conv1D layer after concatenation
        self.conv3_2 = nn.Conv1d(64, 32, 3, padding=1)
        self.conv3_3 = nn.Conv1d(32, 32, 3, padding=1)

        self.lstm = nn.LSTM(input_size=32, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)
        # MaxPool1D layer
        self.maxpool = nn.MaxPool1d(2)

        # Fully connected layers
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, self.classes)

        # Dropout layer
        self.dropout = nn.Dropout(p=self.dr)

    def forward(self, x):
        input1 = x[:,0,:]
        input2 = x[:,1,:]
        input1 = input1.unsqueeze(1)  # Add channel dimension (batch_size, 1, length)
        input2 = input2.unsqueeze(1)  # Add channel dimension (batch_size, 1, length)
        
        # Process input1 through its Conv1D layers
        x2 = F.relu(self.conv1_1(input1))
        x2 = self.dropout(x2)
        x2 = F.relu(self.conv1_2(x2))
        x2 = self.dropout(x2)
        x2 = F.relu(self.conv1_3(x2))
        x2 = self.dropout(x2)

        # Process input2 through its Conv1D layers
        x3 = F.relu(self.conv2_1(input2))
        x3 = self.dropout(x3)
        x3 = F.relu(self.conv2_2(x3))
        x3 = self.dropout(x3)
        x3 = F.relu(self.conv2_3(x3))
        x3 = self.dropout(x3)


        # Concatenate the outputs of input1 and input2
        x = torch.cat([x2, x3], dim=1)  # Concatenate along channel dimension

        # Apply more convolutions and pooling layers
        x = F.relu(self.conv3_1(x))
        x = self.dropout(x)
        x = self.maxpool(x)
        
        x = F.relu(self.conv3_2(x))
        x = self.dropout(x)
        x = self.maxpool(x)
        
        x = F.relu(self.conv3_3(x))
        x = self.dropout(x)
        x = self.maxpool(x)

        # 忽视长度
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        x = self.dropout(x)

        # Fully connected layers
        x = F.selu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x