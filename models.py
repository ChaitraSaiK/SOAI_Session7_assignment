from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Model_1_Simple_CNN_Architecture(nn.Module):
    def __init__(self):
        super(Model_1_Simple_CNN_Architecture, self).__init__()
        
        # Input Block - RF: 3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),  # RF: 3
            nn.ReLU(),
        )

        # Conv Block 1 - RF: 5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1, bias=False),  # RF: 5
            nn.ReLU(),
        )

        # Transition Block 1 - RF: 6
        self.pool1 = nn.MaxPool2d(2, 2)  # RF: 6
        self.convblock3 = nn.Sequential(
            nn.Conv2d(8, 12, 1, bias=False),  # RF: 6
            nn.ReLU(),
        )

        # Conv Block 2 - RF: 14
        self.convblock4 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 10
            nn.ReLU(),
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 14
            nn.ReLU(),
        )

        # Transition Block 2 - RF: 16
        self.pool2 = nn.MaxPool2d(2, 2)  # RF: 16
        self.convblock5 = nn.Sequential(
            nn.Conv2d(12, 12, 1, bias=False),  # RF: 16
            nn.ReLU(),
        )

        # Conv Block 3 - RF: 28
        self.convblock6 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 20
            nn.ReLU(),
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 24
            nn.ReLU(),
            nn.Conv2d(12, 10, 3, padding=1, bias=False),  # RF: 28
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    



class Model_2_BatchNorm(nn.Module):
    def __init__(self):
        super(Model_2_BatchNorm, self).__init__()

        
        # Input Block - RF: 3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),  # RF: 3
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        # Conv Block 1 - RF: 5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1, bias=False),  # RF: 5
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        # Transition Block 1 - RF: 6
        self.pool1 = nn.MaxPool2d(2, 2)  # RF: 6
        self.convblock3 = nn.Sequential(
            nn.Conv2d(8, 12, 1, bias=False),  # RF: 6
            nn.BatchNorm2d(12),
            nn.ReLU(),
        )

        # Conv Block 2 - RF: 14
        self.convblock4 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 10
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 14
            nn.BatchNorm2d(12),
            nn.ReLU(),
        )

        # Transition Block 2 - RF: 16
        self.pool2 = nn.MaxPool2d(2, 2)  # RF: 16
        self.convblock5 = nn.Sequential(
            nn.Conv2d(12, 12, 1, bias=False),  # RF: 16
            nn.BatchNorm2d(12),
            nn.ReLU(),
        )

        # Conv Block 3 - RF: 28
        self.convblock6 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 20
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 24
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 10, 3, padding=1, bias=False),  # RF: 28
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    



class Model_3_Dropout(nn.Module):
    def __init__(self):
        super(Model_3_Dropout, self).__init__()
        dropout_value = 0.05
        
        # Input Block - RF: 3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),  # RF: 3
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        # Conv Block 1 - RF: 5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1, bias=False),  # RF: 5
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        # Transition Block 1 - RF: 6
        self.pool1 = nn.MaxPool2d(2, 2)  # RF: 6
        self.convblock3 = nn.Sequential(
            nn.Conv2d(8, 12, 1, bias=False),  # RF: 6
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        # Conv Block 2 - RF: 14
        self.convblock4 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 10
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 14
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        # Transition Block 2 - RF: 16
        self.pool2 = nn.MaxPool2d(2, 2)  # RF: 16
        self.convblock5 = nn.Sequential(
            nn.Conv2d(12, 12, 1, bias=False),  # RF: 16
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        # Conv Block 3 - RF: 28
        self.convblock6 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 20
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 24
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.Conv2d(12, 10, 3, padding=1, bias=False),  # RF: 28
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    


 

class Model_4_augmnetation(nn.Module):
    def __init__(self):
        super(Model_4_augmnetation, self).__init__()
        dropout_value = 0.05
        
        # Input Block - RF: 3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),  # RF: 3
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        # Conv Block 1 - RF: 5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1, bias=False),  # RF: 5
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        # Transition Block 1 - RF: 6
        self.pool1 = nn.MaxPool2d(2, 2)  # RF: 6
        self.convblock3 = nn.Sequential(
            nn.Conv2d(8, 12, 1, bias=False),  # RF: 6
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        # Conv Block 2 - RF: 14
        self.convblock4 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 10
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 14
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        # Transition Block 2 - RF: 16
        self.pool2 = nn.MaxPool2d(2, 2)  # RF: 16
        self.convblock5 = nn.Sequential(
            nn.Conv2d(12, 12, 1, bias=False),  # RF: 16
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        # Conv Block 3 - RF: 28
        self.convblock6 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 20
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 24
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.Conv2d(12, 10, 3, padding=1, bias=False),  # RF: 28
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    

       

      
    

       


    

       


    

       

