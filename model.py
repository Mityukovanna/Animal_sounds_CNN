import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
# import torchvision
# import torchvision.transforms as transforms
"""
THIS IS MY ATTEMPT TO OPTIMIZE THE CNN
I'VE ADDED:
1. ONE MORE CONV LAYER
2. DROPOUT LAYER (MORE INFO: https://www.reddit.com/r/learnmachinelearning/comments/x89qsi/dropout_in_neural_networks_what_it_is_and_how_it/)
3. ADAOTIVE POOLING LAYER (TO FIX THE SIZE OF THE OUTPUT)
3. DELETED SOFTMAX PREDICTONS
"""

# FORMULA TO CALCULATE THE OUTPUT SIZE
# (width size - filter size + 2*padding)/stride + 1
# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# link to the tutorial: https://www.youtube.com/watch?v=SQ1iIKs190Q
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # first conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=16, 
                kernel_size=3, 
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # second conv layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, 
                kernel_size=3, 
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # third conv layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, 
                out_channels=64, 
                kernel_size=3, 
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # 4th conv layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, 
                out_channels=128, 
                kernel_size=3, 
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, 
                out_channels=256, 
                kernel_size=3, 
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Adaptive Pooling to reduce dependency on input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # flatten and linear layers
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(12544, 5, 5) 

        self.dropout = nn.Dropout(0.5)  # Dropout layer

    def forward(self, input_data):
        # passing the data through layers
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)       
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.adaptive_pool(x)  # Apply adaptive pooling
        x = self.flatten(x) # flatten layer
        x = self.dropout(x) # dropout layer
        #print(x.shape) <-- lifehack from Anyutka: get the shape and then use this value to pass to the linear layer
        logits = self.linear(x)
        return logits


def main():
    cnn = ConvNet().to(device)
    summary(cnn, (1, 64, 157))


if __name__ == "__main__":
    main()

