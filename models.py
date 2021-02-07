## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        ## After each convolutional layer, output size = (W-F)/S+1
        # K - out_channels : the number of filters in the convolutional layer
        # F - kernel_size
        # S - the stride of the convolution (default=1)
        # P - the padding
        # W - the width/height (square) of the previous layer
        
        # Every layer = Conv2d + Relu + Maxpool
        # Assume that input size decreases by two times after maxpool 
        
        # input size  = 224 x 224
        # output size = (W-F)/S +1 = (224-5)/1 +1 = 220 x 220
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        
        # input size  = 110 x 110
        # output size = (W-F)/S+1 = (110-3)/1+1 = 108
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # input size  = 54 x 54
        # output size = (W-F)/S+1 = (108-3)/1+1 = 52
        self.conv3 = nn.Conv2d(64, 128, 3)

        # input size  = 26 x 26
        # output size = (W-F)/S+1 = (26-3)/1+1 = 12
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        # input size  = 6 x 6
        # output size = (W-F)/S+1 = (6-1)/1+1 = 6
        self.conv5 = nn.Conv2d(256, 512, 1)
 
        # maxpool layer
        # pool with kernel_size=2, stride=2
        # output size = (input size)/2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully-connected (linear) layers
        self.fc1 = nn.Linear(512*6*6, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 68*2)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.3)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # 5 layers = conv + relu + pool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        
        # Flatten = 512*6*6
        x = x.view(x.size(0), -1)
        
        # Fully-connected layer
        x = F.relu(self.fc1(x))
        # Dropout with 0.3
        x = self.dropout(x)
        # Fully-connected layer
        x = F.relu(self.fc2(x))
        # Dropout with 0.3
        x = self.dropout(x)
        # Fully-connected layer with output 136
        x = self.fc3(x)
        
        # final output
        return x
