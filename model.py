import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

dropout_value = 0.2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


       # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=10, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(dropout_value),
            # Dilated Conv layer
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), dilation=2, padding=1, bias=False)
        )

        # CONVOLUTION BLOCK 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 2), stride =2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, bias=False)
        )

        # self.pool1 = nn.MaxPool2d(2, 2)

        # CONVOLUTION BLOCK 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Dropout(dropout_value),
            nn.Conv2d(96, 64, 2, stride=2, padding=1, bias=False),
  
        )

          # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            
            # Depthwise Conv layer
            nn.Conv2d(64, 64, 3, groups = 64 ,padding=1, bias=False),
            # nn.ReLU(),
            # nn.BatchNorm2d(64),
            # nn.Dropout(dropout_value),
            # Pointwise conv layer
            nn.Conv2d(64, 16, 1, padding=1, bias=False),
  
            nn.AvgPool2d(kernel_size=9),
            nn.Conv2d(16, 10, 1, padding=0, bias=False))

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Net_10(nn.Module):
    def __init__(self):
      super(Net_10, self).__init__()


       # Prep Layer
      self.convblock1 = nn.Sequential( nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU()
                                       )

      # CONVOLUTION BLOCK 1
      self.convblock2 = nn.Sequential( nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU()
        )

      self.R1 = nn.Sequential(nn.Conv2d (in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                              nn.Conv2d(in_channels=128,  out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
                              nn.BatchNorm2d(128),
                              nn.ReLU())



      # CONVOLUTION BLOCK 2
      self.convblock3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU()
                                      )

      # CONVOLUTION BLOCK 3
      self.convblock4 = nn.Sequential(nn.Conv2d(256, 512, 3,padding=1, bias=False),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU()
            )

      self.R2  = nn.Sequential(nn.Conv2d (in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
                               nn.BatchNorm2d(512),
                               nn.ReLU(),
                               nn.Conv2d (in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
                               nn.BatchNorm2d(512),
                               nn.ReLU()
            )

      self.maxpool = nn.MaxPool2d(kernel_size=4)

      self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
      self.fc = nn.Linear(512, 10)



    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x= self.R1(x) + x
        x = self.convblock3(x)
        x = self.convblock4(x)
        x= self.R2(x) + x
        x = self.maxpool(x)
        x = self.global_pool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return F.log_softmax(x, dim=-1)








