import torch
import torch.nn as nn


class YOLO(nn.Module):
    def __init__(self, split_size, num_boxes, num_classes):
        super(ModifiedYOLO, self).__init__()
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.darkNet = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3, stride=2, bias=False),  # 3,448,448 -> 64,224,224
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # -> 64,112,112

            nn.Conv2d(64, 128, 3, padding=1, bias=False),  # -> 128,112,112
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # -> 128,56,56

            nn.Conv2d(128, 256, 1, bias=False),  # -> 256,56,56
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),  # -> 256,56,56
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # -> 256,28,28

            nn.Conv2d(256, 512, 3, padding=1, bias=False),  # -> 512,28,28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),  # -> 512,28,28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # -> 512,14,14

            nn.Conv2d(512, 1024, 3, padding=1, bias=False),  # -> 1024,14,14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False),  # -> 1024,14,14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False),  # -> 1024,14,14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 2048, 3, padding=1, bias=False),  # -> 2048,14,14
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(2048, 2048, 3, padding=1, bias=False),  # -> 2048,14,14
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)  # -> 2048,7,7
        )
        self.fc = nn.Sequential(
            nn.Linear(2048 * split_size * split_size, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, split_size * split_size * (num_classes + num_boxes * 5)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.darkNet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = x.view(x.shape[0], self.split_size, self.split_size,
                   self.num_boxes * 5 + self.num_classes)
        return x

# Example usage
model = YOLO(split_size=7, num_boxes=2, num_classes=20)
print(model)
