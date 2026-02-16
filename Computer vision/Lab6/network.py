import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size
    ):
        super().__init__()
        
        self.relu = nn.ReLU()
        self.residual_block = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=1,
                padding=1, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels, 
                out_channels, 
                kernel_size=kernel_size,
                stride=1, 
                padding=1, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=1, 
                stride=1,
                padding=0, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
         ) if in_channels != out_channels else None

    def forward(self, image):
        out = self.residual_block(image)
        residual = image if self.shortcut is None else self.shortcut(image)

        return self.relu(out + residual)


class ResNet(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 number_of_blocks,
                 number_of_classes
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels=out_channels[0], 
                kernel_size=7, 
                stride=1,
                padding=1, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU()
        )

        layers = []
        for i in range(number_of_blocks):
            layers += [
                ResidualBlock(
                    in_channels=out_channels[i], 
                    out_channels=out_channels[i + 1],
                    kernel_size=3
                )
            ]

        self.residual_blocks = nn.Sequential(*layers)
        self.avg_pool = nn.AvgPool2d(kernel_size=3)
        self.linear = nn.Linear(out_channels[-1] * 20 * 20, number_of_classes)

    def forward(self, image):
        out = self.conv(image)
        out = self.residual_blocks(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)

        return self.linear(out)