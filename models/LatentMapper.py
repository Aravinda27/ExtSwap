import torch.nn as nn
import numpy as np
import torch
from Models.StyleGan2.model import EqualLinear
import einops

class LatentMapper(nn.Module):
    def __init__(self):
        super().__init__()
        slope = 0.2
        self.model = nn.Sequential(
            nn.Linear(2304, 2048),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=slope),
            nn.Linear(512, 512)
        )
        self.relu = nn.LeakyReLU(0.2)
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=slope)
                nn.init.constant_(m.bias, 0)
        self.num_styles = int(np.log2(1024)) * 2 - 2
        self.linear = EqualLinear(512, 512 * self.num_styles, lr_mul=1)

    def forward(self, input_tensor):
        first = True
        for layer in self.model:
            if not first:
                input_tensor = self.relu(input_tensor)
            input_tensor = layer(input_tensor.float())
            first = False

        # Apply ReLU activation to the input tensor
        x = self.relu(input_tensor)

        # Reshape x to introduce a new dimension at the end, repeated self.num_styles times
        x = x.unsqueeze(dim=-1)
        x = x.repeat(1, 1, self.num_styles)

        # Swap axes of x
        x = x.permute(0, 2, 1)

        # Broadcast x to have the desired shape
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)  # Add a new dimension at index 1
        x = x.repeat(1, 18, 1, 1)  # Repeat 18 times along the new dimension

        return x

# Instantiate the LatentMapper class
#mapper = LatentMapper()

# Example usage:
# Assuming input_tensor has shape (batch_size, 2304)
#input_tensor = torch.randn(4, 2304)
#output_tensor = mapper(input_tensor)
#print("Output shape:", output_tensor.shape)

