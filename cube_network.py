import torch
import torch.nn as nn
import numpy as np

class RubiksDQNRefined(nn.Module):
    def __init__(self):
        super(RubiksDQNRefined, self).__init__()
        
        # Flattened Input: 6 sides * 9 stickers * 6 one-hot color choices = 324
        self.input_layer = nn.Sequential(
            nn.Linear(324, 2048),
            nn.ELU(),
            nn.BatchNorm1d(2048)
        )
        
        # Fully Connected Residual-style Layers for combinatorial processing
        self.hidden_blocks = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ELU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.ELU()
        )
        
        # Final output scalar: Expected steps left to solve the state.
        self.value_output = nn.Linear(512, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_blocks(x)
        return self.value_output(x)

def encode_cube_state(cube_instance):
    """Converts side-based letters to a 324-long tensor for PyTorch."""
    color_map = {col: idx for idx, col in enumerate(cube_instance.COLORS)}
    flattened_numerical = []
    
    for side in cube_instance.SIDES:
        for row in cube_instance.sides[side]:
            for sticker in row:
                flattened_numerical.append(color_map[sticker])
                
    # Create One-Hot matrix
    one_hot = np.eye(6)[flattened_numerical] # Shape (54, 6)
    return torch.tensor(one_hot.flatten(), dtype=torch.float32).unsqueeze(0)