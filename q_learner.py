import os

import torch
from torch import nn

INPUT_DIM = 32
HIDDEN_LAYERS = (64,128)
ACTION_OUTPUT_DIM = 4
PATH_TO_WEIGHTS = "weights/q_learner_latest_weights.pt"

class Q_Learner(nn.Module):
    def __init__(self,input_dim=INPUT_DIM,hidden_layers=HIDDEN_LAYERS,action_output_dim=ACTION_OUTPUT_DIM):
        """
        Parameters
        ----------
        input_dim: integer
        Length of the input (flattened).

        hidden_layers: tuple/list
        List-like object with number of neurons per layer (length of this list == number of linear layers+1).

        action_output_dim: integer
        Number of output units - corresponds to action space of the agent.
        """
        
        super().__init__()
        self.deep_q_layers = nn.Sequential(
            nn.Linear(input_dim,hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0],hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1],action_output_dim)
        )

    def forward(self,X):
        X = torch.flatten(X,start_dim=1)
        return( self.deep_q_layers(X) )

    def save_params(self,path=PATH_TO_WEIGHTS,silent=False):
        if silent != True:
            print("\n... Saving weights for the Q-learner ...\n")
        torch.save(self.state_dict(),path)

    def load_pretrained_w(self,path_to_weights,silent=False):
        if os.path.isfile(path_to_weights):
            if silent != True:
                print("\n... Loading pretrained weights for the Q-learner ...\n")
            self.load_state_dict(torch.load(path_to_weights))
        else:
            raise ValueError("Weights file on this path not found.")