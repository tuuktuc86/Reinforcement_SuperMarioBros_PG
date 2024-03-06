import torch
import matplotlib.pyplot as plt
from torch import nn

import numpy as np
import seaborn as sns
class MarioNet(nn.Module):
    '''mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")


        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*9*9, 512),  # 정확한 출력 크기 계산
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )



    def forward(self, input, model):
        if model == 'online':
            #print("start model")
            #print(input.shape)
            output = self.online(input)
            #print("end model")
            #print(output)
            return F.softmax(output, dim = 1)
        elif model == 'target':
            return self.target(input)

model = MarioNet((4,84,84),2)
model = torch.load("save_dir/mario_net_31500.chkpt")
a = []
b=[]
c=[]
layer = 2
def visualize_model_parameters_heatmap(model):
    for name, param in model.named_parameters():
        if 'weight' in name:  # 가중치 파라미터에 대해서만 시각화
            plt.figure(figsize=(10, 10))
            param_data = param.cpu().detach().numpy()
            ax = sns.heatmap(param_data, cmap='viridis')
            plt.title(name)
            plt.show()

visualize_model_parameters_heatmap(model)
