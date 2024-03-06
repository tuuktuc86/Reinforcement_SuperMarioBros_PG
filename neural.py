from torch import nn
import copy
import torch.nn.functional as F
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
