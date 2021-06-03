import torch
import torch.nn as nn


class LinearClassification(nn.Module):
    def __init__(self, input_features = 256, num_classes = 17):
        super(LinearClassification, self).__init__()
        self.input_features = input_features
        self.num_classes = num_classes
        self.fc = nn.Linear(input_features, num_classes)

    def forward(self, x):
        x = self.fc(x)

        return x


if __name__ == '__main__':

    import torch
    model = LinearClassification().cuda()
    data = torch.rand(10, 256).cuda()
    out = model(data)
