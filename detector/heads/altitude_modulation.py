from torch import nn

class AltitudeModulation(nn.Module):
    def __init__(self, num_features):
        super(AltitudeModulation, self).__init__()
        self.num_features = num_features
        self.condition_fc = nn.Linear(1, num_features * 2)
        self.condition_fc.weight.data[:, :num_features].fill_(1)  # Initialize scaling (gamma) weights to 1
        self.condition_fc.weight.data[:, num_features:].zero_()  # Initialize shifting (beta) weights to 0

    def forward(self, x, condition):
        gamma, beta = self.condition_fc(condition).chunk(2, dim=-1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        return gamma * x + beta