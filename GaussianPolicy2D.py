import torch
import torch.nn as nn
import itertools


# Define gaussian policy with mean and variance
class GaussianPolicy2D(nn.Module):
    def __init__(self, n_gauss_0, n_gauss_1):
        super(GaussianPolicy2D, self).__init__()

        self.n_gauss_0 = n_gauss_0
        self.n_gauss_1 = n_gauss_1

        self.class_0 = nn.ModuleList([nn.Linear(1, 4, bias=False) for _ in range(n_gauss_0)])
        self.class_1 = nn.ModuleList([nn.Linear(1, 4, bias=False) for _ in range(n_gauss_1)])

        self.action_log_std_0 = [nn.Parameter(torch.ones(1, 4)) * 0.05 for _ in range(n_gauss_0)]
        self.action_log_std_1 = [nn.Parameter(torch.ones(1, 4)) * 0.05 for _ in range(n_gauss_1)]

        self.train()
        self.init_weights()

    def init_weights(self):

        for layer in itertools.chain(self.class_0, self.class_1):
            layer.weight.data[0] = 0
            layer.weight.data[1] = 0
            layer.weight.data[2] = 1
            layer.weight.data[3] = 1

    def forward(self):
        fixed_input = torch.ones(1).view(1, 1)
        val_0 = []
        val_1 = []
        std_0 = []
        std_1 = []

        for i, layer in enumerate(self.class_0):
            tmp_mean = layer(fixed_input)
            tmp_std = self.action_log_std_0[i].expand_as(tmp_mean)
            val_0.append(tmp_mean)
            std_0.append(tmp_std)
        for i, layer in enumerate(self.class_1):
            tmp_mean = layer(fixed_input)
            tmp_std = self.action_log_std_1[i].expand_as(tmp_mean)
            val_1.append(tmp_mean)
            std_1.append(tmp_std)

        return val_0, val_1, std_0, std_1

    def get_params(self, data):
        data = torch.cat(data, dim=0)
        mean = torch.zeros([data.size(0), 2])
        cov = torch.zeros([data.size(0), 2, 2])
        mean[:, 0] = data[:, 0]
        mean[:, 1] = data[:, 1]
        cov[:, 0, 0] = data[:, 2]
        cov[:, 0, 1] = 0.0
        cov[:, 1, 0] = 0.0
        cov[:, 1, 1] = data[:, 3]
        return mean, cov

    def act(self):
        val_0, val_1, std_0, std_1 = self()

        normals_0 = [torch.distributions.Normal(m, s) for m, s in zip(val_0, std_0)]
        actions_0 = [d.sample() for d in normals_0]
        action_log_probs_tmp = [d.log_prob(action) for d, action in zip(normals_0, actions_0)]
        action_log_probs_0 = [torch.sum(logprob, dim=1, keepdim=True) for logprob in action_log_probs_tmp]

        normals_1 = [torch.distributions.Normal(m, s) for m, s in zip(val_1, std_1)]
        actions_1 = [d.sample() for d in normals_1]
        action_log_probs_tmp = [d.log_prob(action) for d, action in zip(normals_1, actions_1)]
        action_log_probs_1 = [torch.sum(logprob, dim=1, keepdim=True) for logprob in action_log_probs_tmp]

        # Final log probs is the sum of the log probs
        action_log_probs = torch.zeros(1)
        for logprob in itertools.chain(action_log_probs_0, action_log_probs_1):
            action_log_probs += logprob.squeeze()

        means_0, cov_0 = self.get_params(actions_0)
        means_1, cov_1 = self.get_params(actions_1)

        return [[means_0, means_1],
                [[[cov_0[0][0, 0], cov_0[0][1, 1]], [cov_0[1][0, 0], cov_0[1][1, 1]]], [[cov_1[0][0, 0], cov_1[0][1, 1]],
                 [cov_1[1][0, 0], cov_1[1][1, 1]]]]], action_log_probs
