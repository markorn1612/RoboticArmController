class Mlp(Module):
    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size
    ):
        super().__init__()
        # TODO: initialization
        self.fcs = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = Linear(in_size, next_size)
            self.add_module(f'fc{i}', fc)
            self.fcs.append(fc)
            in_size = next_size
        self.last_fc = Linear(in_size, output_size)

    def forward(self, input):
        h = input
        for fc in self.fcs:
            h = relu(fc(h))
        output = self.last_fc(h)
        return output

class Critic(Module):
    def __init__(self, state_dim, action_dim, n_quantiles, n_nets):
        super().__init__()
        self.nets = []
        self.n_quantiles = n_quantiles
        self.n_nets = n_nets
        for i in range(n_nets):
            net = Mlp(state_dim + action_dim, [512, 512, 512], n_quantiles)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, state, action):
        sa = torch.cat((state, action), dim=1)
        quantiles = torch.stack(tuple(net(sa) for net in self.nets), dim=1)
        return quantiles

class Actor(Module):
  def __init__(self, state_dim, action_dim):
      super().__init__()
      self.action_dim = action_dim
      self.net = Mlp(state_dim, [256, 256], 2 * action_dim)

  def forward(self, obs):
      mean, log_std = self.net(obs).split([self.action_dim, self.action_dim], dim=1)
      log_std = log_std.clamp(*LOG_STD_MIN_MAX)

      if self.training:
          std = torch.exp(log_std)
          tanh_normal = TanhNormal(mean, std)
          action, pre_tanh = tanh_normal.rsample()
          log_prob = tanh_normal.log_prob(pre_tanh)
          log_prob = log_prob.sum(dim=1, keepdim=True)
      else:  # deterministic eval without log_prob computation
          action = torch.tanh(mean)
          log_prob = None
      return action, log_prob

  def select_action(self, obs):
      obs = torch.FloatTensor(obs).to(DEVICE)[None, :]
      action, _ = self.forward(obs)
      action = action[0].cpu().detach().numpy()
      return action
