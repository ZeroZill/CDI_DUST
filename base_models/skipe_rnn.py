import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from base_models.auto_encoder import AutoencoderModel
from utils import check_array, matrix_2_norm, Statistics, MostRecentCache, NormalCache


class SkipERNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1, eta=0.5, activate="hyperplane"):
        super(SkipERNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.eta = eta
        self.n_layers = 3
        self.U = nn.Linear(input_size, hidden_size)
        self.V = nn.Linear(hidden_size, output_size)

        # Initialize parameters
        init.normal_(self.U.weight, mean=0.0, std=1.0 / input_size ** 0.5)
        # init.xavier_uniform_(self.U.weight, gain=4.0)
        # init.constant_(self.U.weight, 1.0)
        init.constant_(self.U.bias, 0.0)

        init.xavier_uniform_(self.V.weight, gain=4.0)
        # init.constant_(self.V.weight, 1.0)
        init.constant_(self.V.bias, 0.0)

        self.activate = activate

        self.x_stats = Statistics()
        self.NS_stats = Statistics()
        self.NHS_stats = Statistics()

        self.grow, self.prune = False, False
        self.min_NS_mean, self.min_NS_std = torch.inf, torch.inf
        self.min_NHS_mean, self.min_NHS_std = torch.inf, torch.inf

    def hyperplane_activation(self, x, y_kt):
        a = self.U(x)
        up = torch.sum(y_kt) - self.output_size * a
        down = torch.sqrt(1 + torch.sum(self.U.weight ** 2) + self.U.bias ** 2)
        d = up / down
        h = torch.exp(-self.eta * d / torch.max(torch.abs(d)))
        return h

    def hyperplane_bp(self, x, o, h):
        x = check_array(x)
        dL_do = o.grad
        dL_dh_w = dL_do @ self.V.weight
        tmp = self.eta * h / torch.max(h)
        dL_dU_w = (tmp * dL_dh_w).T @ x
        dL_dU_b = (tmp * dL_dh_w).view(-1)
        self.U.weight.grad.data = dL_dU_w
        self.U.bias.grad.data = dL_dU_b

    def forward(self, x, y_kt=None):
        x = check_array(x)
        assert x.shape == (1, self.input_size), \
            f"Invalid `x` shape. Expected (1, {self.input_size}), but got {x.shape}"

        if self.activate == "hyperplane":
            y_kt = check_array(y_kt)
            assert y_kt.shape == (1, self.output_size), \
                f"Invalid `y_kt` shape. Expected (1, {self.output_size}), but got {y_kt.shape}"

            h = self.hyperplane_activation(x, y_kt)
        else:
            h = torch.sigmoid(self.U(x))

        # print(h)
        o = self.V(h)
        o.retain_grad()
        y_hat = torch.argmax(o, dim=-1)
        return y_hat, o, h.detach()

    def add_node(self):
        self.hidden_size += 1
        new_U = nn.Linear(self.input_size, self.hidden_size)
        new_V = nn.Linear(self.hidden_size, self.output_size)

        # Initialize parameters
        init.normal_(new_U.weight, mean=0.0, std=1.0 / self.input_size)
        # init.constant_(new_U.weight, 1.0)
        init.constant_(new_U.bias, 0.0)

        init.xavier_uniform_(new_V.weight, gain=4.0)
        # init.constant_(new_V.weight, 1.0)
        init.constant_(new_V.bias, 0.0)

        # Copy original U and V parameters to new layers
        new_U_weight_data = new_U.weight.data.clone()
        new_U_bias_data = new_U.bias.data.clone()
        new_V_weight_data = new_V.weight.data.clone()

        new_U_weight_data[:self.hidden_size - 1, :] = self.U.weight.data
        new_U_bias_data[:self.hidden_size - 1] = self.U.bias.data
        new_V_weight_data[:, :self.hidden_size - 1] = self.V.weight.data

        new_U.weight.data = new_U_weight_data
        new_U.bias.data = new_U_bias_data
        new_V.weight.data = new_V_weight_data

        self.U = new_U
        self.V = new_V

    def delete_node(self, idx):
        self.hidden_size -= 1
        new_U = nn.Linear(self.input_size, self.hidden_size)
        new_V = nn.Linear(self.hidden_size, self.output_size)

        # Copy original U and V parameters to new layers
        new_U_weight_data = new_U.weight.data.clone()
        new_U_bias_data = new_U.bias.data.clone()
        new_V_weight_data = new_V.weight.data.clone()

        new_U_weight_data[:idx, :] = self.U.weight.data[:idx, :]
        new_U_weight_data[idx:, :] = self.U.weight.data[idx + 1:, :]
        new_U_bias_data[:idx] = self.U.bias.data[:idx]
        new_U_bias_data[idx:] = self.U.bias.data[idx + 1:]
        new_V_weight_data[:, :idx] = self.V.weight.data[:, :idx]
        new_V_weight_data[:, idx:] = self.V.weight.data[:, idx + 1:]

        new_U.weight.data = new_U_weight_data
        new_U.bias.data = new_U_bias_data
        new_V.weight.data = new_V_weight_data

        self.U = new_U
        self.V = new_V


class SkipERNNModel:
    def __init__(self, input_size, output_size,
                 hidden_size=1,
                 eta=0.5,
                 activate="hyperplane",
                 criterion=nn.CrossEntropyLoss,
                 optimizer=torch.optim.SGD,
                 lambd=0.001,
                 lr=0.01,
                 momentum=0.95,
                 max_history_cache=2500):
        self.skipe_rnn = SkipERNN(input_size, output_size, hidden_size, eta, activate)
        self.criterion = criterion()
        self.optimizer_class = optimizer
        self.optimizer = optimizer(self.skipe_rnn.parameters(), lr=lr, momentum=momentum)
        self.lr = lr
        self.momentum = momentum
        self.lambd = lambd

        self.hidden_upper_ratio = 0.5 if input_size <= 10 else 0.1

        self.autoencoder = AutoencoderModel(input_size)

        self.history = MostRecentCache(max_history_cache)
        self.pseudo = NormalCache()
        self.states = MostRecentCache(max_history_cache + 100)

        self.n_trained_samples = 0

        self.except_count = 0

    def state_dict(self):
        return copy.deepcopy(self.skipe_rnn.state_dict())

    def apply_penalty(self, another_state):
        self.optimizer.zero_grad(set_to_none=False)
        # Add penalty on grad
        penalty = self.lambd / 2 * \
                  (matrix_2_norm(self.skipe_rnn.U.weight.data) - matrix_2_norm(another_state["U.weight"]))
        self.skipe_rnn.U.weight.grad.data += penalty
        penalty = self.lambd / 2 * \
                  (matrix_2_norm(self.skipe_rnn.U.bias.data) - matrix_2_norm(another_state["U.bias"]))
        self.skipe_rnn.U.bias.grad.data += penalty
        penalty = self.lambd / 2 * \
                  (matrix_2_norm(self.skipe_rnn.V.weight.data) - matrix_2_norm(another_state["V.weight"]))
        self.skipe_rnn.V.weight.grad.data += penalty
        penalty = self.lambd / 2 * \
                  (matrix_2_norm(self.skipe_rnn.V.bias.data) - matrix_2_norm(another_state["V.bias"]))
        self.skipe_rnn.V.bias.grad.data += penalty
        self.optimizer.step()

    def predict(self, x, y_kt):
        self.skipe_rnn.eval()
        y_hat, next_y_kt, _ = self.skipe_rnn(x, y_kt)
        return y_hat, next_y_kt.detach()

    def fit(self, x, y_kt, y, update_ae=False):
        x = check_array(x)

        self.skipe_rnn.train()
        self.optimizer.zero_grad()
        y_hat, o, h = self.skipe_rnn(x, y_kt)
        y = torch.argmax(check_array(y), dim=-1)  # From one-hot to scaler

        loss = self.criterion(o, y)
        # print("y: ", y, "loss: ", loss)
        loss.backward()
        if self.skipe_rnn.activate == "hyperplane":
            self.skipe_rnn.hyperplane_bp(x, o, h)
        self.optimizer.step()

        if update_ae:
            self.autoencoder.fit(x)

    def initial_fit(self, X, y):
        n_samples, n_dim = X.shape
        indices = list(range(-n_samples, 0))
        # Train autoencoder first
        for x_t, y_t in zip(X, y):
            self.autoencoder.fit(x_t)

        # grow, prune = False, False
        for t in range(1, n_samples):
            index = indices[t]
            self.n_trained_samples += 1
            x_t, y_t, y_kt = X[t], y[t], y[t - 1]

            self.update_architecture(x_t, y_kt, y_t)

            self.fit(x_t, y_kt, y_t)

            self.history.insert(index, (x_t, y_t))
            self.states.insert(index, self.state_dict())

    def partial_fit(self, X, y, indices, y_avail=False, last_y=None):
        X = check_array(X)
        y = check_array(y)
        indices = np.asarray([indices]).flatten()
        for idx, single_x, single_y in zip(indices, X, y):
            if y_avail:
                assert last_y is not None
                if idx in self.pseudo:
                    if torch.any(self.pseudo[idx][1] != single_y):
                        try:
                            self.apply_penalty(self.states[idx])
                        except IndexError:
                            self.except_count += 1
                    self.pseudo.pop(idx)

                y_p = last_y.clone().detach()
                self.update_architecture(single_x, y_p, single_y)
                self.fit(single_x, y_p, single_y)
                self.history.insert(idx, (single_x, single_y))
            else:
                self.n_trained_samples += 1
                x_p, y_p = self.autoencoder.find_best_match(single_x, self.history)
                self.update_architecture(single_x, y_p, y_p)
                self.fit(single_x, y_p, y_p)
                self.pseudo.insert(idx, (single_x, y_p))

            self.autoencoder.fit(single_x)
            self.states.insert(idx, self.state_dict())

    def update_architecture(self, x_t, y_kt, y_t):
        # This part refers to the original code
        self.skipe_rnn.eval()
        self.skipe_rnn.x_stats.update(x_t)
        x_bar = self.skipe_rnn.x_stats.mean
        x_bar_2 = x_bar ** 2
        _, Eo, Eh = self.skipe_rnn(x_bar, y_kt)
        _, Eo_2, _ = self.skipe_rnn(x_bar_2, y_kt)

        Eo = Eo.detach()
        Eo_2 = Eo_2.detach()

        self._check_add_node(Eo, y_t)
        self._check_delete_node(Eh, Eo, Eo_2)

    def _check_add_node(self, Eo, y_t):
        NS = torch.mean((Eo - y_t) ** 2)
        self.skipe_rnn.NS_stats.update(NS)
        mean_std_NS = self.skipe_rnn.NS_stats.mean + self.skipe_rnn.NS_stats.std
        if self.n_trained_samples <= 1 or self.skipe_rnn.grow:
            self.skipe_rnn.min_NS_mean = self.skipe_rnn.NS_stats.mean
            self.skipe_rnn.min_NS_std = self.skipe_rnn.NS_stats.std
        else:
            self.skipe_rnn.min_NS_mean = min(self.skipe_rnn.min_NS_mean, self.skipe_rnn.NS_stats.mean)
            self.skipe_rnn.min_NS_std = min(self.skipe_rnn.min_NS_std, self.skipe_rnn.NS_stats.std)

        min_mean_std_NS = self.skipe_rnn.min_NS_mean + (1.3 * torch.exp(-NS) + 0.7) * self.skipe_rnn.min_NS_std

        if self._should_grow(mean_std_NS, min_mean_std_NS):
            self.skipe_rnn.grow = True
            print(f"A new node is added!")
            self.skipe_rnn.add_node()
            self._update_optimizer()
        else:
            self.skipe_rnn.grow = False

    def _check_delete_node(self, Eh, Eo, Eo_2):
        NHS = torch.mean(Eo_2 - Eo ** 2)
        self.skipe_rnn.NHS_stats.update(NHS)

        mean_std_NHS = self.skipe_rnn.NHS_stats.mean + self.skipe_rnn.NHS_stats.std

        if self.n_trained_samples <= 1 or self.skipe_rnn.prune:
            self.skipe_rnn.min_NHS_mean = self.skipe_rnn.NHS_stats.mean
            self.skipe_rnn.min_NHS_std = self.skipe_rnn.NHS_stats.std
        else:
            self.skipe_rnn.min_NHS_mean = min(self.skipe_rnn.min_NHS_mean, self.skipe_rnn.NHS_stats.mean)
            self.skipe_rnn.min_NHS_std = min(self.skipe_rnn.min_NHS_std, self.skipe_rnn.NHS_stats.std)

        min_mean_std_NHS = self.skipe_rnn.min_NHS_mean + (1.3 * torch.exp(-NHS) + 0.7) * self.skipe_rnn.min_NHS_std

        if not self.skipe_rnn.grow and self._should_prune(mean_std_NHS, min_mean_std_NHS):
            idx_to_del = torch.argmin(Eh)
            self.skipe_rnn.prune = True
            print(f"The {idx_to_del}-th node is deleted!")
            self.skipe_rnn.delete_node(idx_to_del)
            self._update_optimizer()
        else:
            self.skipe_rnn.prune = False

    def _should_grow(self, mean_std, min_mean_std):
        return (mean_std >= min_mean_std) and (self.n_trained_samples > 1) and (
                self.skipe_rnn.hidden_size <= self.hidden_upper_ratio * self.skipe_rnn.input_size)

    def _should_prune(self, mean_std, min_mean_std):
        return (not self.skipe_rnn.grow) and (self.skipe_rnn.hidden_size > 1) and (mean_std >= min_mean_std) and (
                self.n_trained_samples > self.skipe_rnn.input_size + 1)

    def _update_optimizer(self):
        target = self.skipe_rnn.parameters()
        self.optimizer = self.optimizer_class(target, lr=self.lr, momentum=self.momentum)
