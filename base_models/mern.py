import copy

from base_models.skipe_rnn import SkipERNN

from utils import AverageStatistic


class MERN:
    def __init__(self, input_size, output_size, hidden_size=1, eta=0.5, activate="hyperplane"):
        super(MERN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.eta = eta
        self.n_hidden_layer = 1
        self.n_layers = 3

        self.hidden_layers = \
            [SkipERNN(input_size, output_size, hidden_size, eta, activate) for _ in range(self.n_hidden_layer)]

        self.best_layer_idx = 0

        self.hidden_layer_acc_over_time = [AverageStatistic()] * self.n_hidden_layer

        self.last_hidden_layer_acc = [0.0] * self.n_hidden_layer

        self.hidden_layer_scores = [0.0] * self.n_hidden_layer

        self.best_layer_ever = copy.deepcopy(self.hidden_layers[self.best_layer_idx])
        self.best_acc_ever = 0.0

        self.activate = activate

    def __getitem__(self, item):
        return self.hidden_layers[item]

    def add_layer(self):
        self.n_hidden_layer += 1
        self.hidden_layer_acc_over_time.append(AverageStatistic())

        new_layer = copy.deepcopy(self.best_layer_ever)
        self.hidden_layers.append(new_layer)

        self.last_hidden_layer_acc.append(0.0)

        self.hidden_layer_scores.append(0.0)

    def update_best_layer_ever(self, best_layer_ever_idx):
        hidden_size = self.hidden_layers[best_layer_ever_idx].hidden_size
        self.best_layer_ever = SkipERNN(self.input_size, self.output_size, hidden_size, self.eta, self.activate)
        self.best_layer_ever.load_state_dict(copy.deepcopy(self.hidden_layers[best_layer_ever_idx].state_dict()))
