from base_models.skipe_rnn import SkipERNN


class SERN(SkipERNN):
    def __init__(self, input_size, output_size, hidden_size=1, eta=0.5, activate="hyperplane"):
        super().__init__(input_size, output_size, hidden_size, eta, activate)
