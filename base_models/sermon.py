import copy

import numpy as np
import torch
import torch.nn as nn

from base_models.mern import MERN
from base_models.sern import SERN
from base_models.auto_encoder import AutoencoderModel
from utils import check_array, matrix_2_norm, MostRecentCache, NormalCache


class SERMONModel:
    def __init__(self, input_size, output_size,
                 hidden_size=1,
                 eta=0.5,
                 activate="hyperplane",
                 criterion=nn.CrossEntropyLoss,
                 optimizer=torch.optim.SGD,
                 layer_score_factor=0.1,
                 lr=0.01,
                 momentum=0.95,
                 n_labeled_samples_to_check=2500,
                 max_history_cache=1000):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.sern = SERN(input_size, output_size, hidden_size, eta, activate)
        self.mern = MERN(input_size, output_size, hidden_size, eta, activate)
        self.criterion = criterion()
        self.optimizer_class = optimizer

        self.optimizer_sern = optimizer(self.sern.parameters(), lr=lr, momentum=momentum)
        self.optimizer_mern = [
            optimizer(self.mern[i].parameters(), lr=lr, momentum=momentum)
            for i in range(self.mern.n_hidden_layer)
        ]

        self.lr = lr
        self.momentum = momentum
        self.layer_score_factor = layer_score_factor

        self.hidden_upper_ratio = 0.5 if input_size <= 10 else 0.1

        self.autoencoder = AutoencoderModel(input_size)

        self.history = MostRecentCache(max_history_cache)
        self.pseudo = NormalCache()
        self.sern_states = MostRecentCache(max_history_cache + 100)

        self.current_time_step = 0
        self.n_trained_samples = 0

        self.n_labeled_samples = 0
        self.n_labeled_samples_to_check = n_labeled_samples_to_check

        self.except_count = 0

    def sern_state_dict(self):
        return copy.deepcopy(self.sern.state_dict())

    def apply_penalty_on_sern(self, prev_state, label_avail_state, next_label_avail_state):
        self.optimizer_sern.zero_grad(set_to_none=False)
        # Add penalty on grad for SERN

        # It's worth noting that there are several issues in this section:
        # - The penalty applied here may be too large, which could potentially result in parameter
        # values becoming infinity.
        # - There exists some points, that the denominator is 0, so that the penalty become infinity.
        # The phenomenon exists in the original code, since the original code performs well,
        # so we keep it that way.

        U_weight_numerator = (matrix_2_norm(self.sern.U.weight.data) - matrix_2_norm(prev_state["U.weight"]))
        U_bias_numerator = (matrix_2_norm(self.sern.U.bias.data) - matrix_2_norm(prev_state["U.bias"]))
        U_weight_denominator = \
            torch.sqrt(torch.abs(
                matrix_2_norm(next_label_avail_state["U.weight"]) - matrix_2_norm(label_avail_state["U.weight"])))
        U_bias_denominator = \
            torch.sqrt(torch.abs(
                matrix_2_norm(next_label_avail_state["U.bias"]) - matrix_2_norm(label_avail_state["U.bias"])))

        V_weight_numerator = (matrix_2_norm(self.sern.V.weight.data) - matrix_2_norm(prev_state["V.weight"]))
        V_bias_numerator = (matrix_2_norm(self.sern.V.bias.data) - matrix_2_norm(prev_state["V.bias"]))
        V_weight_denominator = \
            torch.sqrt(torch.abs(
                matrix_2_norm(next_label_avail_state["V.weight"]) - matrix_2_norm(label_avail_state["V.weight"])))
        V_bias_denominator = \
            torch.sqrt(torch.abs(
                matrix_2_norm(next_label_avail_state["V.bias"]) - matrix_2_norm(label_avail_state["V.bias"])))

        U_weight_penalty = 2 * U_weight_numerator / U_weight_denominator
        U_bias_penalty = 2 * U_bias_numerator / U_bias_denominator

        V_weight_penalty = 2 * V_weight_numerator / V_weight_denominator
        V_bias_penalty = 2 * V_bias_numerator / V_bias_denominator

        self.sern.U.weight.grad.data += U_weight_penalty
        self.sern.U.bias.grad.data += U_bias_penalty
        self.sern.V.weight.grad.data += V_weight_penalty
        self.sern.V.bias.grad.data += V_bias_penalty

        self.optimizer_sern.step()

    def predict(self, x, y_kt_sern, y_kt_mern):
        self.sern.eval()
        y_hat_sern, next_y_kt_sern, _ = self.sern(x, y_kt_sern)

        self.mern[self.mern.best_layer_idx].eval()
        y_hat_mern, next_y_kt_mern, _ = self.mern[self.mern.best_layer_idx](x, y_kt_mern)

        return y_hat_sern, next_y_kt_sern.detach(), y_hat_mern, next_y_kt_mern.detach()

    def fit(self, x, y_kt, y, target="sern", update_ae=False):
        """
        target: one of "sern", "mern", "both"
        """
        x = check_array(x)
        y = torch.argmax(check_array(y), dim=-1)

        if target == "sern" or target == "both":
            self._fit_model(self.sern, self.optimizer_sern, x, y_kt, y)

        if target == "mern" or target == "both":
            for i in range(self.mern.n_hidden_layer):
                self._fit_model(self.mern[i], self.optimizer_mern[i], x, y_kt, y)

        if update_ae:
            self.autoencoder.fit(x)

    def _fit_model(self, model, optimizer, x, y_kt, y):
        model.train()
        optimizer.zero_grad()
        y_hat, o, h = model(x, y_kt)

        loss = self.criterion(o, y)
        loss.backward()

        if model.activate == "hyperplane":
            model.hyperplane_bp(x, o, h)

        optimizer.step()

    def initial_fit(self, X, y):
        n_samples, n_dim = X.shape
        indices = list(range(-n_samples, 0))
        # Train autoencoder first
        for x_t, y_t in zip(X, y):
            self.autoencoder.fit(x_t)

        for t in range(1, n_samples):
            index = indices[t]
            self.n_trained_samples += 1
            self.n_labeled_samples += 1
            x_t, y_t, y_kt = X[t], y[t], y[t - 1]

            self.update_architecture(x_t, y_kt, y_t, target="sern")
            self.fit(x_t, y_kt, y_t, target="sern")

            self.update_architecture(x_t, y_kt, y_t, target="mern")
            self.fit(x_t, y_kt, y_t, target="mern")

            self.history.insert(index, (x_t, y_t))
            self.sern_states.insert(index, self.sern_state_dict())

    def partial_fit(self, X, y, indices, y_kt_mern, y_avail=False, last_y=None):
        X = check_array(X)
        y = check_array(y)
        indices = np.asarray([indices]).flatten()
        y_kt_mern = y_kt_mern.clone().detach()
        for idx, single_x, single_y in zip(indices, X, y):
            if y_avail:
                assert last_y is not None
                if idx in self.pseudo:
                    if torch.any(self.pseudo[idx][1] != single_y):
                        try:
                            self.apply_penalty_on_sern(self.sern_states[self.current_time_step - 1],
                                                       self.sern_states[idx],
                                                       self.sern_states[idx + 1])
                        except IndexError:
                            self.except_count += 1
                    self.pseudo.pop(idx)

                self.n_labeled_samples += 1

                y_p = last_y.clone().detach()
                self.update_architecture(single_x, y_p, single_y, target="sern")
                self.fit(single_x, y_p, single_y, target="sern")

                self.update_architecture(single_x, y_p, single_y, target="mern")
                self.fit(single_x, y_p, single_y, target="mern")

                self.history.insert(idx, (single_x, single_y))

                if self.n_labeled_samples % self.n_labeled_samples_to_check == 0:
                    self._update_mern_info()
            else:
                self.current_time_step = idx
                self.n_trained_samples += 1
                x_p, y_p = self.autoencoder.find_best_match(single_x, self.history)
                self.update_architecture(single_x, y_p, y_p, target="sern")
                self.fit(single_x, y_p, y_p, target="sern")

                self.update_architecture(single_x, y_kt_mern, y_p, target="mern")
                self.fit(single_x, y_kt_mern, y_p, target="mern")

                self.pseudo.insert(idx, (single_x, y_p))

            self.autoencoder.fit(single_x)
            self.sern_states.insert(idx, self.sern_state_dict())

    def update_architecture(self, x_t, y_kt, y_t, target="sern"):
        """
        target: one of "sern", "mern", "both"
        """
        if target == "sern" or target == "both":
            self._update_model_hidden_architecture(x_t, y_kt, y_t, target="sern", idx=None)

        if target == "mern" or target == "both":
            for i in range(self.mern.n_hidden_layer):
                self._update_model_hidden_architecture(x_t, y_kt, y_t, target="mern", idx=i)

    def _update_mern_info(self):
        # Get latest labeled data with number of `self.n_labeled_samples`
        samples, labels = self.history.get_samples_and_labels_as_tensor(self.n_labeled_samples_to_check)

        mern_acc, mismatch_pred_with_mern = self._test_all_model_with_labeled_data(labels, samples)

        # Update the best layer ever ("best layer ever" is stored for hidden layer addition)
        self._update_mern_best_layer_ever()

        # Get the scores of each hidden layer in MERN
        self._calc_hidden_layer_scores(mern_acc)

        self.mern.last_hidden_layer_acc = mern_acc

        self.mern.best_layer_idx = np.argmax(self.mern.hidden_layer_scores)

        # print(
        #     f"\t# MERN_acc = {mern_acc}, best layer = {self.mern.best_layer_idx}, "
        #     f"MERN mismatch_stats: {mismatch_pred_with_mern}")

        # Check whether to add hidden layer at next check point

        # It is crucial to highlight a significant issue in this section:
        # - Achieving a mismatch rate exceeding 50% between the class labels predicted by SERN and MERN
        # is exceedingly challenging. Even if the model were to make random guesses, the expected mismatch
        # rate would still only be 50%.

        # Associated with the issue related to the penalty mechanism mentioned earlier, I have identified the
        # root cause. The penalty applied may be too large (even with value of infinity), resulting in model
        # parameters becoming infinite. Consequently, the predictions generated by SERN remain fixed. In such
        # a scenario, achieving a 50% mismatch rate between SERN and MERN predictions becomes relatively easier.

        if self._should_add_layer(mismatch_pred_with_mern):
            print("A new hidden layer is added!")
            self.mern.add_layer()
            self.optimizer_mern.append(
                self.optimizer_class(self.mern[-1].parameters(), lr=self.lr, momentum=self.momentum))

    def _calc_hidden_layer_scores(self, mern_acc):
        for i in range(self.mern.n_hidden_layer):
            if mern_acc[i] < self.mern.last_hidden_layer_acc[i]:
                self.mern.hidden_layer_scores[i] = max(0.0, mern_acc[i] - self.layer_score_factor)
            else:
                self.mern.hidden_layer_scores[i] = min(1.0, mern_acc[i] + self.layer_score_factor)

    def _update_mern_best_layer_ever(self):
        best_layer_ever_idx = np.argmax(self.mern.hidden_layer_acc_over_time)
        best_acc = self.mern.hidden_layer_acc_over_time[best_layer_ever_idx].mean
        if best_acc > self.mern.best_acc_ever:
            self.mern.best_acc_ever = best_acc
            self.mern.update_best_layer_ever(best_layer_ever_idx)

    def _test_all_model_with_labeled_data(self, labels, samples):
        self.sern.eval()
        for model in self.mern.hidden_layers:
            model.eval()
        mismatch_pred_with_mern = [0] * self.mern.n_hidden_layer
        mern_acc = [0.0] * self.mern.n_hidden_layer
        # Test with the latest samples
        for sample, label, label_kt in zip(samples[1:], labels[1:], labels[0:]):
            y_hat_sern, o_sern, h_sern = self.sern(sample, label_kt)
            for i in range(self.mern.n_hidden_layer):
                model = self.mern[i]
                y_hat_mern_i, o_mern_i, h_mern_i = model(sample, label_kt)

                mismatch_pred_with_mern[i] += (y_hat_mern_i != y_hat_sern)
                correct_or_not = int(y_hat_mern_i == torch.argmax(label))
                self.mern.hidden_layer_acc_over_time[i].update(correct_or_not)
                mern_acc[i] += correct_or_not
        for i in range(self.mern.n_hidden_layer):
            mern_acc[i] /= (self.n_labeled_samples_to_check - 1)
        return mern_acc, mismatch_pred_with_mern

    def _should_add_layer(self, mismatch_pred_with_mern):
        return mismatch_pred_with_mern[self.mern.best_layer_idx] / (self.n_labeled_samples_to_check - 1) \
               > (1 - 1 / self.output_size) and self.mern.n_hidden_layer / self.input_size < 0.25

    def _get_model(self, target, idx):
        if target == "sern":
            model = self.sern
        elif target == "mern":
            assert idx is not None and 0 <= idx < self.mern.n_hidden_layer
            model = self.mern[idx]
        else:
            raise ValueError(f"Target {target} not supported, must be 'sern' or 'mern'")
        return model

    def _update_model_hidden_architecture(self, x_t, y_kt, y_t, target="sern", idx=None):
        model = self._get_model(target, idx)

        model.eval()
        model.x_stats.update(x_t)
        x_bar = model.x_stats.mean
        x_bar_2 = x_bar ** 2
        _, Eo, Eh = model(x_bar, y_kt)
        _, Eo_2, _ = model(x_bar_2, y_kt)

        Eo = Eo.detach()
        Eo_2 = Eo_2.detach()

        self._check_add_node(Eo, y_t, target, idx)
        self._check_delete_node(Eh, Eo, Eo_2, target, idx)

    def _check_add_node(self, Eo, y_t, target="sern", idx=None):
        model = self._get_model(target, idx)

        NS = torch.mean((Eo - y_t) ** 2)
        model.NS_stats.update(NS)

        mean_std_NS = model.NS_stats.mean + model.NS_stats.std

        if self.n_trained_samples <= 1 or model.grow:
            model.min_NS_mean = model.NS_stats.mean
            model.min_NS_std = model.NS_stats.std
        else:
            model.min_NS_mean = min(model.min_NS_mean, model.NS_stats.mean)
            model.min_NS_std = min(model.min_NS_std, model.NS_stats.std)

        min_mean_std_NS = model.min_NS_mean + (1.3 * torch.exp(-NS) + 0.7) * model.min_NS_std

        if self._should_grow(mean_std_NS, min_mean_std_NS, model):
            model.grow = True
            # print(f"A new node is added in {target.upper()}[{idx}]!")
            model.add_node()
            self._update_optimizer(target, idx)
        else:
            model.grow = False

    def _check_delete_node(self, Eh, Eo, Eo_2, target="sern", idx=None):
        model = self._get_model(target, idx)

        NHS = torch.mean(Eo_2 - Eo ** 2)
        model.NHS_stats.update(NHS)

        mean_std_NHS = model.NHS_stats.mean + model.NHS_stats.std

        if self.n_trained_samples <= 1 or model.prune:
            model.min_NHS_mean = model.NHS_stats.mean
            model.min_NHS_std = model.NHS_stats.std
        else:
            model.min_NHS_mean = min(model.min_NHS_mean, model.NHS_stats.mean)
            model.min_NHS_std = min(model.min_NHS_std, model.NHS_stats.std)

        min_mean_std_NHS = model.min_NHS_mean + (1.3 * torch.exp(-NHS) + 0.7) * model.min_NHS_std

        if not model.grow and self._should_prune(mean_std_NHS, min_mean_std_NHS, model):
            idx_to_del = torch.argmin(Eh)
            model.prune = True
            # print(f"The {idx_to_del}-th node is deleted in {target.upper()}[{idx}]!")
            model.delete_node(idx_to_del)
            self._update_optimizer(target, idx)
        else:
            model.prune = False

    def _should_grow(self, mean_std, min_mean_std, model):
        return (mean_std >= min_mean_std) and (self.n_trained_samples > 1) and (
                model.hidden_size <= self.hidden_upper_ratio * model.input_size)

    def _should_prune(self, mean_std, min_mean_std, model):
        return (not model.grow) and (model.hidden_size > 1) and (mean_std >= min_mean_std) and (
                self.n_trained_samples > model.input_size + 1)

    def _update_optimizer(self, target="sern", idx=None):
        if target == "sern":
            target_parameters = self.sern.parameters()
            self.optimizer_sern = self.optimizer_class(target_parameters, lr=self.lr, momentum=self.momentum)
        elif target == "mern":
            assert idx is not None and 0 <= idx < self.mern.n_hidden_layer
            target_parameters = self.mern[idx].parameters()
            self.optimizer_mern[idx] = self.optimizer_class(target_parameters, lr=self.lr, momentum=self.momentum)
        else:
            raise ValueError(f"Target {target} not supported, must be 'sern' or 'mern'")
