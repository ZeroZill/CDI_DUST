import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import Parameter
from typing import Union, List


class HBPModel(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 n_hidden_layers: int,
                 activation_function: nn.Module = nn.Sigmoid,
                 use_cuda: bool = False) -> None:
        super(HBPModel, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.activation_function = activation_function()
        self._initialize_layers()

    def _initialize_layers(self) -> None:
        # Construct hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size), self.activation_function)
        ])
        self.hidden_layers.extend([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size), self.activation_function)
            for _ in range(self.n_hidden_layers - 1)])

        # Construct output layers.
        # No explicit Softmax layer is required, as the softmax operation will be
        # automatically invoked when computing the cross-entropy loss in pytorch.
        # No OUTPUT LAYER CONNECTED TO THE INPUT, deviating from the original paper,
        # as we observed better performance with this modification.
        self.output_layers = nn.ModuleList([
            nn.Linear(self.hidden_size, self.output_size)
            for _ in range(self.n_hidden_layers)])

        # Move to device
        self.to(self.device)

    def forward(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)

        outputs = []
        h_i = X

        for i in range(self.n_hidden_layers):
            h_i = self.hidden_layers[i](h_i)
            outputs.append(self.output_layers[i](h_i))

        outputs = torch.stack(outputs, dim=0)
        return outputs


class HBP(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 n_hidden_layers: int,
                 batch_size: int = 1,
                 beta: float = 0.99,
                 eta: float = 0.01,
                 s: float = 0.2,
                 activation_function: nn.Module = nn.Sigmoid,
                 use_cuda: bool = False,
                 report_loss: bool = False,
                 report_interval: int = 1000
                 ) -> None:
        super().__init__()
        self.hbp = HBPModel(input_size,
                            output_size,
                            hidden_size,
                            n_hidden_layers,
                            activation_function,
                            use_cuda)

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        print(f"HBP is running on {self.device.upper()}.")

        self.batch_size = batch_size

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.n_hidden_layers = n_hidden_layers

        self.beta = Parameter(
            torch.tensor(beta, dtype=torch.float), requires_grad=False
        ).to(self.device)

        self.s = Parameter(
            torch.tensor(s, dtype=torch.float), requires_grad=False
        ).to(self.device)

        # Ensure alpha values are higher for lower layers, but not lower than self.s
        min_alpha = self.s / self.n_hidden_layers
        alpha_values = torch.linspace(start=2 / n_hidden_layers - min_alpha, end=min_alpha, steps=n_hidden_layers)
        self.alpha = Parameter(alpha_values, requires_grad=False).to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.report_loss = report_loss
        if self.report_loss:
            self.loss_array = []
            self.report_interval = report_interval

        self.optimizer = optim.SGD(self.hbp.parameters(), lr=eta)

    def _update_model(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        outputs = self.hbp(X)
        losses = [self.criterion(out, Y) for out in outputs]
        total_loss = sum(self.alpha[i] * loss for i, loss in enumerate(losses))
        total_loss.backward()
        self.optimizer.step()
        self._update_alpha(losses)

        self._print_training_loss(outputs, Y)

    def _update_alpha(self, losses: List[torch.Tensor]) -> None:
        with torch.no_grad():
            for i, loss in enumerate(losses):
                self.alpha[i] *= torch.pow(self.beta, loss)
                self.alpha[i] = torch.max(self.alpha[i], self.s / self.n_hidden_layers)
            self.alpha /= torch.sum(self.alpha)

    def _print_training_loss(self, outputs: torch.Tensor, Y: torch.Tensor) -> None:
        if self.report_loss:
            final_output = self._calc_final_output(outputs)
            final_loss = self.criterion(final_output, Y)
            self.loss_array.append(final_loss.item())
            if len(self.loss_array) % self.report_interval == 0:
                mean_loss = np.mean(self.loss_array)
                print(f"Alpha: {self.alpha.data.cpu().numpy()}")
                print(f"Training Loss: {mean_loss}")
                self.loss_array.clear()

    def _calc_final_output(self, outputs: torch.Tensor) -> torch.Tensor:
        weighted_outputs = torch.sum(self.alpha.view(-1, 1, 1) * outputs, dim=0)
        return weighted_outputs

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        X = self._validate_input(X, is_target=False)
        outputs = self.hbp(X)
        y_pred_probs = F.softmax(self._calc_final_output(outputs), dim=1)
        return y_pred_probs

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        y_pred = torch.argmax(self.predict_proba(X), dim=1)
        return y_pred

    def partial_fit(self,
                    X: Union[np.ndarray, torch.Tensor],
                    Y: Union[np.ndarray, torch.Tensor]
                    ) -> None:
        X = self._validate_input(X, is_target=False)
        Y = self._validate_input(Y, is_target=True)
        self._update_model(X, Y)

    def _validate_input(self,
                        input_data: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
                        is_target: bool = False
                        ) -> torch.Tensor:
        dtype = torch.float if not is_target else torch.long
        if isinstance(input_data, list):
            if len(input_data) == 0:
                raise ValueError("Input cannot be empty.")

            # Stack input_data along the appropriate axis
            input_data = torch.stack([torch.tensor(arr, dtype=dtype) for arr in input_data], dim=0)

        # If input_data is a numpy array, convert it to a PyTorch Tensor
        elif isinstance(input_data, np.ndarray):
            input_data = torch.tensor(input_data, dtype=dtype)

        # Ensure input_data is a PyTorch Tensor
        if not isinstance(input_data, torch.Tensor):
            raise ValueError("Input must be a numpy array, a PyTorch Tensor, "
                             "or a list of numpy arrays/PyTorch Tensor.")

        # Try to reshape input_data to the specified shape
        target_shape = (-1,) if is_target else (-1, self.input_size)
        input_data = input_data.view(*target_shape)
        return input_data


class HBPAdapter(HBP):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 n_hidden_layers: int,
                 batch_size: int = 1,
                 beta: float = 0.99,
                 eta: float = 0.01,
                 s: float = 0.2,
                 activation_function: nn.Module = nn.Sigmoid,
                 use_cuda: bool = False,
                 report_loss: bool = False,
                 report_interval: int = 1000
                 ):
        super().__init__(input_size,
                         output_size,
                         hidden_size,
                         n_hidden_layers,
                         batch_size,
                         beta,
                         eta,
                         s,
                         activation_function,
                         use_cuda,
                         report_loss,
                         report_interval)

    def partial_fit(self, X, Y, classes=None):
        for i in range(0, len(X), self.batch_size):
            batch_x = self._validate_input(X[i:i + self.batch_size], is_target=False)
            batch_y = self._validate_input(Y[i:i + self.batch_size], is_target=True)

            self._update_model(batch_x, batch_y)

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor], to_numpy=True) -> Union[np.ndarray, torch.Tensor]:
        X = self._validate_input(X, is_target=False)

        outputs = self.hbp(X)
        y_pred_probs = F.softmax(self._calc_final_output(outputs), dim=1)

        return y_pred_probs.data.cpu().numpy() if to_numpy else y_pred_probs.data.cpu()

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        y_pred = torch.argmax(self.predict_proba(X, to_numpy=False), dim=1)
        return y_pred
