import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from torch import nn

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

        # Parameters initialization (He. Initialization)
        # nn.init.kaiming_normal_(self.encoder.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.decoder.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.encoder.weight, mean=0.0, std=1.0 / input_size ** 0.5)
        # init.xavier_uniform_(self.U.weight, gain=4.0)
        # init.constant_(self.U.weight, 1.0)
        nn.init.constant_(self.encoder.bias, 0.0)

        nn.init.xavier_uniform_(self.decoder.weight, gain=4.0)
        # init.constant_(self.V.weight, 1.0)
        nn.init.constant_(self.decoder.bias, 0.0)

    def forward(self, x):
        encoded = self.sigmoid(self.encoder(x))
        decoded = self.sigmoid(self.decoder(encoded))
        return decoded

    def encode(self, x):
        return self.sigmoid(self.encoder(x))

    def decode(self, x_p):
        return self.sigmoid(self.decoder(x_p))


class AutoencoderModel:
    def __init__(self, n_features,
                 hidden_size=None,
                 criterion=nn.MSELoss,
                 optimizer=torch.optim.SGD,
                 lr=0.01,
                 momentum=0.95):
        if hidden_size is None:
            hidden_size = int(np.ceil(n_features / 2)) + 1
        self.autoencoder = Autoencoder(n_features, hidden_size)
        self.criterion = criterion()
        self.optimizer = optimizer(self.autoencoder.parameters(), lr=lr, momentum=momentum)

    def find_best_match(self, x, history):
        self.autoencoder.eval()

        samples, labels = history.get_samples_and_labels_as_tensor()

        x_primes = self.autoencoder(samples).detach().numpy()
        x_ = self.autoencoder(x).detach().numpy()

        kd_tree = KDTree(x_primes)
        _, index = kd_tree.query(x_.reshape(1, -1), k=1)

        index = index[0]
        best_match_x, best_match_y = samples[index], labels[index]

        return best_match_x, best_match_y

    def predict(self, x):
        self.autoencoder.eval()
        return self.autoencoder(x)

    def encode(self, x):
        self.autoencoder.eval()
        return self.autoencoder.encode(x)

    def decode(self, x):
        self.autoencoder.eval()
        return self.autoencoder.decode(x)

    def fit(self, x, return_loss=False):
        self.autoencoder.train()
        self.optimizer.zero_grad()
        x_prime = self.autoencoder(x)
        loss = self.criterion(x_prime, x)
        # print(f"loss={loss.item()}")
        loss.backward()
        self.optimizer.step()
        if return_loss:
            return loss
        else:
            return None

#
# num_epochs = 50
#
# ae_model = AutoencoderModel(10, 10, lr=0.01)
#
# # 生成示例数据（随机数据）
# data = torch.randn(1000, 10)
# losses = []
#
# # 训练autoencoder
# for epoch in range(num_epochs):
#     total_loss = 0
#     for single_d in data:
#         single_loss = ae_model.fit(single_d, return_loss=True)
#         total_loss += single_loss.item()
#     average_loss = total_loss / len(data)  # 计算平均损失
#     losses.append(average_loss)  # 将平均损失添加到 losses 列表中
#
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')
#
# print("Autoencoder training complete!")
#
# # 使用训练后的autoencoder进行重构
# reconstructed_data = ae_model.predict(data)
#
# # 查看重构误差
# reconstruction_loss = torch.mean((reconstructed_data - data) ** 2)
# print(f'Reconstruction Loss: {reconstruction_loss.item():.4f}')
#
# plt.plot(np.arange(len(losses)), losses)
# plt.title("Loss")
#
# plt.show()
#
# new_data = torch.randn(1, 10)
# print("New data:\n", new_data)
# print("Reconstructed new data:\n", ae_model.predict(new_data))
#
# a = 1
