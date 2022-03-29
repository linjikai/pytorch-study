import time

import torch
from torch import nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

print(torch.version.__version__)

X, y = datasets.make_regression(
    n_samples=10000,
    noise=0,
    n_features=1,
    random_state=8,
    n_informative=1
)

plt.scatter(X, y, c='blue')
plt.show()

tensor_X = torch.from_numpy(X)
tensor_y = torch.from_numpy(y)
tensor_y = torch.unsqueeze(tensor_y, dim=1)


class SimpleNetWork(nn.Module):
    def __init__(self):
        super(SimpleNetWork, self).__init__()
        self.liner = nn.Sequential(
            nn.Linear(1, 1)
        )

    def forward(self, x):
        predict = self.liner(x)
        return predict


model = SimpleNetWork().double()
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for t in range(4000):

    predict_y = model(tensor_X)
    loss = loss_fn(predict_y, tensor_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 100 == 0:
        # time.sleep(1)
        print(f"step:{t},loss:{loss}")

for name, param in model.named_parameters():
    print(name, param)


def test():
    test_x = np.linspace(-4, 4, 10)
    print(f"test_x:{test_x}")
    test_y = model(torch.unsqueeze(torch.from_numpy(test_x), dim=1))

    print(f"predict_y:{test_y}")

    plt.plot(test_x, test_y.detach().numpy(), color='red')
    plt.scatter(X, y, c='blue')
    plt.show()


test()
