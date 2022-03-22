import torch
from torch import nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

class SimpleNetWork(nn.Module):
    def __init__(self):
        super(SimpleNetWork,self).__init__()
        self.liner = nn.Sequential(
            nn.Linear(1,1)
        )

    def forward(self,x):
        predict = self.liner(x)
        return predict
model = SimpleNetWork()
print(model)

loss_fn = nn.MSELoss()

tensor_X = torch.from_numpy(X)
tensor_y = torch.from_numpy(y)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

