import torch
import torch.nn as nn
import numpy as np
from process_data import train_loader, test_loader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class MyRnn(nn.Module):
    def __init__(self):
        super(MyRnn, self).__init__()
        weight_numpy = np.load("/Users/ljk/Documents/code/pytorch/pytorch-study/com/study/rnn/bd.dim300.ckpt.npy")
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weight_numpy))
        self.embedding_dim = 300

        self.rnn_layer = nn.RNN(
            self.embedding_dim,
            hidden_size=50,
            num_layers=1,
            batch_first=True
        )

        self.liner_layer = nn.Linear(
            in_features=50,
            out_features=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        word_embed = self.embedding(x)
        out, h = self.rnn_layer(word_embed)
        last_out = out[:,-1,:]
        liner_out = self.liner_layer(last_out)
        predict_y = self.sigmoid(liner_out)
        p_y = predict_y.view(-1)
        return p_y


model = MyRnn()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_losses = []  # 记录测试数据集的损失
    num_correct = 0  # 记录正确预测的数量
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.float()).item()
            test_losses.append(test_loss)
            pred = torch.round(pred.squeeze())  # 将模型四舍五入为0和1
            correct_tensor = pred.eq(y.float().view_as(pred))  # 计算预测正确的数据
            correct = np.squeeze(correct_tensor.cpu().numpy())
            num_correct += np.sum(correct)
    test_loss /= num_batches
    num_correct /= size
    print(f"Test Error: \n Accuracy: {(100 * num_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 50
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
