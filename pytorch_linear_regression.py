import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        #1 input and 1 output
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        y_prediction = self.linear(x)
        return y_prediction

model = Model() 

criterion = torch.nn.MSELoss(size_average=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)

    print(epoch, loss.data)
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

output = Variable(torch.Tensor([[4.0]]))
y_pred = model(output)

print("Prediction after training: ", 4, model(output).data[0][0])
