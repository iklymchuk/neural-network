import torch

from torch.autograd import Variable
import numpy as np 
import torch.nn.functional as F 

input_data = "./data/diabetes.csv"

xy = np.loadtxt(input_data, delimiter=',', dtype=np.float32)

x_data = Variable(torch.from_numpy(xy[:,0:-1]))
y_data = Variable(torch.from_numpy(xy[:,[-1]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Describe first Layer of the Network
        self.l1 = torch.nn.Linear(8, 6)

        # The second layer of the Network 
        self.l2 = torch.nn.Linear(6, 4)

        # Binary classification as output
        self.l3 = torch.nn.Linear(4, 1)

        # Activation function
        self.sigmoid = torch.nn.Sigmoid()

    # Implement the forward path
    def forward(self, x):
        out1 = F.relu(self.l1(x))
        out2 = F.relu(self.l2(out1))
        y_prediction = self.sigmoid(self.l3(out2))
        return y_prediction

model = Model()

criterion = torch.nn.BCELoss(size_average=True)

# For optimizer we selected SGD - stohastic gradient descent with learning rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training process
for epoch in range (1000):
    y_prediction = model(x_data)

    loss = criterion(y_prediction, y_data)
    print(epoch, loss.data)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
