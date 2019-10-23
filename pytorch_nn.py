import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([1.0]), requires_grad=True)

def forward(x):
    return x * w

def loss(x, y):
    y_prediction = forward(x)
    return (y_prediction - y)*(y_prediction - y)

# Need to use data
print("Prediction before training", 4, forward(4).data[0])

for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)

        # sample back propagation process
        l.backward()

        print("tgrad: ", x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01* w.grad.data 

        # after one loop need to reset (w) to 0
        w.grad.data.zero_()
    print("Progress: ", epoch, l.data[0])

print("Prediction after training", 4, forward(4).data[0])