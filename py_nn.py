x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

# Calculate our prediction (forward path)
def forward(x):
    return x*w

# Calculate our loss function
def loss(x, y):
    y_prediction = forward(x)
    return (y_prediction-y)*(y_prediction-y)

# Calculate the mathematical ocasion
def gradient(x, y):
    return 2*x*(x*w-y)

# Prediction before training
print("Prediction before training", 4, forward(4))

# Training proces
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):

        grad = gradient(x_val, y_val)

        w = w -0.01*grad
        print("\tgrad: ", x_val, y_val, round(grad, 2))

        l = loss(x_val, y_val)
    print("Progress: ", epoch, "w=", round(w, 2), "loss=", round(l,2))

# Prediction after training
print("Prediction after training", forward(4))