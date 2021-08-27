import torch
import matplotlib.pyplot as plt

x_observe = []
y_observe = []
z_observe = []

with open('day_length_weight.csv') as dataset:
    data = dataset.readlines()[1:]
    for line in data:
        z, x, y = line.split(',')   #splits on ','
        x_observe.append(float(x))
        y_observe.append(float(y))
        z_observe.append(float(z))


x_train = torch.tensor(x_observe).reshape(-1, 1)
y_train = torch.tensor(y_observe).reshape(-1, 1)
z_train = torch.tensor(z_observe).reshape(-1, 1)


class LinearRegressionModel:
    def __init__(self):
        self.W1 = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.W2 = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # predictor
    def f(self, x1, x2):
        return x1 @ self.W1 + x2 @ self.W2 + self.b

    # Uses Mean Squared Error
    def loss(self, x1, x2, y):
        return torch.nn.functional.mse_loss(self.f(x1, x2), y)


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W1, model.W2, model.b], 0.0001)
for epoch in range(100000):
    model.loss(x_train, y_train, z_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W1 = %s, W2 = %s b = %s, loss = %s" % (model.W1, model.W2, model.b, model.loss(x_train, y_train, z_train)))

# Visualize result
fig = plt.figure('Linear regression 3d')
ax = plt.axes(projection='3d', title="Predict days based on length and weight")
# Information for making the plot understandable
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
ax.set_xticks([])  # Removes the lines and information from axes
ax.set_yticks([])
ax.set_zticks([])
ax.w_xaxis.line.set_lw(0)
ax.w_yaxis.line.set_lw(0)
ax.w_zaxis.line.set_lw(0)
ax.quiver([0], [0], [0], [torch.max(x_train + 1)], [0],
          [0], arrow_length_ratio=0.05, color='black')
ax.quiver([0], [0], [0], [0], [torch.max(y_train + 1)],
          [0], arrow_length_ratio=0.05, color='black')
ax.quiver([0], [0], [0], [0], [0], [torch.max(z_train + 1)],
          arrow_length_ratio=0, color='black')
# Plot
ax.scatter(x_observe, y_observe, z_observe)
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
y = torch.tensor([[torch.min(y_train)], [torch.max(y_train)]])
ax.plot(x.flatten(), y.flatten(), model.f(
    x, y).detach().flatten(), label='$f(x)=x1W1+x2W2+b$', color="orange")
# TODO: Fix 3D plane
ax.legend()
plt.show()