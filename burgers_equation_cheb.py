import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs  # Latin Hypercube Sampling from the pyDOE package
from torch.nn import init

# Define the neural network architecture
class ChNNModel(nn.Module):
    def __init__(self,  num_degree):
        super(ChNNModel, self).__init__()
        self.fc = nn.Linear((num_degree+1)**2, 1)
        init.xavier_uniform_(self.fc.weight)  # 初始化隐藏层的权重

    def forward(self, input_data):
        x = input_data[:,0].reshape(-1,1)
        xx = x*torch.ones(1,5)
        y = input_data[:, 1].reshape(-1, 1)
        yy = y*torch.ones(1,5)
        T1_x = x
        T2_x = 2 * x ** 2 - 1
        T3_x = 4 * x ** 3 - 3 * x
        T4_x = 8 * x ** 4 - 8 * x ** 2 + 1
        T1_y = y
        T2_y = 2*y**2-1
        T3_y = 4 * y**3-3*y
        T4_y = 8 * y ** 4 - 8 * y**2 + 1
        xx[:,0] = torch.ones_like(x[:,0])
        xx[:, 1] = T1_x.reshape(-1,)
        xx[:, 2] = T2_x.reshape(-1,)
        xx[:, 3] = T3_x.reshape(-1,)
        xx[:, 4] = T4_x.reshape(-1, )
        yy[:, 0] = torch.ones_like(y[:, 0])
        yy[:, 1] = T1_y.reshape(-1,)
        yy[:, 2] = T2_y.reshape(-1,)
        yy[:, 3] = T3_y.reshape(-1,)
        yy[:, 4] = T4_y.reshape(-1, )
        # 使用torch.kron进行张量积操作
        # T = torch.cat(((xx[:, 0]*yy[:, 0]).reshape(-1,1),(xx[:, 0]*yy[:, 1]).reshape(-1,1),(xx[:, 0]*yy[:, 2]).reshape(-1,1),(xx[:, 0]*yy[:, 3]).reshape(-1,1),(xx[:, 0]*yy[:, 4]).reshape(-1,1),
        #                (xx[:, 1]*yy[:, 0]).reshape(-1,1),(xx[:, 1]*yy[:, 1]).reshape(-1,1),(xx[:, 1]*yy[:, 2]).reshape(-1,1),(xx[:, 1]*yy[:, 3]).reshape(-1,1),(xx[:, 1]*yy[:, 4]).reshape(-1,1),
        #                (xx[:, 2]*yy[:, 0]).reshape(-1,1),(xx[:, 2]*yy[:, 1]).reshape(-1,1),(xx[:, 2]*yy[:, 2]).reshape(-1,1),(xx[:, 2]*yy[:, 3]).reshape(-1,1),(xx[:, 2]*yy[:, 4]).reshape(-1,1),
        #                (xx[:, 3]*yy[:, 0]).reshape(-1,1),(xx[:, 3]*yy[:, 1]).reshape(-1,1),(xx[:, 3]*yy[:, 2]).reshape(-1,1),(xx[:, 3]*yy[:, 3]).reshape(-1,1),(xx[:, 3]*yy[:, 4]).reshape(-1,1),
        #                (xx[:, 4]*yy[:, 0]).reshape(-1,1),(xx[:, 4]*yy[:, 1]).reshape(-1,1),(xx[:, 4]*yy[:, 2]).reshape(-1,1),(xx[:, 4]*yy[:, 3]).reshape(-1,1),(xx[:, 4]*yy[:, 4]).reshape(-1,1)),dim=1)
        # T = torch.cat(((xx[:, 0] * yy[:, 0]).reshape(-1, 1), (xx[:, 0] * yy[:, 1]).reshape(-1, 1),
        #                (xx[:, 0] * yy[:, 2]).reshape(-1, 1), (xx[:, 0] * yy[:, 3]).reshape(-1, 1),
        #
        #                (xx[:, 1] * yy[:, 0]).reshape(-1, 1), (xx[:, 1] * yy[:, 1]).reshape(-1, 1),
        #                (xx[:, 1] * yy[:, 2]).reshape(-1, 1), (xx[:, 1] * yy[:, 3]).reshape(-1, 1),
        #
        #                (xx[:, 2] * yy[:, 0]).reshape(-1, 1), (xx[:, 2] * yy[:, 1]).reshape(-1, 1),
        #                (xx[:, 2] * yy[:, 2]).reshape(-1, 1), (xx[:, 2] * yy[:, 3]).reshape(-1, 1),
        #
        #                (xx[:, 3] * yy[:, 0]).reshape(-1, 1), (xx[:, 3] * yy[:, 1]).reshape(-1, 1),
        #                (xx[:, 3] * yy[:, 2]).reshape(-1, 1), (xx[:, 3] * yy[:, 3]).reshape(-1, 1)), dim=1)
        T = torch.cat(((xx[:, 0] * yy[:, 0]).reshape(-1, 1), (xx[:, 0] * yy[:, 1]).reshape(-1, 1),
                       (xx[:, 0] * yy[:, 2]).reshape(-1, 1), (xx[:, 0] * yy[:, 3]).reshape(-1, 1),
                       (xx[:, 0] * yy[:, 4]).reshape(-1, 1),
                       (xx[:, 1] * yy[:, 0]).reshape(-1, 1), (xx[:, 1] * yy[:, 1]).reshape(-1, 1),
                       (xx[:, 1] * yy[:, 2]).reshape(-1, 1), (xx[:, 1] * yy[:, 3]).reshape(-1, 1),
                       (xx[:, 1] * yy[:, 4]).reshape(-1, 1),
                       (xx[:, 2] * yy[:, 0]).reshape(-1, 1), (xx[:, 2] * yy[:, 1]).reshape(-1, 1),
                       (xx[:, 2] * yy[:, 2]).reshape(-1, 1), (xx[:, 2] * yy[:, 3]).reshape(-1, 1),
                       (xx[:, 2] * yy[:, 4]).reshape(-1, 1),
                       (xx[:, 3] * yy[:, 0]).reshape(-1, 1), (xx[:, 3] * yy[:, 1]).reshape(-1, 1),
                       (xx[:, 3] * yy[:, 2]).reshape(-1, 1), (xx[:, 3] * yy[:, 3]).reshape(-1, 1),
                       (xx[:, 3] * yy[:, 4]).reshape(-1, 1),
                       (xx[:, 4] * yy[:, 0]).reshape(-1, 1), (xx[:, 4] * yy[:, 1]).reshape(-1, 1),
                       (xx[:, 4] * yy[:, 2]).reshape(-1, 1), (xx[:, 4] * yy[:, 3]).reshape(-1, 1),
                       (xx[:, 4] * yy[:, 4]).reshape(-1, 1)
                       ),
                      dim=1)

        h1 = T
        return self.fc(h1)


def prepare_data(num_samples=500, num_boundary=100):
    # Interior points
    lhc = lhs(3, samples=num_samples)
    x = 2 * lhc[:, 0:1] - 1  # Scale x to [-1, 1]
    y = 2 * lhc[:, 1:2] - 1  # Scale y to [-1, 1]
    t = 4 * lhc[:, 2:3]  # Scale t to [0, 4]
    interior_points = torch.tensor(np.hstack((x, y, t)), dtype=torch.float32)

    # Boundary and initial points
    # Create boundary points at the edges of the spatial domain for all t
    xb = np.hstack((np.ones(num_boundary // 4), -np.ones(num_boundary // 4),-np.ones(num_boundary // 4), np.ones(num_boundary // 4)))
    yb = np.hstack((np.ones(num_boundary // 4), -np.ones(num_boundary // 4),np.ones(num_boundary // 4), -np.ones(num_boundary // 4)))
    tb = np.hstack((np.random.uniform(0, 4, num_boundary//4),np.random.uniform(0, 4, num_boundary//4),np.random.uniform(0, 4, num_boundary//4),np.random.uniform(0, 4, num_boundary//4)))  # Random times for boundary points
    boundary_points = torch.tensor(np.vstack((xb, yb, tb)).T, dtype=torch.float32)
    # Initial points at t=0
    lhc_init = lhs(2, samples=num_boundary//2)
    xi = 2 * lhc_init[:, 0:1] - 1
    yi = 2 * lhc_init[:, 1:2] - 1
    ti = np.zeros(num_boundary//2).T.reshape(-1,1)
    initial_points = torch.tensor(np.hstack((xi, yi, ti)), dtype=torch.float32)
    # Combine interior, boundary, and initial points
    all_points = torch.cat((interior_points, boundary_points, initial_points), 0)
    return all_points,interior_points,boundary_points


# Custom loss function
def burgers_loss(model, all_inputs, boundary_inputs, initial_inputs):
    # Calculate predictions for all points
    all_inputs = torch.clone(all_inputs).detach().requires_grad_(True)

    u = model(all_inputs)
    u_t = torch.autograd.grad(u, all_inputs, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0][:, 2:3]
    u_x = torch.autograd.grad(u, all_inputs, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0][:, 0:1]
    u_y = torch.autograd.grad(u, all_inputs, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0][:, 1:2]
    u_xx = torch.autograd.grad(u_x, all_inputs, grad_outputs=torch.ones_like(u_x),
                               create_graph=True, retain_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, all_inputs, grad_outputs=torch.ones_like(u_y),
                               create_graph=True, retain_graph=True)[0][:, 1:2]

    # The Burgers' equation
    f_residual = u_t + u * u_x + u * u_y - (u_xx + u_yy)
    # print('f_residual',torch.mean(f_residual**2).item())
    # Boundary condition loss (Assuming the boundary condition is u=0 on the boundary)
    boundary_preds = model(boundary_inputs)
    boundary_true = exact_solution(boundary_inputs[:,0],boundary_inputs[:,1],boundary_inputs[:,2])
    boundary_condition_loss = torch.mean((boundary_preds - boundary_true) ** 2)
    # print(' boundary_condition_loss',boundary_condition_loss.item())
    # Initial condition loss (Assuming the initial condition is known and stored in initial_inputs)
    initial_preds = model(initial_inputs)
    true_initial_condition = exact_solution(initial_inputs[:,0],initial_inputs[:,1],initial_inputs[:,2])
    initial_condition_loss = torch.mean((initial_preds - true_initial_condition) ** 2)
    # print('initial_condition_loss',initial_condition_loss.item())
    # sum = torch.abs(model.weight_initial)+torch.abs(model.weight_boundary)+torch.abs(model.weight_residual)
    # init_w = (torch.abs(model.weight_initial))/sum
    # bound_w = (torch.abs(model.weight_boundary))/sum
    # res_w = (torch.abs(model.weight_residual))/sum
    # Combine all losses
    # total_loss = res_w*torch.mean(f_residual ** 2) + bound_w * boundary_condition_loss + init_w*initial_condition_loss
    total_loss = torch.mean(f_residual ** 2) + boundary_condition_loss + initial_condition_loss + torch.mean((u-exact_solution(all_inputs[:,0],all_inputs[:,1],all_inputs[:,2]))**2)
    # print('total_loss',total_loss)
    return total_loss



# L2 relative error function
def l2_relative_error(true, pred):
    return torch.norm(pred - true) / torch.norm(true)


# Training the model
def train(model, train_loader, init,bound,epochs=2000, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x = train_loader
    model.train()
    ls = []
    test_data = generate_test_data(num_test=100)
    for epoch in range(epochs):
        loss = burgers_loss(model, x,init,bound)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            # print(f'Epoch {epoch}, Loss: {loss.item()}')
            error = test_model(model, test_data)
            ls.append(error.item())
    return ls



# L2 relative error function


# Exact solution function
def exact_solution(x, y, t):
    if isinstance(x, torch.Tensor):
        x_tensor = x
        y_tensor = y
        t_tensor = t
    else:
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y)
        t_tensor = torch.from_numpy(t)
    return 1 / (1 + torch.exp((x_tensor + y_tensor - t_tensor) / 2))



# Generate test data and evaluate the model
def test_model(model, test_data):
    model.eval()
    test_points, true_u = test_data
    predicted_u = model(test_points).reshape(true_u.shape)
    error = l2_relative_error(true_u, predicted_u)
    print( error.item())
    return error

def generate_test_data(num_test=100):
    lhc_test = lhs(3, samples=num_test)
    x_test = 2 * lhc_test[:, 0:1] - 1
    y_test = 2 * lhc_test[:, 1:2] - 1
    t_test = 4 * lhc_test[:, 2:3]
    test_points = torch.tensor(np.hstack((x_test, y_test, t_test)), dtype=torch.float32)
    true_u = exact_solution(x_test, y_test, t_test)
    true_u = true_u.clone().detach().type(torch.float32).view(-1, 1)

    return test_points, true_u
def main():
    model = ChNNModel(4)
    data,init,bound = prepare_data()
    train_loader = data
    ls = train(model, train_loader,init,bound)
    print(ls)
    # Generate test data and evaluate the model
    test_data = generate_test_data(num_test=100)
    test_model(model, test_data)

def new_data(num_test=50):
    lhc_test = lhs(3, samples=num_test)
    x_test = 2 * lhc_test[:, 0:1] - 1
    y_test = 2 * lhc_test[:, 1:2] - 1
    t_test = 4 * lhc_test[:, 2:3]
    test_points = torch.tensor(np.hstack((x_test, y_test, t_test)), dtype=torch.float32)
    return test_points

if __name__ == "__main__":
    main()
