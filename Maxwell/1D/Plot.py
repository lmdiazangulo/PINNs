# %%
import torch
from main import PINN, x_min, x_max, t_min, t_max
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pinn = PINN()
pinn.net.load_state_dict(torch.load("./Wave/1D/weight.pt"))

x = np.arange(x_min, x_max, 0.01)
t = np.arange(t_min, t_max, 0.01)

x_mesh, t_mesh = np.meshgrid(x, t)

X = x_mesh.reshape((-1, 1))
T = t_mesh.reshape((-1, 1))

x_ts = torch.tensor(X, dtype=torch.float32).to(device)
t_ts = torch.tensor(T, dtype=torch.float32).to(device)
xt_ts = torch.hstack([x_ts, t_ts])
with torch.no_grad():
    u_pred = pinn.net(xt_ts).cpu().numpy().reshape(t_mesh.shape).T

# %%
fig, axs = plt.subplots()
c1 = axs.pcolormesh(t_mesh, x_mesh, u_pred.T, shading='auto', cmap='rainbow')
axs.set_title('Predicted E(t, x)')
axs.set_ylabel('x')
fig.colorbar(c1, ax=axs, label='Predicted')
plt.show()
# %%
