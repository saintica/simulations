from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp  # Replace mlx.core with jax.numpy
from jax import grad  # Import grad separately from jax

from matplotlib import rcParams
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_rgb, LightSource, PowerNorm

# Utility functions and definitions
def norm(v, axis=-1, keepdims=False, eps=0.0):
    return jnp.sqrt((v*v).sum(axis, keepdims=keepdims).clip(eps))

def normalize(v, axis=-1, eps=1e-20):
    return v / norm(v, axis, keepdims=True, eps=eps)

def peak_f(x, mu, sigma, a):
    return a * jnp.exp(-((x - mu) / sigma)**2)

Params = namedtuple('Params', 'mu_k sigma_k w_k mu_g sigma_g c_rep')
Fields = namedtuple('Fields', 'U G R E')

class ParticleLenia():
    def __init__(self, p: Params, initial_coordinates):
        self.params = p
        self.points0 = initial_coordinates
        self.history = [initial_coordinates]
        
    def fields_f(self, points, x):
        r = jnp.sqrt(jnp.clip(jnp.square(x-points).sum(axis=-1), 1e-10, None))
        U = peak_f(r, self.params.mu_k, self.params.sigma_k, self.params.w_k).sum()
        G = peak_f(U, self.params.mu_g, self.params.sigma_g, 1.)
        R = (self.params.c_rep/2 * jnp.clip(1.0-r, 0.0, None)**2).sum()
        return Fields(U, G, R, E=R-G)

    def field_x(self, points):
        return jnp.vectorize(lambda x : self.fields_f(points, x))

    def motion_f(self, points):
        grad_E = grad(lambda x : self.fields_f(points, x).E)
        return jnp.vectorize(grad_E)

    def odeint_euler(self, dt, n):
        def step_f(x):
            x = x - dt * self.motion_f(x)(x)
            return x
        
        print(f"Starting simulation with params {self.params}")
        for i in range(1, n):
            self.history.append(step_f(self.history[-1]))
            if i % 2000 == 0:
                print(f"Step {i}")
        print(f"Ending simulation with params {self.params} \n")

dt = 0.1
n_particles = 400
params = Params(mu_k=2.75, sigma_k=1.25, w_k=0.02625, mu_g=0.7, sigma_g=0.16666666666666669, c_rep=1.0)
points0 = np.random.uniform(-6., 6., [n_particles, 2])  # Use numpy for initial random values
pl = ParticleLenia(params, points0)

pl.odeint_euler(dt, 10000)

size = [-22, 22, 800]
index = -1
grid_x, grid_y = np.meshgrid(np.linspace(*size), np.linspace(*size))
grid = np.array([grid_x.reshape(-1), grid_y.reshape(-1)]).T
grid = jnp.array(grid)

E_field = pl.field_x(pl.history[index])(grid)[3].reshape((size[-1], -1))

fig, ax = plt.subplots(figsize=(8, 8))
fig.set_facecolor("#f4f0e8")
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax.set_facecolor("#f4f0e8") 
ls = LightSource(azdeg=315, altdeg=45)
ax.scatter(pl.history[index][:,0], pl.history[index][:,1], edgecolors="#383b3e", c="#f4f0e8", s=10, linewidths=1)
E_field_shaded = ls.shade(np.array(E_field), cmap=plt.cm.RdGy, vert_exag=4, blend_mode='hsv', vmin=-0.7, vmax=0.5)
ax.imshow(E_field_shaded, extent=(size[0], size[1], size[0], size[1]),
          origin="lower", interpolation="bicubic")

ax.set_aspect('equal', 'box')
ax.set_axis_off()
ax.set_xlim([-22, 22])
ax.set_ylim([-22, 22])
plt.show()
