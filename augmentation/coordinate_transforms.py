import jax.numpy as jnp

from jax.scipy.ndimage import map_coordinates
from jax import config, random
from scipy.interpolate import griddata

config.update("jax_enable_x64", True)

def build_matrix(xi, N, p=1.0):
    S = [xi,]
    S += [jnp.sin(2*jnp.pi*k*xi)/(2*jnp.pi*k)**p for k in range(1, N+1)]
    S += [(1 - jnp.cos(2*jnp.pi*k*xi))/(2*jnp.pi*k)**p for k in range(1, N+1)]
    return jnp.stack(S, 0)

def build_d_matrix(xi, N, p=1.0):
    S = [jnp.ones_like(xi),]
    S += [jnp.cos(2*jnp.pi*k*xi)/(2*jnp.pi*k)**(p-1) for k in range(1, N+1)]
    S += [jnp.sin(2*jnp.pi*k*xi)/(2*jnp.pi*k)**(p-1) for k in range(1, N+1)]
    return jnp.stack(S, 0)

def build_d2_matrix(xi, N, p=1.0):
    S = [jnp.zeros_like(xi),]
    S += [-jnp.sin(2*jnp.pi*k*xi)/(2*jnp.pi*k)**(p-2) for k in range(1, N+1)]
    S += [jnp.cos(2*jnp.pi*k*xi)/(2*jnp.pi*k)**(p-2) for k in range(1, N+1)]
    return jnp.stack(S, 0)

def get_coeff(key, N, beta=1e-2, p=1.0):
    w = jnp.array([1/(2*jnp.pi*k)**(p-1) for k in range(1, N+1)]*2)
    coeff = random.normal(key, (2*N,))
    coeff = coeff / (sum(abs(coeff*w)) + beta)
    coeff = jnp.hstack([1, coeff])
    return coeff

def get_transform(coeff, A):
    return jnp.sum(coeff.reshape(-1, 1)*A, 0)

def get_transform_2d(coeff_list, A_list):
    A_xi, A_eta = A_list
    coeff1, coeff2, coeff3, coeff4 = coeff_list
    x = jnp.outer(1 - A_eta[0], get_transform(coeff1, A_xi)) + jnp.outer(A_eta[0], get_transform(coeff2, A_xi))
    y = jnp.outer(get_transform(coeff3, A_eta), 1 - A_xi[0]) + jnp.outer(get_transform(coeff4, A_eta), A_xi[0])
    return jnp.stack([x, y], 0)

def get_first_derivatives(coeff_list, A_list, dA_list):
    A_xi, A_eta = A_list
    dA_xi, dA_eta = dA_list
    coeff1, coeff2, coeff3, coeff4 = coeff_list
    a = jnp.outer(get_transform(coeff3, dA_eta), 1 - A_xi[0]) + jnp.outer(get_transform(coeff4, dA_eta), A_xi[0])
    b = jnp.outer(dA_eta[0], get_transform(coeff1, A_xi) - get_transform(coeff2, A_xi))
    c = jnp.outer(get_transform(coeff3, A_eta) - get_transform(coeff4, A_eta), dA_xi[0])
    d = jnp.outer(1 - A_eta[0], get_transform(coeff1, dA_xi)) + jnp.outer(A_eta[0], get_transform(coeff2, dA_xi))
    J = a*d - b*c
    J_inv = jnp.stack([jnp.stack([a, b], 0), jnp.stack([c, d], 0)], 0) / J.reshape(1, 1, J.shape[0], J.shape[1])
    return J_inv, J

def get_second_derivatives(coeff_list, A_list, dA_list, d2A_list):
    A_xi, A_eta = A_list
    dA_xi, dA_eta = dA_list
    d2A_xi, d2A_eta = d2A_list
    coeff1, coeff2, coeff3, coeff4 = coeff_list
    a = jnp.outer(1 - A_eta[0], get_transform(coeff1, d2A_xi)) + jnp.outer(A_eta[0], get_transform(coeff2, d2A_xi))
    b = jnp.outer(dA_eta[0], get_transform(coeff2, dA_xi) - get_transform(coeff1, dA_xi))
    c = jnp.outer(get_transform(coeff3, d2A_eta), 1 - A_xi[0]) + jnp.outer(get_transform(coeff4, d2A_eta), A_xi[0])
    d = jnp.outer(get_transform(coeff4, dA_eta) - get_transform(coeff3, dA_eta), dA_xi[0])
    x_ = jnp.stack([jnp.stack([a, b], 0), jnp.stack([b, a*0], 0)], 0)
    y_ = jnp.stack([jnp.stack([c*0, d], 0), jnp.stack([d, c], 0)], 0)
    return jnp.stack([x_, y_], 0)

def interpolate(x_new, v_old):
    # ok only for the NxN grid
    return map_coordinates(v_old.T, x_new*(v_old.shape[0]-1), 1, mode="nearest")
