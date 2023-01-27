import jax.numpy as jnp
from jax import config, random, vmap, jit
config.update("jax_enable_x64", True)

from augmentation import coordinate_transforms

def augment_dataset(features, targets, key, augment_sample, augmentation_factor, M=5, beta=1e-5, p=1.0):
    N = 100
    xi = jnp.linspace(0, 1, N)
    A = coordinate_transforms.build_matrix(xi, M)
    dA = coordinate_transforms.build_d_matrix(xi, M)
    d2A = coordinate_transforms.build_d2_matrix(xi, M)
    augment_ = lambda a, b, c, d, e, f: augment_sample(a, b, c, d, e, f, beta=beta, p=p)

    F, T = [features,], [targets,]
    for _ in range(augmentation_factor):
        keys = random.split(key, features.shape[0]+1)
        key = keys[-1]
        f_, t_ = vmap(augment_, in_axes=(0, 0, 0, None, None, None))(features, targets, keys[:-1], A, dA, d2A)
        F.append(jnp.stack(f_, 0))
        T.append(jnp.stack(t_, 0))
    F = jnp.vstack(F)
    T = jnp.vstack(T)
    return F, T

@jit
def eliptic_augment_sample(features, target, key, A, dA, d2A, beta=1e-5, p=1.0):
    M = (A.shape[0] - 1) // 2
    coeff = coordinate_transforms.get_coeff(key, M, beta=beta, p=p)
    x = coordinate_transforms.get_transform_1d(coeff, A)
    dx = coordinate_transforms.get_transform_1d(coeff, dA)
    a_ = coordinate_transforms.interpolate(x, features[0]) / dx[0]
    f_ = coordinate_transforms.interpolate(x, features[1]) * dx[0]
    t_ = coordinate_transforms.interpolate(x, target[0])
    return jnp.stack([a_, f_], 0), jnp.expand_dims(t_, 0)

@jit
def convection_diffusion_augment_sample(features, target, key, A, dA, d2A, beta=1e-5, p=1.0):
    M = (A.shape[0] - 1) // 2
    coeff = coordinate_transforms.get_coeff(key, M, beta=beta, p=p)
    x = coordinate_transforms.get_transform_1d(coeff, A)
    dx = coordinate_transforms.get_transform_1d(coeff, dA)
    d2x = coordinate_transforms.get_transform_1d(coeff, d2A)
    a_ = coordinate_transforms.interpolate(x, features[0]) / dx[0]**2
    v_ = (coordinate_transforms.interpolate(x, features[1]) + a_ * d2x[0]) / dx[0]
    init_ = coordinate_transforms.interpolate(x, features[2]) * dx[0]
    t_ = coordinate_transforms.interpolate(x, target[0]) * dx[0]
    return jnp.stack([a_, v_, init_], 0), jnp.expand_dims(t_, 0)

@jit
def wave_augment_sample(features, target, key, A, dA, d2A, beta=1e-5, p=1.0):
    M = (A.shape[0] - 1) // 2
    coeff = coordinate_transforms.get_coeff(key, M, beta=beta, p=p)
    x = coordinate_transforms.get_transform_1d(coeff, A)
    dx = coordinate_transforms.get_transform_1d(coeff, dA)
    d2x = coordinate_transforms.get_transform_1d(coeff, d2A)
    phi = coordinate_transforms.interpolate(x, features[0])
    c = coordinate_transforms.interpolate(x, features[1]) / dx[0]**2
    d = coordinate_transforms.interpolate(x, features[2]) / dx[0] - c * d2x[0] / dx[0]
    e = coordinate_transforms.interpolate(x, features[3])
    features_ = jnp.stack([phi, c, d, e])
    target_ = jnp.expand_dims(target[0], 0)
    return features_, target_
