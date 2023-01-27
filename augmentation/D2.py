import jax.numpy as jnp
from jax import config, random, jit, vmap
from jax.lax import dot_general
config.update("jax_enable_x64", True)

from augmentation import coordinate_transforms

def transform_matrix(a11, a22, a12, J_inv):
    # perform transform a^{ij} \frac{\partial \xi{\alpha}}{\partial x^{i}} \frac{\partial \xi{\beta}}{\partial x^{j}}
    row1 = jnp.stack([a11, a12], 0)
    row2 = jnp.stack([a12, a22], 0)
    A = jnp.stack([row1, row2], 0)
    A_ = jnp.transpose(dot_general(J_inv, A, (((1,), (0,)), ((2, 3), (2, 3)))), [2, 3, 0, 1])
    A_ = jnp.transpose(dot_general(J_inv, A_, (((1,), (1,)), ((2, 3), (2, 3)))), [3, 2, 0, 1])
    # same operation with einsum
    # A_ = jnp.einsum("ijab, klab, jlab -> ikab", J_inv, J_inv, A)
    return A_[0, 0], A_[1, 1], A_[0, 1]

@jit
def elliptic_augment_sample(feature, target, key, A_list, dA_list, d2A_list, beta=1e-5, p=1.0):
    M = (A_list[0].shape[0] - 1) // 2
    keys = random.split(key, 4)
    coeff_list = [coordinate_transforms.get_coeff(key, M, beta=beta, p=p) for key in keys]

    # generating new coordinates
    c = coordinate_transforms.get_transform_2d(coeff_list, A_list)
    J_inv, J = coordinate_transforms.get_first_derivatives(coeff_list, A_list, dA_list)

    # interpolate on the new grid
    features_ = jnp.stack([coordinate_transforms.interpolate(c, feature[i]) for i in range(feature.shape[0])], 0)
    targets_ = jnp.stack([coordinate_transforms.interpolate(c, target[i]) for i in range(target.shape[0])], 0)

    new_a = transform_matrix(features_[0], features_[1], features_[2], J_inv)
    new_a = [J*a for a in new_a]
    features_ = jnp.stack(new_a + [features_[-1]*J], 0)

    return features_, targets_

@jit
def wave_augment_sample(feature, target, key, A_list, dA_list, d2A_list, beta=1e-5, p=1.0):
    M = (A_list[0].shape[0] - 1) // 2
    keys = random.split(key, 4)
    coeff_list = [coordinate_transforms.get_coeff(key, M, beta=beta, p=p) for key in keys]

    # generating new coordinates
    c = coordinate_transforms.get_transform_2d(coeff_list, A_list)
    J_inv, J = coordinate_transforms.get_first_derivatives(coeff_list, A_list, dA_list)
    d2 = coordinate_transforms.get_second_derivatives(coeff_list, A_list, dA_list, d2A_list)

    # interpolate on the new grid
    features_ = jnp.stack([coordinate_transforms.interpolate(c, feature[i]) for i in range(feature.shape[0])], 0)
    targets_ = jnp.stack([coordinate_transforms.interpolate(c, target[i]) for i in range(target.shape[0])], 0)

    new_a = transform_matrix(features_[0], features_[1], features_[2], J_inv)
    row1 = jnp.stack([new_a[0], new_a[2]], 0)
    row2 = jnp.stack([new_a[2], new_a[1]], 0)
    A = jnp.stack([row1, row2], 0)
    A = dot_general(d2, A, (((1, 2), (0, 1)), ((3, 4), (2, 3))))
    A = jnp.transpose(dot_general(J_inv, A, (((1,), (2,)), ((2, 3), (0, 1)))), [2, 0, 1])
    v = jnp.transpose(dot_general(J_inv, features_[3:5], (((1,), (0,)), ((2, 3), (1, 2)))), [2, 0, 1]) - A
    features_ = jnp.stack(list(new_a) + [v[0], v[1]] + [features_[-1]], 0)

    return features_, targets_

@jit
def convection_diffusion_augment_sample(feature, target, key, A_list, dA_list, d2A_list, beta=1e-5, p=1.0):
    M = (A_list[0].shape[0] - 1) // 2
    keys = random.split(key, 4)
    coeff_list = [coordinate_transforms.get_coeff(key, M, beta=beta, p=p) for key in keys]

    # generating new coordinates
    c = coordinate_transforms.get_transform_2d(coeff_list, A_list)
    J_inv, J = coordinate_transforms.get_first_derivatives(coeff_list, A_list, dA_list)
    d2 = coordinate_transforms.get_second_derivatives(coeff_list, A_list, dA_list, d2A_list)

    # interpolate on the new grid
    features_ = jnp.stack([coordinate_transforms.interpolate(c, feature[i]) for i in range(feature.shape[0])], 0)
    targets_ = jnp.stack([coordinate_transforms.interpolate(c, target[i]) for i in range(target.shape[0])], 0)

    new_a = transform_matrix(features_[0], features_[1], features_[2], J_inv)
    row1 = jnp.stack([new_a[0], new_a[2]], 0)
    row2 = jnp.stack([new_a[2], new_a[1]], 0)
    A = jnp.stack([row1, row2], 0)
    d2 = dot_general(d2, J_inv, (((0, 1), (1, 0)), ((3, 4), (2, 3)))) # 100, 100, 2
    A = jnp.transpose(dot_general(d2, A, (((2,), (1,)), ((0, 1), (2, 3)))), [2, 0, 1])
    v = jnp.transpose(dot_general(J_inv, features_[3:5], (((1,), (0,)), ((2, 3), (1, 2)))), [2, 0, 1]) - A

    features_ = jnp.stack(list(new_a) + [v[0], v[1]] + [features_[-1] * J], 0)

    return features_, targets_ * J.reshape(1, 100, 100)

def augment_dataset(features, targets, key, augment_sample, augmentation_factor, M=5, beta=1e-5, p=1.0):
    N = 100
    xi = jnp.linspace(0, 1, N)
    eta = jnp.linspace(0, 1, N)
    xi_, eta_ = jnp.meshgrid(xi, eta)
    x_old = jnp.stack([xi_, eta_], 0)

    A_list = [coordinate_transforms.build_matrix(xi, M, p=p), coordinate_transforms.build_matrix(eta, M, p=p)]
    dA_list = [coordinate_transforms.build_d_matrix(xi, M, p=p), coordinate_transforms.build_d_matrix(eta, M, p=p)]
    d2A_list = [coordinate_transforms.build_d2_matrix(xi, M, p=p), coordinate_transforms.build_d2_matrix(eta, M, p=p)]
    augment_ = lambda a, b, c, d, e, f: augment_sample(a, b, c, d, e, f, beta=beta, p=p)

    F, T = [features,], [targets,]
    for _ in range(augmentation_factor):
        keys = random.split(key, features.shape[0]+1)
        key = keys[-1]
        f_, t_ = vmap(augment_, in_axes=(0, 0, 0, None, None, None))(features, targets, keys[:-1], A_list, dA_list, d2A_list)
        F.append(jnp.stack(f_, 0))
        T.append(jnp.stack(t_, 0))
    F = jnp.vstack(F)
    T = jnp.vstack(T)
    return F, T

def elliptic_augmentation_I(features, targets, key, augmentation_factor):
    # relative error >= 1
    return augment_dataset(features, targets, key, elliptic_augment_sample, augmentation_factor, M=1, beta=1e-5, p=1.0)

def elliptic_augmentation_II(features, targets, key, augmentation_factor):
    # relative error ~ .5
    return augment_dataset(features, targets, key, elliptic_augment_sample, augmentation_factor, M=5, beta=1e-5, p=1.0)

def elliptic_augmentation_III(features, targets, key, augmentation_factor):
    # relative error < .5
    return augment_dataset(features, targets, key, elliptic_augment_sample, augmentation_factor, M=10, beta=1e-5, p=1.0)

def convection_diffusion_augmentation_I(features, targets, key, augmentation_factor):
    # relative error: convection ~ 5-10, the rest ~ 1
    return augment_dataset(features, targets, key, convection_diffusion_augment_sample, augmentation_factor, M=1, beta=0.25, p=2.0)

def convection_diffusion_augmentation_II(features, targets, key, augmentation_factor):
    # relative error: convection ~ 1, the rest ~ .1
    return augment_dataset(features, targets, key, convection_diffusion_augment_sample, augmentation_factor, M=5, beta=5, p=2.0)

def convection_diffusion_augmentation_III(features, targets, key, augmentation_factor):
    # relative error: convection ~ .5, the rest ~ .05
    return augment_dataset(features, targets, key, convection_diffusion_augment_sample, augmentation_factor, M=10, beta=1e-5)

def wave_augmentation_I(features, targets, key, augmentation_factor):
    # relative error: convection > 1, the rest ~ 1
    return augment_dataset(features, targets, key, wave_augment_sample, augmentation_factor, M=1, beta=0.25, p=2.0)

def wave_augmentation_II(features, targets, key, augmentation_factor):
    # relative error: convection ~ 1, the rest ~ .1
    return augment_dataset(features, targets, key, wave_augment_sample, augmentation_factor, M=5, beta=5, p=2.0)

def wave_augmentation_III(features, targets, key, augmentation_factor):
    # relative error: convection ~ .5, the rest ~ .05
    return augment_dataset(features, targets, key, wave_augment_sample, augmentation_factor, M=10, beta=15, p=2.0)
