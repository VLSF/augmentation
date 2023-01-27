import jax.numpy as jnp

import equinox as eqx
import optax

from jax import jit, vmap, grad, config, random
from jax.lax import dot_general
from jax.nn import relu
from typing import Callable
config.update("jax_enable_x64", True)

def apply_operators(data, operators):
    for n, O in enumerate(operators):
        transposition = [*range(1, n+1)] + [0] + [*range(n+1, len(data.shape))]
        data = jnp.transpose(dot_general(O, data, (((1,), (n)), ((), ()))), transposition)
    return data

class tinyMLP(eqx.Module):
    A: list
    b: list
    activation: Callable

    def __init__(self, shapes, key, activation=relu):
        keys = random.split(key)
        keys_A = random.split(keys[0], (len(shapes)-1)*(len(shapes[0]))).reshape(-1, len(shapes[0]), 2)
        keys_b = random.split(keys[1], (len(shapes)-1))
        self.A = [[random.normal(k, (s_out, s_in)) / jnp.sqrt(s_out*s_in) for k, s_in, s_out in zip(key, shape_in, shape_out)] for key, shape_in, shape_out in zip(keys_A, shapes[:-1], shapes[1:])]
        self.b = [random.normal(key, [shape[0]] + [1,]*(len(shape)-1)) for key, shape in zip(keys_b, shapes[1:])]
        self.activation = activation

    def __call__(self, x):
        for A, b in zip(self.A[:-1], self.b[:-1]):
            x = self.activation(apply_operators(x, A) + b)
        x = apply_operators(x, self.A[-1]) + self.b[-1]
        return x

def compute_loss(model, input, target):
    output = vmap(lambda z: model(z), in_axes=(0,))(input)
    l = jnp.mean(jnp.linalg.norm((output - target).reshape(input.shape[0], -1), axis=1)**2)
    return l

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

def good_leaf(leaf):
    if eqx.is_array(leaf):
        return leaf.dtype != jnp.int64
    else:
        return eqx.is_array(leaf)

@eqx.filter_jit
def make_step(model, input, target, optim, opt_state):
    loss, grads = compute_loss_and_grads(model, input, target)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, good_leaf))
    model = eqx.apply_updates(model, updates)
    return loss, model,
