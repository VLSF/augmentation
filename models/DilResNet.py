import jax.numpy as jnp
import equinox as eqx

from typing import Callable
from jax import config, random, vmap
from jax.nn import relu
config.update("jax_enable_x64", True)

class DilatedConvBlock(eqx.Module):
    convolutions: list
    activation: Callable

    def __init__(self, channels, dilations_D, kernel_sizes_D, key, activation=relu):
        kernel_sizes_D = [[k if k%2 == 1 else k+1 for k in kernel_sizes] for kernel_sizes in kernel_sizes_D]
        paddings_D = [[d*(k//2) for d, k in zip(dilations, kernel_sizes)] for dilations, kernel_sizes in zip(dilations_D, kernel_sizes_D)]
        keys = random.split(key, len(channels))
        D = len(kernel_sizes_D[0])
        self.convolutions = [eqx.nn.Conv(num_spatial_dims=D, in_channels=f_i, out_channels=f_o, dilation=d, kernel_size=k, padding=p, key=key) for f_i, f_o, d, k, p, key in zip(channels[:-1], channels[1:], dilations_D, kernel_sizes_D, paddings_D, keys)]
        self.activation = activation

    def __call__(self, x):
        for conv in self.convolutions[:-1]:
            x = self.activation(conv(x))
        x = self.convolutions[-1](x)
        return x

    def linear_call(self, x):
        for conv in self.convolutions:
            x = conv(x)
        return x

class DilatedResNet(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    processor: list
    activation: Callable

    def __init__(self, key, channels, n_cells, activation=relu, kernel_size=3, D=1):
        in_channels, processor_channels, out_channels = channels
        keys = random.split(key, 3)
        self.encoder = DilatedConvBlock([in_channels, processor_channels], [[1,]*D,], [[kernel_size,]*D,], keys[0])
        self.decoder = DilatedConvBlock([processor_channels, out_channels], [[1,]*D,], [[kernel_size,]*D,], keys[1])
        keys = random.split(keys[2], n_cells)
        channels_ = [processor_channels,]*8
        dilations = [[1,]*D, [2,]*D, [4,]*D, [8,]*D, [4,]*D, [2,]*D, [1,]*D]
        kernel_sizes = [[kernel_size,]*D,]*7
        self.processor = [DilatedConvBlock(channels_, dilations, kernel_sizes, key, activation=activation) for key in keys]
        self.activation = activation

    def __call__(self, x):
        x = self.encoder(x)
        for pr in self.processor:
            x = self.activation(pr(x)) + x
        x = self.decoder(x)
        return x

    def linear_call(self, x):
        x = self.encoder(x)
        for pr in self.processor:
            x = pr(x) + x
        x = self.decoder(x)
        return x

def compute_loss(model, input, target):
    output = vmap(lambda z: model(z), in_axes=(0,))(input)
    l = jnp.mean(jnp.linalg.norm((output - target).reshape(input.shape[0], -1), axis=1)**2)
    return l

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

@eqx.filter_jit
def make_step(model, input, target, optim, opt_state):
    loss, grads = compute_loss_and_grads(model, input, target)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state
