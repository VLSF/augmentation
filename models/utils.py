import jax.numpy as jnp

from jax import config, random
config.update("jax_enable_x64", True)

def standard_data_loader(train_data, key, chunk_size):
    train_features, train_targets = train_data
    n = random.permutation(key, jnp.arange(train_features.shape[0], dtype=int))
    train_features, train_targets = train_features[n], train_targets[n]
    chunks = [*range(0, train_features.shape[0] + 1, chunk_size)]
    if chunks[-1] < train_features.shape[0]:
        chunks.append(train_features.shape[0])
    for i, j in zip(chunks[:-1], chunks[1:]):
        yield train_features[i:j], train_targets[i:j]
        
def training_loop(train_data, test_data, model, opt_state, optimizer, Save_each_N, make_step, compute_loss, chunk_size, N_epoch, key):
    train_losses, test_losses = [], []
    for j in range(N_epoch):
        key = random.split(key)[0]
        data_loader = standard_data_loader(train_data, key, chunk_size)
        for features, targets in data_loader:
            train_loss, model, opt_state = make_step(model, features, targets, optimizer, opt_state)
        if (j+1) % Save_each_N == 0:
            train_loss = compute_loss(model, train_data[0], train_data[1])
            train_losses.append(train_loss.item())
            test_loss = compute_loss(model, test_data[0], test_data[1])
            test_losses.append(test_loss.item())
    return model, opt_state, train_losses, test_losses
