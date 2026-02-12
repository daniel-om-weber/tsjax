# %% [markdown]
# # Model and Training Options
# Compares RNN cell types, multi-layer models, loss functions,
# and shows how to inspect training history.

# %%
from pathlib import Path

from tsjax import GRULearner, RNNLearner, create_simulation_dls, normalized_mse, rmse

# %%
DATASET = Path(__file__).resolve().parent.parent / "test_data/WienerHammerstein"

pipeline = create_simulation_dls(
    u=["u"],
    y=["y"],
    dataset=DATASET,
    bs=16,
    win_sz=500,
    stp_sz=10,
    preload=True,
)

# %% [markdown]
# ## LSTM (default) vs GRU

# %%
lrn_lstm = RNNLearner(pipeline, hidden_size=64, n_skip=40, metrics=[rmse])
lrn_lstm.fit_flat_cos(n_epoch=3, lr=1e-3)

# %%
lrn_gru = GRULearner(pipeline, hidden_size=64, n_skip=40, metrics=[rmse])
lrn_gru.fit_flat_cos(n_epoch=3, lr=1e-3)

# %% [markdown]
# ## Multi-layer RNN

# %%
lrn_deep = RNNLearner(pipeline, hidden_size=64, num_layers=2, n_skip=40, metrics=[rmse])
lrn_deep.fit_flat_cos(n_epoch=3, lr=1e-3)

# %% [markdown]
# ## Custom loss function
# Default is `normalized_mae`; switch to `normalized_mse` or any
# `(pred, target, y_mean, y_std) -> scalar` callable.

# %%
lrn_mse = RNNLearner(pipeline, hidden_size=64, n_skip=40, loss_func=normalized_mse, metrics=[rmse])
lrn_mse.fit(n_epoch=3, lr=1e-3)

# %% [markdown]
# ## Training history
# `train_losses`, `valid_losses`, and `valid_metrics` are recorded per epoch.

# %%
print(f"Train losses: {lrn_lstm.train_losses}")
print(f"Valid losses: {lrn_lstm.valid_losses}")
print(f"RMSE per epoch: {lrn_lstm.valid_metrics['rmse']}")
