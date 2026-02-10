# %% [markdown]
# # Minimal tsjax Example
# JAX equivalent of `00_minimal_example.ipynb`.
# Uses the Wiener-Hammerstein dataset (u → y).

# %%
from pathlib import Path

from tsjax import RNNLearner, create_simulation_dls, rmse

# %%
_root = Path(__file__).resolve().parent.parent
pipeline = create_simulation_dls(
    u=['u'], y=['y'],
    dataset=_root / 'test_data/WienerHammerstein',
    bs=16, win_sz=500, stp_sz=10,
    preload=True
)

# %%
lrn = RNNLearner(pipeline, rnn_type='lstm', hidden_size=64, n_skip=10, metrics=[rmse])
lrn.fit_flat_cos(n_epoch=1, lr=1e-3)
