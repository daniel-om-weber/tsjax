# %% [markdown]
# # Minimal TSFast Example
# PyTorch/fastai equivalent of `00_minimal_example_jax.py`.
# Uses the Wiener-Hammerstein dataset (u → y).

# %%
from pathlib import Path

from tsfast.basics import RNNLearner, create_dls

# %%
_root = Path(__file__).resolve().parent.parent
dls = create_dls(
    u=['u'], y=['y'],
    dataset=_root / 'test_data/WienerHammerstein',
    bs=16, win_sz=500, stp_sz=10
).cpu()

# %%
lrn = RNNLearner(dls, rnn_type='lstm', hidden_size=64, n_skip=10)
lrn.fit_flat_cos(n_epoch=3, lr=1e-3)

# %%
