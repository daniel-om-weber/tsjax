# %% [markdown]
# # Quickstart
# Minimal tsjax example using the Wiener-Hammerstein dataset (u → y).

# %%
from pathlib import Path

from tsjax import RNNLearner, create_simulation_dls, rmse

# %%
DATASET = Path(__file__).resolve().parent.parent / "test_data/WienerHammerstein"

pipeline = create_simulation_dls(
    u=["u"], y=["y"],
    dataset=DATASET,
    bs=16, win_sz=500, stp_sz=10,
    preload=True,
)

# %%
lrn = RNNLearner(pipeline, hidden_size=64, n_skip=40, metrics=[rmse])
lrn.fit_flat_cos(n_epoch=3, lr=1e-3)

# %%
lrn.show_results(n=2)
