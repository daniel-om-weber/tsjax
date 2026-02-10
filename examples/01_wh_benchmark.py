# %% [markdown]
# # Wiener-Hammerstein via IdentiBench
# Uses `create_grain_dls_from_spec` to auto-download the dataset
# and apply benchmark defaults.

# %%
import identibench as idb

from tsjax import RNNLearner, create_grain_dls_from_spec, rmse

# %%
pipeline = create_grain_dls_from_spec(idb.BenchmarkWH_Simulation, bs=16)
pipeline.train = pipeline.train.slice(slice(200))  # first 200 batches per epoch  

# %%
lrn = RNNLearner(pipeline, rnn_type='gru', hidden_size=64, n_skip=10, metrics=[rmse])
lrn.fit_flat_cos(n_epoch=5, lr=1e-3)

# %%
