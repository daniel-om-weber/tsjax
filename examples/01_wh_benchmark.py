# %% [markdown]
# # Wiener-Hammerstein via IdentiBench
# Uses `create_grain_dls_from_spec` to auto-download the dataset
# and apply benchmark defaults.

# %%
import identibench as idb

from tsjax import GRULearner, create_grain_dls_from_spec, rmse

# %%
if __name__ == "__main__":
    pipeline = create_grain_dls_from_spec(idb.BenchmarkRobotForward_Simulation, bs=32, stp_sz=1,preload=True, worker_count=0)
    pipeline.n_train_batches = 300  # limit to 300 batches per epoch

    # %%
    lrn = GRULearner(pipeline, hidden_size=64, n_skip=10, metrics=[rmse])
    lrn.fit_flat_cos(n_epoch=5, lr=1e-3)

# %%
