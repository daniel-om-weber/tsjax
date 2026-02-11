# Requires: pip install identibench
# Downloads benchmark dataset from the internet on first run.

# %% [markdown]
# # IdentiBench Benchmark
# Uses `create_grain_dls_from_spec` to auto-download a benchmark dataset
# and apply its defaults.

# %%
import identibench as idb

from tsjax import GRULearner, create_grain_dls_from_spec, rmse

# %%
if __name__ == "__main__":
    pipeline = create_grain_dls_from_spec(
        idb.BenchmarkRobotForward_Simulation,
        bs=32, stp_sz=1, preload=True, worker_count=2,
    )
    pipeline.n_train_batches = 300  # limit to 300 batches per epoch

    # %%
    lrn = GRULearner(pipeline, hidden_size=64, n_skip=80, metrics=[rmse])
    lrn.show_batch(n=4)

    # %%
    lrn.fit_flat_cos(n_epoch=5, lr=1e-3)

    # %%
    lrn.show_results()
