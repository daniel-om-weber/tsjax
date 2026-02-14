# %% [markdown]
# # GRU vs MinGRU vs S5
# Compares accuracy and training speed of three sequence models
# on the Wiener-Hammerstein benchmark (u → y).

# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt
from flax import nnx

from tsjax import (
    S5,
    Denormalize,
    GRULearner,
    Learner,
    MinGRU,
    Normalize,
    NormalizedModel,
    create_simulation_dls,
    rmse,
)

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
# ## Shared settings

# %%
HIDDEN = 64
N_SKIP = 40
N_EPOCH = 30
LR = 1e-3
SEED = 42
METRICS = [rmse]

# %% [markdown]
# ## Helper: create a Learner for any sequence model


# %%
def make_learner(model_cls, pipeline, **model_kw):
    """Wrap a sequence model with normalization and return a Learner."""
    u_stats = pipeline.stats()[pipeline.input_keys[0]]
    y_stats = pipeline.stats()[pipeline.target_keys[0]]
    in_sz = len(u_stats.mean)
    out_sz = len(y_stats.mean)

    raw = model_cls(input_size=in_sz, output_size=out_sz, rngs=nnx.Rngs(SEED), **model_kw)
    model = NormalizedModel(
        raw,
        norm_in=Normalize(in_sz, u_stats.mean, u_stats.std),
        norm_out=Denormalize(out_sz, y_stats.mean, y_stats.std),
    )
    return Learner(model, pipeline, n_skip=N_SKIP, metrics=METRICS)


# %% [markdown]
# ## Train all three models

# %%
print("=" * 60)
print("GRU")
print("=" * 60)
lrn_gru = GRULearner(
    pipeline, hidden_size=HIDDEN, num_layers=2, n_skip=N_SKIP, seed=SEED, metrics=METRICS
)
t0 = time.perf_counter()
lrn_gru.fit_flat_cos(N_EPOCH, lr=LR)
t_gru = time.perf_counter() - t0

# %%
print("=" * 60)
print("MinGRU")
print("=" * 60)
lrn_mingru = make_learner(MinGRU, pipeline, hidden_size=HIDDEN, num_layers=2)
t0 = time.perf_counter()
lrn_mingru.fit_flat_cos(N_EPOCH, lr=LR)
t_mingru = time.perf_counter() - t0

# %%
print("=" * 60)
print("S5")
print("=" * 60)
lrn_s5 = make_learner(S5, pipeline, hidden_size=HIDDEN, state_size=HIDDEN, num_layers=5)
t0 = time.perf_counter()
lrn_s5.fit_flat_cos(N_EPOCH, lr=LR)
t_s5 = time.perf_counter() - t0

# %% [markdown]
# ## Summary table

# %%
results = {
    "GRU": (lrn_gru, t_gru),
    "MinGRU": (lrn_mingru, t_mingru),
    "S5": (lrn_s5, t_s5),
}

print(f"\n{'Model':<10} {'Final RMSE':>12} {'Valid Loss':>12} {'Time (s)':>10}")
print("-" * 48)
for name, (lrn, t) in results.items():
    final_rmse = lrn.valid_metrics["rmse"][-1]
    final_vloss = lrn.valid_losses[-1]
    print(f"{name:<10} {final_rmse:>12.6f} {final_vloss:>12.6f} {t:>10.1f}")

# %% [markdown]
# ## Learning curves

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

for name, (lrn, _) in results.items():
    epochs = range(1, len(lrn.valid_losses) + 1)
    ax1.plot(epochs, lrn.valid_losses, label=name, marker="o", markersize=3)
    ax2.plot(epochs, lrn.valid_metrics["rmse"], label=name, marker="o", markersize=3)

ax1.set(xlabel="Epoch", ylabel="Valid Loss (norm. MAE)", title="Validation Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set(xlabel="Epoch", ylabel="RMSE", title="Validation RMSE")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Predictions on validation data

# %%
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for ax, (name, (lrn, _)) in zip(axes, results.items()):
    lrn.show_results(n=1)
    plt.suptitle(name)
    plt.show()
