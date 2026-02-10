"""JAX-based training library for time series system identification."""

__version__ = "0.1.0"

from .factory import GRULearner as GRULearner
from .factory import RNNLearner as RNNLearner
from .factory import create_gru as create_gru
from .factory import create_rnn as create_rnn
from .hdf5_index import HDF5MmapIndex as HDF5MmapIndex
from .hdf5_index import SignalInfo as SignalInfo
from .learner import Learner as Learner
from .models import GRU as GRU
from .models import RNN as RNN
from .pipeline import GrainPipeline as GrainPipeline
from .pipeline import create_grain_dls as create_grain_dls
from .sources import FullSequenceSource as FullSequenceSource
from .sources import WindowedHDF5Source as WindowedHDF5Source
from .stats import compute_norm_stats as compute_norm_stats
from .train import normalized_mae as normalized_mae
from .train import normalized_mse as normalized_mse
from .train import rmse as rmse
