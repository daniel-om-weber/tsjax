"""Quaternion algebra, losses, metrics, and data transforms.

Quaternion convention: ``[w, x, y, z]`` with *w* as the scalar component.
"""

from ._algebra import inclination_angle as inclination_angle
from ._algebra import pitch_angle as pitch_angle
from ._algebra import quat_conjugate as quat_conjugate
from ._algebra import quat_diff as quat_diff
from ._algebra import quat_multiply as quat_multiply
from ._algebra import quat_normalize as quat_normalize
from ._algebra import quat_relative as quat_relative
from ._algebra import relative_angle as relative_angle
from ._algebra import roll_angle as roll_angle
from ._algebra import rot_vec as rot_vec
from .losses import abs_inclination as abs_inclination
from .losses import abs_rel_angle as abs_rel_angle
from .losses import inclination_loss as inclination_loss
from .losses import inclination_loss_abs as inclination_loss_abs
from .losses import mean_inclination_deg as mean_inclination_deg
from .losses import mean_rel_angle_deg as mean_rel_angle_deg
from .losses import ms_inclination as ms_inclination
from .losses import ms_rel_angle as ms_rel_angle
from .losses import nan_safe as nan_safe
from .losses import rms_inclination as rms_inclination
from .losses import rms_inclination_deg as rms_inclination_deg
from .losses import rms_pitch_deg as rms_pitch_deg
from .losses import rms_rel_angle_deg as rms_rel_angle_deg
from .losses import rms_roll_deg as rms_roll_deg
from .losses import smooth_inclination as smooth_inclination
from .transforms import quat_interp as quat_interp
from .transforms import quaternion_augmentation as quaternion_augmentation
from .viz import plot_quaternion_results as plot_quaternion_results
