# Case1/magnetic_module.py
"""
简单的磁场 + 磁力矩模块，适配 Elastica-RL-control 里的 elastica 版本（Python 3.7）。
提供：
- BaseMagneticField / ConstantMagneticField / SingleModeOscillatingMagneticField
- MagneticForces：作为 external_forces 加到 CosseratRod 上
"""

from typing import Union

import numpy as np
from elastica.external_forces import NoForces
from elastica.rod.cosserat_rod import CosseratRod
from elastica._linalg import _batch_cross, _batch_matvec, _batch_norm


# ========= 一点小工具：时间 ramp =========
def compute_ramp_factor(time: float,
                        ramp_interval: float,
                        start_time: float,
                        end_time: float) -> float:
    """
    简单线性 ramp：
    - t < start_time                -> 0
    - start_time ~ start_time+Δt    -> 线性从 0 到 1
    - start_time+Δt ~ end_time      -> 1
    - t > end_time                  -> 0
    """
    t = float(time)

    if t < start_time:
        return 0.0

    if ramp_interval > 0.0 and t < start_time + ramp_interval:
        return (t - start_time) / ramp_interval

    if t <= end_time:
        return 1.0

    return 0.0


# ========= 磁场基类和两种具体形式 =========
class BaseMagneticField(object):
    """所有磁场类的基类，必须实现 value(time)。"""

    def value(self, time: np.float64 = 0.0) -> np.ndarray:
        raise NotImplementedError


class ConstantMagneticField(BaseMagneticField):
    """
    常值磁场：B(t) = factor(t) * B0
    B0: shape (3,)
    """

    def __init__(self,
                 magnetic_field_amplitude: np.ndarray,
                 ramp_interval: float,
                 start_time: float,
                 end_time: float):
        self.magnetic_field_amplitude = np.asarray(magnetic_field_amplitude,
                                                   dtype=float).reshape(3,)
        self.ramp_interval = float(ramp_interval)
        self.start_time = float(start_time)
        self.end_time = float(end_time)

    def value(self, time: np.float64 = 0.0) -> np.ndarray:
        factor = compute_ramp_factor(time,
                                     self.ramp_interval,
                                     self.start_time,
                                     self.end_time)
        return factor * self.magnetic_field_amplitude


class SingleModeOscillatingMagneticField(BaseMagneticField):
    """
    单模正弦磁场：
    B(t) = factor(t) * A * sin(ω t + φ)
    其中 A, ω, φ 都是 shape (3,) 的向量。
    """

    def __init__(self,
                 magnetic_field_amplitude: np.ndarray,
                 magnetic_field_angular_frequency: np.ndarray,
                 magnetic_field_phase_difference: np.ndarray,
                 ramp_interval: float,
                 start_time: float,
                 end_time: float):
        self.magnetic_field_amplitude = np.asarray(magnetic_field_amplitude,
                                                   dtype=float).reshape(3,)
        self.magnetic_field_angular_frequency = np.asarray(
            magnetic_field_angular_frequency, dtype=float
        ).reshape(3,)
        self.magnetic_field_phase_difference = np.asarray(
            magnetic_field_phase_difference, dtype=float
        ).reshape(3,)
        self.ramp_interval = float(ramp_interval)
        self.start_time = float(start_time)
        self.end_time = float(end_time)

    def value(self, time: np.float64 = 0.0) -> np.ndarray:
        factor = compute_ramp_factor(time,
                                     self.ramp_interval,
                                     self.start_time,
                                     self.end_time)
        phases = (self.magnetic_field_angular_frequency * float(time)
                  + self.magnetic_field_phase_difference)
        return factor * self.magnetic_field_amplitude * np.sin(phases)


# ========= 磁力矩 =========
class MagneticForces(NoForces):
    """
    均匀外磁场下的磁力矩：
      τ = m × B_local
    其中：
      - external_magnetic_field: BaseMagneticField 子类实例，value(time)->(3,)
      - magnetization_density: 标量或 (n_elems,) 数组
      - magnetization_direction: 世界坐标系下的磁化方向，shape 可以是：
            (3,), (3, n_elems) 或 (n_elems, 3)
      - rod_volume: (n_elems,) 每个单元体积
      - rod_director_collection: (3, 3, n_elems) Elastica 的 director
    """

    def __init__(self,
                 external_magnetic_field: BaseMagneticField,
                 magnetization_density: Union[float, int, np.ndarray],
                 magnetization_direction: np.ndarray,
                 rod_volume: np.ndarray,
                 rod_director_collection: np.ndarray):
        super(MagneticForces, self).__init__()

        # ==== 外磁场对象 ====
        self.external_magnetic_field = external_magnetic_field

        # ==== 体积 / 单元数 ====
        rod_volume = np.asarray(rod_volume, dtype=float).reshape(-1)
        n_elem = rod_volume.shape[0]

        # ------------ 方向处理 (世界坐标) ------------
        mag_dir = np.asarray(magnetization_direction, dtype=float)

        if mag_dir.shape == (3,):
            mag_dir = np.repeat(mag_dir[:, None], n_elem, axis=1)  # (3, n_elem)
        elif mag_dir.shape == (n_elem, 3):
            mag_dir = mag_dir.T  # (3, n_elem)
        elif mag_dir.shape == (3, n_elem):
            pass
        else:
            raise ValueError(
                "Invalid magnetization_direction shape {} "
                "(expected (3,), (3, n_elems) or (n_elems, 3))".format(
                    mag_dir.shape
                )
            )

        # 归一化
        norm = _batch_norm(mag_dir)   # (n_elem,)
        if np.any(norm == 0):
            raise ValueError("magnetization_direction 包含零向量。")
        mag_dir /= norm

        # 世界坐标 -> 材料坐标
        mag_dir_material = _batch_matvec(rod_director_collection, mag_dir)  # (3, n_elem)

        # ------------ 磁化强度 ------------
        if np.isscalar(magnetization_density):
            mag_den = float(magnetization_density)
            scale = mag_den * rod_volume
        else:
            mag_den = np.asarray(magnetization_density, dtype=float).reshape(-1)
            if mag_den.shape[0] != n_elem:
                raise ValueError(
                    "Invalid magnetization_density shape {}, expected scalar or (n_elems,)".format(
                        mag_den.shape
                    )
                )
            scale = mag_den * rod_volume  # (n_elem,)

        # 每个单元磁矩 m_i = (mag_den_i * volume_i) * dir_i
        self.magnetization_collection = mag_dir_material * scale  # (3, n_elem)

    # 注意：这里的签名必须接收 system=..., time=...
    def apply_torques(self,
                      system: CosseratRod = None,
                      time: np.float64 = 0.0,
                      *args,
                      **kwargs):
        """
        PyElastica 在同步 external_forces 时会调用：
            func(system=rod, time=time)
        所以这里要显式接收 system 关键字。
        """
        rod = system
        if rod is None:
            raise ValueError("MagneticForces.apply_torques 需要 system 参数。")

        # 外磁场（世界坐标） -> (3, n_elems)
        B_world = self.external_magnetic_field.value(time=time).reshape(3, 1) \
                  * np.ones((rod.n_elems,))

        # 变到材料坐标系
        B_material = _batch_matvec(rod.director_collection, B_world)  # (3, n_elems)

        # 磁力矩：m × B_local
        rod.external_torques += _batch_cross(self.magnetization_collection,
                                             B_material)

    def apply_forces(self,
                     system: CosseratRod = None,
                     time: np.float64 = 0.0,
                     *args,
                     **kwargs):
        """
        均匀磁场下不产生合力，这里留空，但要接收 system 关键字以免报错。
        """
        return
