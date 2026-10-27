"""Primuus 障碍地形任务；平地任务的运动、动力学和奖励配置继续复用。"""

from .primuus_config import PrimuusCfg, PrimuusCfgPPO


class PrimuusObstaclesCfg(PrimuusCfg):
    class terrain(PrimuusCfg.terrain):
        mesh_type = 'trimesh'
        horizontal_scale = 0.05  # 高度网格水平间距 [m]
        vertical_scale = 0.0025  # 高度网格量化间距 [m]
        border_size = 5.0  # 整张地形外侧的平地缓冲 [m]
        terrain_length = 8.0  # 单块地形的 X 长度 [m]
        terrain_width = 8.0  # 单块地形的 Y 宽度 [m]
        num_rows = 4  # 行间线性插值障碍幅度，复位时随机选行
        terrain_types = ('blocks', 'stairs_up', 'stairs_down', 'slope_up', 'slope_down')
        num_cols = len(terrain_types)  # 每列一种地形，环境按列均衡分配
        curriculum = False  # 当前按幅度随机采样，不启用地形晋级
        measure_heights = True  # critic 的187维高度输入使用真实地形采样
        platform_size = 4.0  # 中央安全复位平台边长 [m]；覆盖±1m扰动及台阶网格取整
        block_height_range = [0.02, 0.08]  # 随机障碍/凹陷的最大绝对高度 [m]
        block_size_range = [0.4, 1.2]  # 随机矩形的长宽范围 [m]
        num_blocks = 24  # 每块 blocks 地形的随机矩形数量
        step_height_range = [0.02, 0.08]  # 单级台阶高度 [m]，不是整片台阶总高
        step_width = 0.30  # 单级踏面深度 [m]
        slope_range = [0.05, 0.25]  # 坡度 rise/run，约2.9°～14.0°
        slope_treshold = 0.35  # 网格转三角面时修正陡边，保留低台阶立面
        edge_margin = 0.5  # 距本块边界不足此距离时按超时截断并复位 [m]


class PrimuusObstaclesCfgPPO(PrimuusCfgPPO):
    class runner(PrimuusCfgPPO.runner):
        experiment_name = 'primuus_obstacles'  # 障碍任务检查点与平地任务分开
        run_name = 'obstacles'
