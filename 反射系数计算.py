import numpy as np
import matplotlib.pyplot as plt
def compute_reflectivity(impedance):
    """
    计算二维波阻抗数组的反射系数数组。
    输入: impedance - 形状为 (nz, nx) 的二维numpy数组，nz 为深度/时间方向，nx 为横向方向。
    输出: reflectivity - 形状与输入相同的反射系数数组。
    
    反射系数计算公式: R[i] = (Z[i+1] - Z[i]) / (Z[i+1] + Z[i])
    为了保持相同形状，在第一个位置补0，后续位置计算界面反射系数，最后一个位置保持为0（或可调整）。
    """
    if not isinstance(impedance, np.ndarray) or impedance.ndim != 2:
        raise ValueError("输入必须是二维numpy数组")
    
    # 创建相同形状的反射系数数组，初始化为0
    reflectivity = np.zeros_like(impedance, dtype=float)
    
    # 沿着深度方向 (axis=0) 计算相邻层间的反射系数
    # 添加一个小值防止除零错误
    numerator = impedance[1:] - impedance[:-1]
    denominator = impedance[1:] + impedance[:-1] + 1e-10
    r_interface = numerator / denominator
    
    # 将计算结果放置在下层位置（从索引1开始）
    reflectivity[1:] = r_interface
    
    return reflectivity

# 示例使用
# 假设你的波阻抗数组如下（你可以替换为自己的数据）
Z = np.load(r"data\impe.npy")

R = compute_reflectivity(Z)

# np.save(r"data\reflectivity.npy", R)
print("波阻抗数组 Z:\n", Z)
print("反射系数数组 R:\n", R)

plt.figure()
plt.imshow(R, "seismic", aspect='auto')
plt.colorbar()
plt.show()