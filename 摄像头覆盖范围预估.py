import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.font_manager import FontProperties

# 设置中文字体，确保路径正确
font_path = 'NotoSansSC-Black.ttf'
font = FontProperties(fname=font_path)

# 固定输入值
a = float(input("请输入 a 的值（摄像头位置 z 坐标）："))
b = float(input("请输入 b 的值（最远观测点 y 坐标）："))
m = float(input("请输入 m 的值（摄像头垂直视场角）："))
n = float(input("请输入 n 的值（摄像头水平视场角）："))

# 角度转换为弧度
m = math.radians(m)
n = math.radians(n)

# 计算角 OAB 和角 OAD
theta_OAB = math.atan(b / a)
theta_OAD = theta_OAB - m / 2
theta_OAE = theta_OAB - m

# 计算点 D 和 E 的位置
y_D = math.tan(theta_OAD) * a
y_E = math.tan(theta_OAE) * a

# 计算 AD, DO, EO 的长度
AD = math.sqrt(a ** 2 + y_D ** 2)
DO = y_D
EO = y_E

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 绘制 yOz 平面图
O = np.array([0, 0])
A = np.array([0, a])
B = np.array([b, 0])
C = np.array([0, -a])
D = np.array([y_D, 0])
E = np.array([y_E, 0])

# 绘制点
ax1.scatter([O[0], A[0], B[0], C[0], D[0], E[0]], [O[1], A[1], B[1], C[1], D[1], E[1]], color='red')
ax1.text(*O, 'O', color='black', fontproperties=font)
ax1.text(*A, 'A', color='black', fontproperties=font)
ax1.text(*B, 'B', color='black', fontproperties=font)
ax1.text(*C, 'C', color='black', fontproperties=font)
ax1.text(*D, 'D', color='black', fontproperties=font)
ax1.text(*E, 'E', color='black', fontproperties=font)

# 绘制连线
ax1.plot([O[0], A[0]], [O[1], A[1]], color='blue')
ax1.plot([O[0], B[0]], [O[1], B[1]], color='blue')
ax1.plot([O[0], C[0]], [O[1], C[1]], color='blue')
ax1.plot([O[0], D[0]], [O[1], D[1]], color='blue')
ax1.plot([A[0], B[0]], [A[1], B[1]], color='green')
ax1.plot([A[0], D[0]], [A[1], D[1]], color='green')
ax1.plot([A[0], E[0]], [A[1], E[1]], color='green')

# 设置 yOz 平面图的标题和标签
ax1.set_title('yOz 平面图', fontproperties=font)
ax1.set_xlabel('y', fontproperties=font)
ax1.set_ylabel('z', fontproperties=font)
ax1.grid(True)
ax1.set_aspect('equal', 'box')

# 绘制 xOy 平面图
angles = np.linspace(-n / 2, n / 2, 100)
x_EO = EO * np.sin(angles)
y_EO = EO * np.cos(angles)
ax2.plot(x_EO, y_EO, label='EO 扇形', color='blue')

# 以 O 为原点，OB 为半径，顶角为 n 的扇形
OB = b
x_OB = OB * np.sin(angles)
y_OB = OB * np.cos(angles)
ax2.plot(x_OB, y_OB, label='OB 扇形', color='red')

# 计算并标记 E1, E2, D1, D2 的坐标
E1 = np.array([EO * np.sin(-n/2), EO * np.cos(-n/2)])
E2 = np.array([EO * np.sin(n/2), EO * np.cos(n/2)])
D1 = np.array([OB * np.sin(-n/2), OB * np.cos(-n/2)])
D2 = np.array([OB * np.sin(n/2), OB * np.cos(n/2)])
ax2.scatter([E1[0], E2[0], D1[0], D2[0]], [E1[1], E2[1], D1[1], D2[1]], color='red')
ax2.text(*E1, 'E1', color='black', fontproperties=font)
ax2.text(*E2, 'E2', color='black', fontproperties=font)
ax2.text(*D1, 'D1', color='black', fontproperties=font)
ax2.text(*D2, 'D2', color='black', fontproperties=font)

# 绘制并填充曲边四边形区域
def fill_curved_quad(ax, EO, OB, num_points=100):
    # 曲线弧 E1E2
    angles_inner = np.linspace(-n / 2, n / 2, num_points)
    x_inner = EO * np.sin(angles_inner)
    y_inner = EO * np.cos(angles_inner)

    # 曲线弧 D1D2
    x_outer = OB * np.sin(angles_inner)
    y_outer = OB * np.cos(angles_inner)

    # 填充弧线之间的区域
    ax.fill(np.concatenate([x_outer, x_inner[::-1]]),
            np.concatenate([y_outer, y_inner[::-1]]), color='gray', alpha=0.5)

# 填充曲面四边形区域
fill_curved_quad(ax2, EO, OB)

# 设置 xOy 平面图的标题和标签
ax2.set_title('xOy 平面图', fontproperties=font)
ax2.set_xlabel('x', fontproperties=font)
ax2.set_ylabel('y', fontproperties=font)
ax2.grid(True)

# 设置相同的横纵比例
ax2.set_aspect('equal', 'box')

# 手动设置 x 和 y 轴的范围，以确保比例一致
max_range = max(OB, EO)
ax2.set_xlim(-max_range, max_range)
ax2.set_ylim(0, max_range)

# 设置图例
ax2.legend(prop=font)

# 添加计算结果到图像上
result_text = (
    f"AD 的长度: {AD:.2f}\n"
    f"DO 的长度: {DO:.2f}\n"
    f"AD 与 DO 的比值: {DO/AD:.2f}\n"
    f"锥面展开的圆周角 q: {DO/AD*360:.2f}°\n"
    f"需要的摄像头个数为: {math.ceil((DO/AD*math.pi*2)/n)}"
)
ax2.text(-max_range, -max_range / 3, result_text, fontsize=10, va='top', ha='left', fontproperties=font)

# 显示图形
plt.tight_layout()
plt.show()
