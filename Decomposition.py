import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 定义向量
yw = np.array([3, 0, 0])
yls = np.array([0, 4, 0])
y = np.array([0, 4, 4])

# 计算误差向量
y_yls_error = y - yls
yls_yw_error = yls - yw
y_yw_error = y - yw

# 画向量
ax.quiver(0, 0, 0, yw[0], yw[1], yw[2], color='red', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, yls[0], yls[1], yls[2], color='green', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, y[0], y[1], y[2], color='blue', arrow_length_ratio=0.1)

# 画误差向量
ax.quiver(yls[0], yls[1], yls[2], y_yls_error[0], y_yls_error[1], y_yls_error[2], color='purple', linestyle='dashed', arrow_length_ratio=0.1)
ax.quiver(yw[0], yw[1], yw[2], yls_yw_error[0], yls_yw_error[1], yls_yw_error[2], color='orange', linestyle='dashed', arrow_length_ratio=0.1)
ax.quiver(yw[0], yw[1], yw[2], y_yw_error[0], y_yw_error[1], y_yw_error[2], color='cyan', linestyle='dashed', arrow_length_ratio=0.1)

# 向量标签
ax.text(yw[0]/2, yw[1]/2, yw[2]/2, 'yw', color='red')
ax.text(yls[0]/2, yls[1]/2, yls[2]/2, 'yls', color='green')
ax.text(y[0]/2, y[1]/2, y[2]/2, 'y', color='blue')

# 误差向量标签
ax.text(yls[0] + y_yls_error[0]/4, yls[1] + y_yls_error[1]/4, yls[2] + y_yls_error[2]/2, 'y-yls', color='purple')
ax.text(yw[0] + yls_yw_error[0]/2, yw[1] + yls_yw_error[1]/2, yw[2] + yls_yw_error[2]/2, 'yls-yw', color='orange')
ax.text(yw[0] + y_yw_error[0]/4, yw[1] + y_yw_error[1]/4, yw[2] + y_yw_error[2]/2, 'y-yw', color='cyan')

ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_zlim([0, 5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=20, azim=30)  # 调整视角

plt.show()
