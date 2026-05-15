import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.ioff()

# ==================== F1: Sphere ====================
x1 = np.linspace(-40, 40, 100)
y1 = np.linspace(-40, 40, 100)
X1, Y1 = np.meshgrid(x1, y1)
Z1 = X1**2 + Y1**2

fig1 = plt.figure(figsize=(8, 7))
ax1 = fig1.add_subplot(111, projection='3d')
surf1 = ax1.plot_surface(X1, Y1, Z1, cmap='jet', edgecolor='none', antialiased=True)
ax1.set_title('Sphere Function', fontsize=13, pad=15)
ax1.set_xlabel('x₁', fontsize=11, labelpad=10)
ax1.set_ylabel('x₂', fontsize=11, labelpad=10)
ax1.set_zlabel('f(x₁,x₂)', fontsize=11, labelpad=10)
ax1.set_xlim(-40, 40)
ax1.set_ylim(-40, 40)
ax1.set_zlim(0, 4000)
ax1.view_init(elev=25, azim=-60)
fig1.colorbar(surf1, shrink=0.6, aspect=12, pad=0.08)
plt.tight_layout(pad=2.0)
plt.savefig('F1_Sphere.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.close()
print("[✓] F1_Sphere.png")

# ==================== F2: Schwefel 2.22 ====================
x2 = np.linspace(-10, 10, 100)
y2 = np.linspace(-10, 10, 100)
X2, Y2 = np.meshgrid(x2, y2)
Z2 = np.abs(X2) + np.abs(Y2) + np.abs(X2) * np.abs(Y2)

fig2 = plt.figure(figsize=(8, 7))
ax2 = fig2.add_subplot(111, projection='3d')
surf2 = ax2.plot_surface(X2, Y2, Z2, cmap='jet', edgecolor='none', antialiased=True)
ax2.set_title('Schwefel 2.22 Function', fontsize=13, pad=15)
ax2.set_xlabel('x₁', fontsize=11, labelpad=10)
ax2.set_ylabel('x₂', fontsize=11, labelpad=10)
ax2.set_zlabel('f(x)', fontsize=11, labelpad=10)
ax2.set_xlim(-10, 10)
ax2.set_ylim(-10, 10)
ax2.set_zlim(0, 130)
ax2.view_init(elev=25, azim=-60)
fig2.colorbar(surf2, shrink=0.6, aspect=12, pad=0.08)
plt.tight_layout(pad=2.0)
plt.savefig('F2_Schwefel222.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.close()
print("[✓] F2_Schwefel222.png")

# ==================== F3: Ackley ====================
x3 = np.linspace(-5, 5, 100)
y3 = np.linspace(-5, 5, 100)
X3, Y3 = np.meshgrid(x3, y3)
Z3 = -20 * np.exp(-0.2 * np.sqrt((X3**2 + Y3**2) / 2)) - \
     np.exp((np.cos(2 * np.pi * X3) + np.cos(2 * np.pi * Y3)) / 2) + 20 + np.e

fig3 = plt.figure(figsize=(8, 7))
ax3 = fig3.add_subplot(111, projection='3d')
surf3 = ax3.plot_surface(X3, Y3, Z3, cmap='jet', edgecolor='none', antialiased=True)
ax3.set_title('Ackley Function', fontsize=13, pad=15)
ax3.set_xlabel('x₁', fontsize=11, labelpad=10)
ax3.set_ylabel('x₂', fontsize=11, labelpad=10)
ax3.set_zlabel('f(x)', fontsize=11, labelpad=10)
ax3.set_xlim(-5, 5)
ax3.set_ylim(-5, 5)
ax3.set_zlim(0, 16)
ax3.view_init(elev=25, azim=-60)
fig3.colorbar(surf3, shrink=0.6, aspect=12, pad=0.08)
plt.tight_layout(pad=2.0)
plt.savefig('F3_Ackley.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.close()
print("[✓] F3_Ackley.png")

# ==================== F4: Rastrigin ====================
x4 = np.linspace(-5.12, 5.12, 100)
y4 = np.linspace(-5.12, 5.12, 100)
X4, Y4 = np.meshgrid(x4, y4)
Z4 = X4**2 + Y4**2 - 10 * np.cos(2 * np.pi * X4) - 10 * np.cos(2 * np.pi * Y4) + 20

fig4 = plt.figure(figsize=(8, 7))
ax4 = fig4.add_subplot(111, projection='3d')
surf4 = ax4.plot_surface(X4, Y4, Z4, cmap='jet', edgecolor='none', antialiased=True)
ax4.set_title('Rastrigin Function', fontsize=13, pad=15)
ax4.set_xlabel('x₁', fontsize=11, labelpad=10)
ax4.set_ylabel('x₂', fontsize=11, labelpad=10)
ax4.set_zlabel('f(x)', fontsize=11, labelpad=10)
ax4.set_xlim(-5.12, 5.12)
ax4.set_ylim(-5.12, 5.12)
ax4.view_init(elev=15, azim=-60)
fig4.colorbar(surf4, shrink=0.6, aspect=12, pad=0.08)
plt.tight_layout(pad=2.0)
plt.savefig('F4_Rastrigin.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.close()
print("[✓] F4_Rastrigin.png")

print("\n全部完成！4张独立基准函数图已生成。")