import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

# === 1. สร้างข้อมูลตัวอย่าง ===
# Class 1
x1, y1 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(-1.0, -1.0),
                    cluster_std=0.5,
                    random_state=69)

# Class 2
x2, y2 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(1.0, 1.0),
                    cluster_std=0.5,
                    random_state=69)

# === 2. กำหนดฟังก์ชัน Decision Function ===
def decision_function(x1, x2):
    return x1 + x2 - 0.01  # สามารถปรับค่าฟังก์ชันนี้ได้

# === 3. สร้าง Decision Plane ===
# สร้าง Grid ของจุดข้อมูล
x1_range = np.linspace(-3, 3, 500)  # ขอบเขตแกน x1
x2_range = np.linspace(-3, 3, 500)  # ขอบเขตแกน x2
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# คำนวณค่าฟังก์ชันการตัดสินใจ
g_values = decision_function(x1_grid, x2_grid)

# === 4. วาดกราฟรวม ===
plt.figure(figsize=(6, 4))  # กำหนดขนาดกราฟ
plt.title('Perceptron Decision Plane', fontsize=16)  # ชื่อกราฟ

# วาด Decision Regions
plt.contourf(
    x1_grid, x2_grid, g_values,
    levels=[-np.inf, 0, np.inf],  # แบ่งพื้นที่ตามค่าฟังก์ชัน
    colors=['red', 'blue'],  # สีแต่ละคลาส
    alpha=0.5
)

# วาดเส้น Decision Boundary
plt.contour(
    x1_grid, x2_grid, g_values,
    levels=[0],  # เส้นแบ่งเขตตัดสินใจที่ค่า 0
    colors='black',
    linewidths=2
)

# วาดจุดข้อมูลของ Class 1 และ Class 2 ซ้อนบน Decision Boundary
plt.scatter(x1[:, 0], x1[:, 1], c='purple', edgecolor='k', alpha=0.6, label="Class 1")
plt.scatter(x2[:, 0], x2[:, 1], c='yellow', edgecolor='k', alpha=0.6, label="Class 2")

# ตั้งค่ากราฟ
plt.xlabel('Feature x1', fontsize=12)
plt.ylabel('Feature x2', fontsize=12)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True)
plt.xlim([-3, 3])  # กำหนดช่วงแกน x
plt.ylim([-3, 3])  # กำหนดช่วงแกน y
plt.show()
