import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

# ข้อมูลตัวอย่าง
# Class 1
x1, y1 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(-1.0, -1.0),
                    cluster_std=0.5, # ปรับการกระจายของจุดข้อมูล
                    random_state=69)

# Class 2
x2, y2 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(1.0, 1.0),
                    cluster_std=0.5,  # ปรับการกระจายของจุดข้อมูล
                    random_state=69)

# กำหนดฟังก์ชัน Decision Function 
def decision_function(x1, x2):
    return x1 + x2 - 0.01  # สามารถปรับค่าฟังก์ชันนี้ได้

# สร้าง Grid ของจุดข้อมูล
x1_range = np.linspace(-3, 3, 500)  # ขอบเขตแกน x1
x2_range = np.linspace(-3, 3, 500)  # ขอบเขตแกน x2
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range) # สร้างตาราง Grid

# คำนวณค่า decision function
g_values = decision_function(x1_grid, x2_grid)

# สร้างคลาส Perceptron 
class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        self.weights = np.zeros(input_size)  # เริ่มต้น weights ที่ศูนย์
        self.bias = 0  # เริ่มต้น bias ที่ 0
        self.learning_rate = learning_rate # กำหนดค่าพารามิเตอร์ learning rate (อัตราการเรียนรู้) 
        self.epochs = epochs # กำหนดจำนวนรอบการ training

    def predict(self, X):
        # ฟังก์ชันการตัดสินใจ ฟังก์ชันนี้ใช้การตัดสินใจตามเงื่อนไขของค่าที่ได้จากฟังก์ชันการกระตุ้น (_activation_function) 
        return np.where(self._activation_function(X) >= 0, 1, 0)

    def _activation_function(self, X):
        # ฟังก์ชันการกระตุ้น 
        return np.dot(X, self.weights) + self.bias # การคำนวณผลคูณจุด (Dot Product) ระหว่างอินพุต X และเวกเตอร์น้ำหนัก self.weights

    # ฟังก์ชันนี้ใช้สำหรับการฝึกอบรมโมเดล ซึ่งเป็นโมเดลการเรียนรู้แบบเชิงเส้น (Linear Classifier)
    def train(self, X, y):
        for _ in range(self.epochs): # วนลูปซ้ำตามจำนวน epochs
            for i in range(len(X)):  # วนลูปซ้ำทีละตัวอย่างข้อมูล
                if y[i] != self.predict(X[i]):  # ตรวจสอบว่าผลลัพธ์ผิดหรือไม่
                    error = y[i] - self.predict(X[i])  # คำนวณค่า error
                    self.weights += self.learning_rate * error * X[i] # ปรับ weights
                    self.bias += self.learning_rate * error  # ปรับ bias

# ฝึก Perceptron 
X = np.vstack((x1, x2))  # รวมข้อมูลจากทั้งสองคลาส
y = np.hstack((np.zeros(100), np.ones(100)))  # label: 0 สำหรับ Class 1, label: 1 สำหรับ Class 2

perceptron = Perceptron(input_size=2)  # สร้าง Perceptron โดยมี input 2 ค่า (x1, x2)
perceptron.train(X, y)  # ฝึก Perceptron

# predict ผลลัพธ์
y_pred = perceptron.predict(X)  # ทำนายผลลัพธ์จากข้อมูล X

# คำนวณ Accuracy
accuracy = np.mean(y_pred == y) * 100  # คำนวณ Accuracy ของ Perceptron
print(f"Weights: {perceptron.weights}")  # แสดงผลค่า weights
print(f"Accuracy: {accuracy:.2f}%")  # แสดงผลค่า Accuracy

# วาดกราฟผลลัพธ์
plt.figure(figsize=(6, 4))  # กำหนดขนาดกราฟ
plt.title('Perceptron Decision Plane', fontsize=16)  # กำหนดชื่อกราฟ

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
