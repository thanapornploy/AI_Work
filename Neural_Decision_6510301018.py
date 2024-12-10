import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense

# Generate dataset with make_blobs
n_samples = 100  # กำหนดจำนวน Data sample ในแต่ละคลาส (class)
std_dev = 0.75   # กำหนดส่วนเบี่ยงเบนมาตรฐานของทั้ง 2 Data set ของข้อมูลในแต่ละคลาส

# Class A (center at [2.0, 2.0])
x_a, y_a = make_blobs(n_samples=n_samples, centers=[[2.0, 2.0]], cluster_std=std_dev, random_state=42)
# ใช้ make_blobs สร้างข้อมูลชุดแรก โดยข้อมูลมีค่าเฉลี่ย (center) ที่ [2.0, 2.0]

# Class B (center at [3.0, 3.0])
x_b, y_b = make_blobs(n_samples=n_samples, centers=[[3.0, 3.0]], cluster_std=std_dev, random_state=42)
# ใช้ make_blobs สร้างข้อมูลชุดที่สอง โดยข้อมูลมีค่าเฉลี่ย (center) ที่ [3.0, 3.0]

# รวมข้อมูลจากสองคลาสเข้าในตัวแปรเดียวกัน
X = np.vstack((x_a, x_b))
# กำหนด label สำหรับแต่ละคลาส: Class A = 0, Class B = 1
y = np.hstack((np.zeros(n_samples), np.ones(n_samples))) 

# แบ่งข้อมูลออกเป็นชุด training และชุด testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# โดยใช้สัดส่วน 80% สำหรับฝึก และ 20% สำหรับทดสอบ
# กำหนดให้การสุ่มแบ่งข้อมูลเกิดขึ้นในรูปแบบที่สามารถทำซ้ำได้ หากใช้ค่าเดียวกันในการรันใหม่

# Perceptron Model
class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        # ฟังก์ชันเริ่มต้น (constructor) สำหรับกำหนดขนาด Input, learning rate, และ epochs
        self.weights = np.zeros(input_size)  # กำหนดค่าเริ่มต้นของ weights เป็น 0
        self.bias = 0  # กำหนดค่าเริ่มต้นของ bias เป็น 0
        self.learning_rate = learning_rate  # กำหนด learning rate
        self.epochs = epochs  # กำหนดจำนวน epochs

    def predict(self, X):
        # ฟังก์ชันสำหรับทำนายผลลัพธ์
        return np.where(self._activation_function(X) >= 0, 1, 0)
        # ใช้ฟังก์ชัน activation หากค่า >= 0 ให้เป็น 1, ถ้าไม่อย่างนั้นก็เป็น 0

    def _activation_function(self, X):
        # ฟังก์ชันสำหรับคำนวณผลรวมเชิงเส้น (linear)
        return np.dot(X, self.weights) + self.bias

    def train(self, X, y):
        # ฟังก์ชันสำหรับฝึกโมเดล
        for _ in range(self.epochs): # วนลูปตามจำนวนรอบการฝึก
            for i in range(len(X)):  # วนลูปในแต่ละตัวอย่าง
                # คำนวณความคลาดเคลื่อน (error) ระหว่างค่าจริงและค่าที่ทำนาย
                error = y[i] - self.predict(X[i])
                # ปรับปรุง weights ตาม error และ learning rate
                self.weights += self.learning_rate * error * X[i]
                # ปรับปรุง bias ตาม error และ learning rate
                self.bias += self.learning_rate * error
                
# Train Perceptron
perceptron = Perceptron(input_size=2)  # สร้าง Perceptron โดยมี input 2 มิติ
perceptron.train(X, y)  # ฝึกโมเดลด้วยข้อมูล X และ y

# ทำนายผลลัพธ์ด้วยโมเดล Perceptron
y_pred_perceptron = perceptron.predict(X)  

# Neural Network Model
model = Sequential() # กำหนดโมเดล Neural Network แบบ Sequential
model.add(Dense(16, input_dim=2, activation='relu')) # Layer 1 มี 16 นิวรอนและใช้ ReLU activation
model.add(Dense(1, activation='sigmoid')) # Layer สุดท้ายมี 1 นิวรอนและใช้ Sigmoid activation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile โมเดลด้วย loss function เป็น binary crossentropy และ optimizer เป็น Adam

# Train Neural Network
# ฝึก Neural Network โดยใช้ชุด training data และมีการกำหนด Parameter เพื่อควบคุมกระบวนการฝึกโมเดล
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

# Predict with Neural Network
# ทำนายผลลัพธ์ด้วย Neural Network และแปลงผลลัพธ์ให้อยู่ในรูปแบบตัวเลข (0 หรือ 1)
y_pred_nn = np.round(model.predict(X)).astype(int).ravel()

# Plot Neural Network Decision Boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # ตั้งขอบเขตของแกน x
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # ตั้งขอบเขตของแกน y
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01)) # สร้างเส้น grid ของค่าที่จะใช้วาด decision boundary

# Decision boundary for Neural Network
sum = model.predict(np.c_[xx.ravel(), yy.ravel()])  # คำนวณผลลัพธ์จาก Neural Network สำหรับทุกจุดใน grid
sum = np.round(sum).reshape(xx.shape)  # ปรับผลลัพธ์ให้อยู่ในรูปแบบ grid

# Plot Neural Network Decision Boundary
plt.figure(figsize=(7, 5))  # กำหนดขนาดของกราฟ
plt.contourf(xx, yy, sum, alpha=0.4, levels=[-0.5, 0.5, 1.5], colors=['red', 'blue']) # วาด decision boundary โดยใช้สีแดงและน้ำเงินแสดงพื้นที่ของแต่ละคลาส
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')  # วาดจุดข้อมูล
plt.title('Neural Network Decision Boundary')  # ตั้งชื่อกราฟ
plt.xlabel('Feature x1')  # ตั้งชื่อแกน x
plt.ylabel('Feature x2')  # ตั้งชื่อแกน y
plt.show()  # แสดงกราฟ
