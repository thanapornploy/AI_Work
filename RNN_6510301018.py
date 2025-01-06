import keras.api.models as mod
import keras.api.layers as lay
import numpy as np
import matplotlib.pyplot as plt

# สร้างโมเดลแรกและบันทึกไฟล์
model = mod.Sequential()
model.add(lay.SimpleRNN(units=1, input_shape=(1, 1), activation="relu"))
model.summary()
model.save("RNN.h5")

# การสร้างข้อมูลตัวอย่าง
pitch = 20
step = 1
N = 1000
n_train = int(N * 0.7)  # 70% สำหรับ Training set

def gen_data(x):
    return (x % pitch) / pitch

t = np.arange(1, N + 1)
y = ([gen_data(i) for i in t])
y = np.sin(0.01*t*10) + 0.1 * np.random.rand(N) 
y = np.array(y)

# ฟังก์ชันสำหรับแปลงข้อมูลเป็น Matrix
def convertToMatrix(data, step=1):
    X, Y = [], []
    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d])
        Y.append(data[d])
    return np.array(X), np.array(Y)

# แบ่งข้อมูลเป็น Training และ Testing
train, test = y[0:n_train], y[n_train:N]
x_train, y_train = convertToMatrix(train, step)
x_test, y_test = convertToMatrix(test, step)

print("Dimension (Before): ", train.shape, test.shape)
print("Dimension (After): ", x_train.shape, x_test.shape)

# การสร้างและ Train Recurrent Neural Network (RNN)
model = mod.Sequential()
model.add(lay.SimpleRNN(units=32, input_shape=(step, 1), activation="relu"))
model.add(lay.Dense(units=1))

model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
hist = model.fit(x_train, y_train, epochs=30, batch_size=1, verbose=1)

# ทำนายผลลัพธ์และเปรียบเทียบกับข้อมูลจริง
y_pred = model.predict(x_test)

# แสดงกราฟเปรียบเทียบผลลัพธ์
plt.figure()
plt.plot(y_test, label="Original", color="blue")  # ข้อมูลจริง
plt.plot(y_pred, label="Predict", linestyle="--", color="red")  # ผลลัพธ์ที่พยากรณ์ได้
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
