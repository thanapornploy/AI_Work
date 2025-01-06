import json
import matplotlib.pyplot as plt

# อ่านข้อมูลจากไฟล์ JSON
file_path = "Regression_data_6510301018.json"  
with open(file_path, "r") as file:
    data = json.load(file)

# แยกค่า x และ y
x = data["x"]
y = data["y"]

# คำนวณค่าเฉลี่ยของ x และ y
mean_x = sum(x) / len(x)
mean_y = sum(y) / len(y)

# คำนวณ Sum of Squares และ Sum of Products
Sxx = sum((xi - mean_x) ** 2 for xi in x)
Syy = sum((yi - mean_y) ** 2 for yi in y)
Sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

# คำนวณ Slope (m) และ Intercept (c)
m = Sxy / Sxx
c = mean_y - m * mean_x

# คำนวณค่า Predict จากสมการ Linear Regression
y_pred = [m * xi + c for xi in x]

# แสดงผลลัพธ์
print(f"ค่าเฉลี่ยของ x: {mean_x:.2f}")
print(f"ค่าเฉลี่ยของ y: {mean_y:.2f}")
print(f"Sum of Squares for x (Sxx): {Sxx:.2f}")
print(f"Sum of Squares for y (Syy): {Syy:.2f}")
print(f"Sum of Products (Sxy): {Sxy:.2f}")
print(f"Regression equation: y = {m:.2f}x {c:.2f}")

# สร้างกราฟ Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Actual Data')  # จุดของ Data
plt.plot(x, y_pred, color='red', label='Regression Line')  # เส้น Linear Regression
plt.xlabel('x (High Temp in °C)')
plt.ylabel('y (Iced Tea Orders)')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
