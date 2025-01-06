import json

# Data Linear Regression
data = {
    "x": [29, 28, 34, 31, 25],
    "y": [77, 62, 93, 84, 59]
}

# สร้างไฟล์ .json
file_name = "Regression_data_6510301018.json"
with open(file_name, "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f"File '{file_name}' is success !")
