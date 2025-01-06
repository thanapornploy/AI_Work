import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

# === 1. สร้างข้อมูลตัวอย่าง ===
# Class 1
x1, y1 = make_blobs(n_samples=100,
                    n_features=2,