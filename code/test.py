import numpy as np
import matplotlib.pyplot as plt

# Đây là một ví dụ về các điểm dữ liệu mô phỏng. Thay thế chúng bằng kết quả thực tế của bạn.
f1_scores = np.array([0.85, 0.92, 0.78, 0.89, 0.95, 0.88, 0.75, 0.91, 0.84, 0.93])

# Sắp xếp các điểm dữ liệu theo thứ tự tăng dần.
sorted_scores = np.sort(f1_scores)

# Tạo một mảng chứa các giá trị của CDF.
cdf_values = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

# Vẽ biểu đồ CDF.
plt.plot(sorted_scores, cdf_values, marker='o', linestyle='-', color='b')
plt.xlabel('F1-score')
plt.ylabel('CDF')
plt.title('Cumulative Distribution Function (CDF) of F1-scores')
plt.grid(True)

plt.show()