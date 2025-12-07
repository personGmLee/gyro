import pandas as pd

# 기존 방식 (경고 발생)
# X_train = pd.read_csv("UCI HAR Dataset/train/X_train.txt", delim_whitespace=True, header=None)

# 권장 방식 (경고 없음)
X_train = pd.read_csv(r"D:/gyro/UCI HAR Dataset/train/X_train.txt", sep='\s+', header=None)
y_train = pd.read_csv(r"D:/gyro/UCI HAR Dataset/train/y_train.txt", header=None)

print("훈련 데이터 크기:", X_train.shape)
print("레이블 크기:", y_train.shape)
