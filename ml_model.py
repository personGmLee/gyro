import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. 데이터 불러오기
# -----------------------------
X_train = pd.read_csv(r"D:/gyro/UCI HAR Dataset/train/X_train.txt", sep='\s+', header=None)
y_train = pd.read_csv(r"D:/gyro/UCI HAR Dataset/train/y_train.txt", header=None)
X_test = pd.read_csv(r"D:/gyro/UCI HAR Dataset/test/X_test.txt", sep='\s+', header=None)
y_test = pd.read_csv(r"D:/gyro/UCI HAR Dataset/test/y_test.txt", header=None)

# 활동 라벨 불러오기
activity_labels = pd.read_csv(r"D:/gyro/UCI HAR Dataset/activity_labels.txt", sep='\s+', header=None, index_col=0)
activity_labels_dict = activity_labels[1].to_dict()

# -----------------------------
# 2. 모델 학습
# -----------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train.values.ravel())

# -----------------------------
# 3. 예측
# -----------------------------
y_pred = clf.predict(X_test)

# -----------------------------
# 4. 평가
# -----------------------------
print("테스트 정확도:", accuracy_score(y_test, y_pred))
print("\n분류 리포트:\n", classification_report(y_test, y_pred, target_names=activity_labels_dict.values()))

# -----------------------------
# 5. 혼동 행렬 시각화
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=activity_labels_dict.values(),
            yticklabels=activity_labels_dict.values())
plt.xlabel("예측 라벨")
plt.ylabel("실제 라벨")
plt.title("혼동 행렬 (Confusion Matrix)")
plt.show()
