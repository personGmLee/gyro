import pandas as pd
import matplotlib
matplotlib.use("Tkagg")
import matplotlib.pyplot as plt
import seaborn as sns

X_train = pd.read_csv(r"D:/gyro/UCI HAR Dataset/train/X_train.txt", sep='\s+', header=None)
y_train = pd.read_csv(r"D:/gyro/UCI HAR Dataset/train/y_train.txt", header=None)
X_test = pd.read_csv(r"D:/gyro/UCI HAR Dataset/train/X_train.txt", sep='\s+', header=None)
y_test = pd.read_csv(r"D:/gyro/UCI HAR Dataset/train/y_train.txt", header=None)

activity_labels = pd.read_csv(r"D:/gyro/UCI HAR Dataset/activity_labels.txt", sep='\s+', header=None, index_col=0)
activity_labels_dict = activity_labels[1].to_dict()

print("훈련 데이터 크기:", X_train.shape)
print("테스트 데이터 크기:", X_test.shape)
print("레이블 분포 (훈련):")
print(y_train[0].value_counts())

y_train_named = y_train[0].map(activity_labels_dict)
y_test_named = y_test[0].map(activity_labels_dict)

plt.figure(figsize=(8,5))
sns.countplot(x=y_train_named, order=y_train_named.value_counts().index, color="skyblue")
plt.title("train data")
plt.xticks(rotation=30)
plt.show()


plt.figure(figsize=(10,6))
sns.boxplot(data=X_train.iloc[:, :10], orient="h", palette="coolwarm")
plt.title("Distribution of First 10 Features (Training Data)")
plt.show()

corr = X_train.iloc[:, :20].corr()  # 앞 20개 feature만
plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (First 20 Features)")
plt.show()
