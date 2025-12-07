import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

X_train = pd.read_csv(r"D:/gyro/UCI HAR Dataset/train/X_train.txt", sep='\s+', header=None)
y_train = pd.read_csv(r"D:/gyro/UCI HAR Dataset/train/y_train.txt", header=None)
X_test = pd.read_csv(r"D:/gyro/UCI HAR Dataset/test/X_test.txt", sep='\s+', header=None)
y_test = pd.read_csv(r"D:/gyro/UCI HAR Dataset/test/y_test.txt", header=None)

activity_labels = pd.read_csv(r"D:/gyro/UCI HAR Dataset/activity_labels.txt", sep='\s+', header=None, index_col=0)
activity_labels_dict = activity_labels[1].to_dict()

encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train.values.ravel())
y_test_enc = encoder.transform(y_test.values.ravel())


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(activity_labels_dict), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train_enc, epochs=20, batch_size=64, validation_split=0.1)

y_pred_probs = model.predict(X_test)
y_pred = y_pred_probs.argmax(axis=1)

print("테스트 정확도:", accuracy_score(y_test_enc, y_pred))
print("\n분류 리포트:\n", classification_report(y_test_enc, y_pred, target_names=activity_labels_dict.values()))

cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=activity_labels_dict.values(),
            yticklabels=activity_labels_dict.values())
plt.xlabel("예측 라벨")
plt.ylabel("실제 라벨")
plt.title("혼동 행렬 (Confusion Matrix)")
plt.show()
