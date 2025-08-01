# 載入資料集
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# 資料探索
print("資料形狀:", df.shape)
print("前5行資料:")
print(df.head())

# 視覺化
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='target', y=feature, data=df)
    plt.title(f'{feature} by Species')
plt.tight_layout()
plt.show()


# 建立預測模型
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 分割資料
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 訓練模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 評估模型
y_pred = rf.predict(X_test)
print("準確率:", rf.score(X_test, y_test))
print("\n詳細報告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 特徵重要性
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\n特徵重要性:")
print(feature_importance)