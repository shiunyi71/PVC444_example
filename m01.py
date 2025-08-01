# 1. 載入資料
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# 2. 分割資料
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 3. 選擇模型
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
# from sklearn.svm import SVC
# model = SVC(kernel='rbf')

# 4. 訓練模型
model.fit(X_train, y_train)

# 5. 預測與評估
predictions = model.predict(X_test)
from sklearn.metrics import accuracy_score
print(f"準確率: {accuracy_score(y_test, predictions):.2f}")

