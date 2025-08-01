# 1. 載入必要的函式庫
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 解決 matplotlib 圖表中文顯示問題
# 依序尋找系統中可用的字體，優先使用 'Microsoft JhengHei' (微軟正黑體)
plt.rcParams['font.sans-serif'] = [
    'Microsoft JhengHei', 'PingFang TC', 'Noto Sans CJK TC', 
    'WenQuanYi Zen Hei', 'sans-serif'
]
# 解決負號 '-' 顯示為方塊的問題
plt.rcParams['axes.unicode_minus'] = False

# 2. 載入並探索資料
try:
    # 假設 'Mall_Customers.csv' 在同一個資料夾
    df = pd.read_csv('Mall_Customers.csv')
except FileNotFoundError:
    print("錯誤: 'Mall_Customers.csv' 檔案不存在。請確認檔案路徑是否正確。")
    exit()

print("資料集資訊:")
df.info()
print("\n前5筆資料:")
print(df.head())

# 3. 選擇特徵進行分群
# 我們選擇 'Annual Income (k$)' 和 'Spending Score (1-100)'
# .values 會將 pandas DataFrame 轉換為 NumPy 陣列
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# 4. 資料標準化 (可選，但建議)
# 由於 K-Means 是基於距離的演算法，特徵縮放可以避免某些特徵因尺度較大而主導結果
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. 使用手肘法 (Elbow Method) 找到最佳的 k 值
wcss = [] # Within-Cluster Sum of Squares (組內平方和)
for i in range(1, 11):
    # n_init=10 避免陷入局部最佳解
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# 繪製手肘法圖
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('手肘法 (Elbow Method) 尋找最佳 K 值')
plt.xlabel('分群數量 (k)')
plt.ylabel('WCSS (組內平方和)')
plt.grid(True)
plt.show()

# 從圖中我們可以看到 k=5 是一個不錯的 "手肘" 點

# 6. 使用最佳 k 值進行 K-Means 分群
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

# 將分群結果加回原始 DataFrame
df['Cluster'] = y_kmeans

# 7. 視覺化分群結果
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', 
                data=df, palette=sns.color_palette("hsv", n_colors=optimal_k), s=100, alpha=0.7)

# 繪製群心 (Centroids)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.9, marker='X', label='Centroids')

plt.title('客戶分群 (K-Means Clustering)')
plt.xlabel('年收入 (k$)')
plt.ylabel('消費分數 (1-100)')
plt.legend()
plt.grid(True)
plt.show()

# 8. 顯示各群的統計資訊
print(f"\n已將客戶分為 {optimal_k} 群")
print("\n各群的特徵平均值:")
print(df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())