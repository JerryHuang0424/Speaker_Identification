import numpy as np

features = np.load('features.npy')
labels = np.load('labels.npy')

from sklearn.preprocessing import StandardScaler

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

from sklearn.model_selection import train_test_split

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

from sklearn.svm import SVC

# 创建SVM分类器
svm_classifier = SVC(kernel='linear')  # 你可以选择不同的核函数，如 'rbf', 'poly' 等

# 训练模型
svm_classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# 预测测试集
y_pred = svm_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# 打印分类报告
print(classification_report(y_test, y_pred))

# 实现在线训练，可以实时新添加数据
# 对输入进行输出，可以把语音对应的说话人输出出来

