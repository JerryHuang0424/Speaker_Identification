import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def accuracy(features_scaled, labels):
    # 将数据拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)

    # 训练随机森林模型
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # 进行预测
    y_pred = rf_classifier.predict(X_test)

    # 计算准确率
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc:.4f}')

def model_train(input_file):
    # 加载CSV文件到DataFrame
    df = pd.read_csv(input_file)

    # 显示DataFrame的前几行
    print("Input Data Preview:")
    print(df.head())

    # 分离特征和标签
    features = df.drop(columns=['label']).values
    labels = df['label'].values

    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 计算并打印准确率
    accuracy(features_scaled, labels)

    # 初始化随机森林分类器
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # 训练模型
    rf_classifier.fit(features_scaled, labels)

    # 保存模型
    model_filename = 'random_forest_model.pkl'
    joblib.dump(rf_classifier, model_filename)
    print(f'Model saved to {model_filename}')

    # （可选）保存标准化器以便后续使用
    scaler_filename = 'standard_scaler.pkl'
    joblib.dump(scaler, scaler_filename)
    print(f'Scaler saved to {scaler_filename}')

# 示例调用
if __name__ == "__main__":
    # 确保在调用时提供正确的输入文件路径
    model_train('features_labels.csv')