import numpy as np
import extraction_function
import main_train
import joblib
import pandas as pd

def model():
    root_path = r"D:\JerryHuang\作业\大三下半学期\Iot Security\lab1_ voice identity\pythonProject2\dev-clean"
    feature_file = extraction_function.extraction_main(root_path)
    main_train.model_trian(feature_file)


def t_SNE(file_name):
    df = pd.read_csv(file_name)
    print(df.head())

    # Assuming the last column is the label
    features = df.iloc[:, :-1]  # All columns except the last one
    labels = df.iloc[:, -1]  # The last column

    print(features.head())
    print(labels.head())

    from sklearn.preprocessing import StandardScaler

    # Standardize the features
    scaler = StandardScaler()
    features_std = scaler.fit_transform(features)

    print(features_std[:5])

    from sklearn.manifold import TSNE

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features_std)

    print(features_tsne[:5])

    import matplotlib.pyplot as plt

    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

if __name__ == "__main__":
    test_file = "dev-clean/1272/128104/1272-128104-0000.flac"

    features = extraction_function.extractSingleFeature(test_file)

    features = np.array(features)  # 如果 features 还不是 numpy 数组，先转换
    features = features.reshape(1, -1)  # 将一维数组转换为二维数组

    # Load the model
    model_filename = 'random_forest_model.pkl'
    rf_classifier = joblib.load(model_filename)
    scaler_name = 'standard_scaler.pkl'
    scaler = joblib.load(scaler_name)

    # Standardize the features
    features_scaled = scaler.transform(features)

    # Predict the label
    label = rf_classifier.predict(features_scaled)
    print(f'Random forest model Predicted label: {label}')

    # Load the SVM model
    model_filename = 'svm_model.pkl'
    svm_classifier = joblib.load(model_filename)

    # Predict the label
    label = svm_classifier.predict(features_scaled)
    print(f'SVM model Predicted label: {label}')



