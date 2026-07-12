import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
ZIP_FILE = "UCI_HAR_Dataset.zip"
EXTRACT_FOLDER = "UCI HAR Dataset"

def download_and_extract():
    if not os.path.exists(EXTRACT_FOLDER):
        if not os.path.exists(ZIP_FILE):
            print("Downloading the dataset...")
            urllib.request.urlretrieve(DATASET_URL, ZIP_FILE)
            print("Download complete.")
        
        print("Extracting the dataset...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Extraction complete.")
    else:
        print("Dataset already exists.")

def load_data():
    print("Loading data...")
    features_path = os.path.join(EXTRACT_FOLDER, "features.txt")
    features = pd.read_csv(features_path, sep=r'\s+', header=None, usecols=[1])[1].values
    
    # Load training data
    X_train = pd.read_csv(os.path.join(EXTRACT_FOLDER, "train", "X_train.txt"), sep=r'\s+', header=None)
    y_train = pd.read_csv(os.path.join(EXTRACT_FOLDER, "train", "y_train.txt"), sep=r'\s+', header=None).values.ravel()
    
    # Load testing data
    X_test = pd.read_csv(os.path.join(EXTRACT_FOLDER, "test", "X_test.txt"), sep=r'\s+', header=None)
    y_test = pd.read_csv(os.path.join(EXTRACT_FOLDER, "test", "y_test.txt"), sep=r'\s+', header=None).values.ravel()
    
    print("Data loaded successfully.")
    return X_train, y_train, X_test, y_test

def evaluate_model(y_test, y_pred, model_name):
    print(f"\n--- Evaluation for {model_name} ---")
    
    # Activity labels mapping (Dataset uses numbers 1 to 6)
    activity_labels = ["Walking", "Walking Upstairs", "Walking Downstairs", "Sitting", "Standing", "Laying"]
    
    # Print some real examples for the user to see the actual activity names
    print("\n[Real Examples of Predictions]")
    for i in range(5):
        actual = activity_labels[y_test[i] - 1]
        predicted = activity_labels[y_pred[i] - 1]
        print(f"Example {i+1} -> Actual: {actual:18} | Predicted: {predicted}")
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=activity_labels, yticklabels=activity_labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual Activity')
    plt.xlabel('Predicted Activity')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    print(f"Confusion matrix saved as {model_name}_confusion_matrix.png")

def main():
    download_and_extract()
    X_train, y_train, X_test, y_test = load_data()
    
    # 1. K-Nearest Neighbors (KNN)
    print("\nTraining K-Nearest Neighbors (KNN)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    evaluate_model(y_test, knn_preds, "KNN")
    
    # 2. Support Vector Machine (SVM)
    print("\nTraining Support Vector Machine (SVM)...")
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    evaluate_model(y_test, svm_preds, "SVM")

if __name__ == "__main__":
    main()
