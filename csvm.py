import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Directories for Alzheimer and non-Alzheimer images
alzheimers_dir = "AD"
non_alzheimers_dir = "NAD"

# Function to load and preprocess images
def load_images(image_dir, label, image_size=(128, 128)):
    images = []
    labels = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img.flatten())  # Flatten the image to a 1D array
            labels.append(label)  # Assign the appropriate label
    return images, labels

# Load Alzheimer and non-Alzheimer images
alz_images, alz_labels = load_images(alzheimers_dir, 1)  # Label: 1 for Alzheimer
non_alz_images, non_alz_labels = load_images(non_alzheimers_dir, 0)  # Label: 0 for non-Alzheimer

# Combine data
X = np.array(alz_images + non_alz_images)
y = np.array(alz_labels + non_alz_labels)

# Standardize the feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optionally, reduce dimensionality using PCA
pca = PCA(n_components=100)  # Reducing to 100 principal components
X_pca = pca.fit_transform(X_scaled)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

svm_model = SVC(kernel='linear')  # Or 'rbf'
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)

# Evaluate
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("Classification Report")
print(classification_report(y_test, y_pred))
