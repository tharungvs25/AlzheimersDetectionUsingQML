import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Directories for Alzheimer's and non-Alzheimer's images
alzheimers_dir = "AD"
non_alzheimers_dir = "NAD"

# Function to load and preprocess images
def load_images(image_dir, label, image_size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img.flatten())  # Flatten the image into a 1D array
            labels.append(label)  # Label: 1 for Alzheimer's, 0 for non-Alzheimer's
    return images, labels

# Load the Alzheimer and non-Alzheimer images
alz_images, alz_labels = load_images(alzheimers_dir, 1)
non_alz_images, non_alz_labels = load_images(non_alzheimers_dir, 0)

# Combine the images and labels
X = np.array(alz_images + non_alz_images)
y = np.array(alz_labels + non_alz_labels)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import ZZFeatureMap

# Define a quantum feature map
feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2, entanglement='linear')

# Define a quantum kernel
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=QuantumInstance(Aer.get_backend('qasm_simulator')))

# Apply the quantum kernel to the classical SVM model
X_train_quantum = quantum_kernel.evaluate(X_train)  # Transform training data
X_test_quantum = quantum_kernel.evaluate(X_test)    # Transform test data
