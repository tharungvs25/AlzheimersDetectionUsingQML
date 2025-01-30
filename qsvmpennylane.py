from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from PIL import Image

# Image dimensions (example 64x64 pixels)
img_width, img_height = 64, 64

# Function to load images and labels
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize((img_width, img_height))
        img_array = np.asarray(img).flatten()  # Flatten image to 1D
        images.append(img_array)
        labels.append(label)
    return images, labels

# Paths to the Alzheimer's and non-Alzheimer's image folders
alzheimers_path = 'AD'
non_alzheimers_path = 'NAD'

# Load Alzheimer's and non-Alzheimer's images
alzheimers_images, alzheimers_labels = load_images_from_folder(alzheimers_path, 1)
non_alzheimers_images, non_alzheimers_labels = load_images_from_folder(non_alzheimers_path, 0)

# Combine and split the dataset
X = np.array(alzheimers_images + non_alzheimers_images)
y = np.array(alzheimers_labels + non_alzheimers_labels)

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define Quantum Feature Map
feature_dimension = X_scaled.shape[1]  # Set feature dimension based on image data
feature_map = ZZFeatureMap(feature_dimension=feature_dimension, reps=2, entanglement="linear")

# Initialize the StatevectorSampler
sampler = StatevectorSampler()

# Manually calculate the quantum kernel (alternative to ComputeUncompute)
def calculate_kernel_matrix(sampler, feature_map, x_data):
    # Prepare feature map circuit for each sample and calculate the statevectors
    feature_map_circuits = [feature_map.assign_parameters(sample) for sample in x_data]
    results = sampler.run(feature_map_circuits).result()
    
    # Extract statevectors from results
    statevectors = results.quasi_dists
    
    # Calculate fidelity between statevectors to form the kernel matrix
    kernel_matrix = np.zeros((len(x_data), len(x_data)))
    for i in range(len(x_data)):
        for j in range(i, len(x_data)):
            fidelity = np.abs(np.dot(np.conjugate(statevectors[i]), statevectors[j])) ** 2
            kernel_matrix[i, j] = fidelity
            kernel_matrix[j, i] = fidelity
    return kernel_matrix

# Compute kernel matrix for training data
kernel_matrix_train = calculate_kernel_matrix(sampler, feature_map, x_train)

# Create and fit QSVC with manually computed kernel matrix
qsvm_model = QSVC(kernel='precomputed')
qsvm_model.fit(kernel_matrix_train, y_train)

# Compute kernel matrix for test data
kernel_matrix_test = calculate_kernel_matrix(sampler, feature_map, x_test)

# Predict on test data
y_pred = qsvm_model.predict(kernel_matrix_test)

# Evaluate performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Example prediction for a new image (flattened and preprocessed)
# Here, I'm using random values as a placeholder, replace with actual image
sample_image = np.random.rand(1, feature_dimension)
sample_image_scaled = scaler.transform(sample_image)
sample_kernel_matrix = calculate_kernel_matrix(sampler, feature_map, sample_image_scaled)
print("Predicted class for new sample image:", qsvm_model.predict(sample_kernel_matrix))