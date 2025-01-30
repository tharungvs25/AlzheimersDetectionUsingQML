import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pennylane as qml

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

# Set the new image size to fit the number of qubits (4x4 pixels = 16 features)
new_image_size = (4, 4)

# Load the Alzheimer and non-Alzheimer images with the new size
alz_images, alz_labels = load_images(alzheimers_dir, 1, image_size=new_image_size)
non_alz_images, non_alz_labels = load_images(non_alzheimers_dir, 0, image_size=new_image_size)

# Combine the images and labels
X = np.array(alz_images + non_alz_images)
y = np.array(alz_labels + non_alz_labels)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Set the number of qubits to 16
number_of_qubits = 16
dev = qml.device('default.qubit', wires=number_of_qubits)

# Define a quantum circuit as a feature map
@qml.qnode(dev)
def quantum_feature_map(x):
    for i in range(len(x)):
        qml.Hadamard(wires=i)  # Apply Hadamard gates
        qml.RX(x[i], wires=i)  # Apply rotations based on input data
    # Apply entanglement using CNOT gates between qubits
    for i in range(number_of_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(number_of_qubits)]  # Measure Pauli-Z expectations

# Apply the quantum feature map to the training and testing data
def apply_quantum_feature_map(X_data):
    quantum_features = np.array([quantum_feature_map(x) for x in X_data])
    return quantum_features

X_train_quantum = apply_quantum_feature_map(X_train)
X_test_quantum = apply_quantum_feature_map(X_test)

# Train a classical SVM on the quantum features
clf = SVC(kernel='linear')
clf.fit(X_train_quantum, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_quantum)

# Calculate accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", confusion_mat)
print("Classification Report:\n", class_report)