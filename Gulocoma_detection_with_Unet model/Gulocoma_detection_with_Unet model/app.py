import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, jaccard_score

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['GRAPH_FOLDER'] = 'static/graphs/'

# Make sure the folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GRAPH_FOLDER'], exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model('optic_disc_cup_segmentation_model.h5')

def preprocess_image(image_path, target_shape=(512, 512)):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, target_shape)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Define evaluation functions
def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true.flatten(), y_pred.flatten())

def compute_precision(y_true, y_pred):
    return precision_score(y_true.flatten(), y_pred.flatten())

def compute_recall(y_true, y_pred):
    return recall_score(y_true.flatten(), y_pred.flatten())

def compute_iou(y_true, y_pred):
    return jaccard_score(y_true.flatten(), y_pred.flatten())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    image = preprocess_image(filepath)
    predicted_mask = model.predict(image)
    predicted_mask = (predicted_mask[0] > 0.5).astype(np.uint8)  # Binarizing the predicted mask

    # Save the predicted mask
    mask_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_mask.png')
    cv2.imwrite(mask_filepath, predicted_mask * 255)  # Save mask as binary image

    # Compute metrics (dummy ground truth for example; replace with your actual data)
    # Replace 'ground_truth_masks' with your actual loaded ground truth masks
    # For demonstration, assuming you have a placeholder ground truth for comparison
    ground_truth_masks = np.zeros(predicted_mask.shape, dtype=np.uint8)  # Placeholder
    # Load your actual ground truth masks here

    accuracy = compute_accuracy(ground_truth_masks, predicted_mask)
    precision = compute_precision(ground_truth_masks, predicted_mask)
    recall = compute_recall(ground_truth_masks, predicted_mask)
    iou = compute_iou(ground_truth_masks, predicted_mask)

    cdr = np.sum(predicted_mask) * 0.7  # Placeholder for OC area calculation
    # Avoid division by zero
    cdr = cdr / np.sum(predicted_mask) if np.sum(predicted_mask) > 0 else 0

    # Save graphs (assuming you have functions to generate these)
    plot_loss_graph()  # Implement this function based on your model training
    plot_dice_coefficient_graph()

    return render_template('index.html', filename=file.filename, cdr=cdr,
                           accuracy=accuracy, precision=precision, recall=recall, iou=iou)

def plot_loss_graph():
    # Your logic to generate loss graph
    plt.figure()
    plt.plot([0, 1])  # Replace with actual loss data
    plt.title('Loss Graph')
    plt.savefig(os.path.join(app.config['GRAPH_FOLDER'], 'loss_graph.png'))
    plt.close()

def plot_dice_coefficient_graph():
    # Your logic to generate dice coefficient graph
    plt.figure()
    plt.plot([0, 1])  # Replace with actual dice coefficient data
    plt.title('Dice Coefficient Graph')
    plt.savefig(os.path.join(app.config['GRAPH_FOLDER'], 'dice_coefficient_graph.png'))
    plt.close()

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/graphs/<filename>')
def uploaded_graph(filename):
    return send_from_directory(app.config['GRAPH_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
