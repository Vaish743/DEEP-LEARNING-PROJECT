import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_and_preprocess_data

def visualize_predictions(model_path='mnist_model.h5', num_samples=10):
    """
    Load model, predict on test samples, and visualize.
    Saves: plot as 'sample_predictions.png'
    """
    # Load data (only test set needed)
    _, (X_test, y_test) = load_and_preprocess_data()
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Predict
    predictions = model.predict(X_test[:num_samples])
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test[:num_samples], axis=1)
    
    # Visualize
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(X_test[i].squeeze(), cmap='gray')
        plt.title(f'True: {true_classes[i]}\nPred: {predicted_classes[i]}')
        plt.axis('off')
        
        # Highlight errors
        if predicted_classes[i] != true_classes[i]:
            plt.subplot(2, num_samples, i + 1 + num_samples)
            plt.imshow(X_test[i].squeeze(), cmap='gray')
            plt.title(f'ERROR: True {true_classes[i]}, Pred {predicted_classes[i]}')
            plt.axis('off')
        else:
            plt.subplot(2, num_samples, i + 1 + num_samples)
            plt.text(0.5, 0.5, 'Correct!', ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()
    print("Sample predictions plot saved as 'sample_predictions.png'")
    
    # Print accuracy on these samples
    sample_accuracy = np.mean(predicted_classes == true_classes)
    print(f"Accuracy on {num_samples} samples: {sample_accuracy:.2f}")

if __name__ == "__main__":
    visualize_predictions()