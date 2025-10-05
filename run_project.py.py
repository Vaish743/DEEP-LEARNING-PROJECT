from train import train_model
from predict import visualize_predictions

if __name__ == "__main__":
    print("Starting MNIST CNN Project...")
    print("Step 1: Training the model...")
    train_model()
    
    print("\nStep 2: Visualizing predictions...")
    visualize_predictions()
    
    print("\nProject complete! Check saved files: mnist_model.h5, training_history.png, sample_predictions.png")