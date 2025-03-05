from autoencoder_class import Autoencoder
import matplotlib.pyplot as plt

test_digit = 3
num_hidden_units = [5, 30, 60, 120, 200]

mse_results = {}
val_loss_results = {}
for num_units in num_hidden_units:
    print(f"Processing {num_units} hidden units...")
    # Initialize a 3-layer autoencoder
    autoencoder_model = Autoencoder(num_units)

    # Train autoencoder model
    autoencoder_model.train_model()

    # Calculate MSE of test digit
    mse = autoencoder_model.eval_model(test_digit)
    mse_results[num_units] = mse

    # Store losses to plot learning curves
    val_loss_results[num_units] = autoencoder_model.training_history.history['val_loss']

print(f"Results for prediction on digit {test_digit}...")
plt.figure(figsize=(12, 8))
for num_units in num_hidden_units:
    print(f"Model with {num_units} hidden units resulted in MSE of {mse}.")

    # Plot learning curve vs iterations
    plt.plot(val_loss_results[num_units], label=f'Test/Val (num hidden units: {num_units})')

# Visualize learning curve vs iterations across all trained models
plt.title('Learning Curves vs Epochs (Test/Validation Dataset)')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
