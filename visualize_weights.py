from autoencoder_class import Autoencoder
import matplotlib.pyplot as plt

test_digit = 3
num_hidden_units = 20

print(f"Processing {num_hidden_units} hidden units...")
# Initialize a 3-layer autoencoder
autoencoder_model = Autoencoder(num_hidden_units)

# Train autoencoder model
autoencoder_model.train_model()

# Obtain weights converging to each neuron for hidden layer
weights = autoencoder_model.extract_layer_weights(1)

# Visualize weights (each img 28x28)
sub_img_size = 28
plt.figure(figsize=(12, 6))
plt.suptitle(f'Set of weights for model with {num_hidden_units} hidden units')
for i in range(num_hidden_units):
    plt.subplot(2, num_hidden_units//2, i+1)

    img = weights[:, i].reshape(sub_img_size, sub_img_size)

    plt.title(f'Neuron {i+1}')
    plt.axis('off')
    plt.imshow(img, cmap = "gray")

plt.show()