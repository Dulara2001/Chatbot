import torch
import torch.nn as nn

# Define a neural network class that inherits from nn.Module
class NeuralNet(nn.Module):
    # Constructor to initialize the neural network architecture
    def __init__(self, input_size, hidden_size, num_classes):
        # Call the constructor of the parent class (nn.Module)
        super(NeuralNet, self).__init__()

        # Define the first linear layer: input_size to hidden_size
        self.l1 = nn.Linear(input_size, hidden_size)

        # Define the second linear layer: hidden_size to hidden_size
        self.l2 = nn.Linear(hidden_size, hidden_size)

        # Define the third linear layer: hidden_size to num_classes
        self.l3 = nn.Linear(hidden_size, num_classes)

        # Define the activation function (Rectified Linear Unit - ReLU)
        self.relu = nn.ReLU()

    # Define the forward pass of the neural network
    def forward(self, x):
        # Pass input through the first linear layer
        out = self.l1(x)

        # Apply the ReLU activation function
        out = self.relu(out)

        # Pass the output through the second linear layer
        out = self.l2(out)

        # Apply the ReLU activation function
        out = self.relu(out)

        # Pass the output through the third linear layer
        out = self.l3(out)

        # No activation and no softmax at the end (intended for cross-entropy loss)
        return out
