# Import necessary libraries and modules
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Check if CUDA (GPU) is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from a JSON file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load pre-trained model and related data
FILE = "data.pth"
data = torch.load(FILE)

# Extract relevant information from the loaded data
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize and load the neural network model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Function to get a response based on user input
def get_response(msg):
    # Tokenize the user's input
    sentence = tokenize(msg)
    
    # Convert the tokenized sentence into a bag of words representation
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    
    # Convert the input to a PyTorch tensor and move it to the appropriate device (CPU or GPU)
    X = torch.from_numpy(X).to(device)

    # Forward pass to get predictions from the model
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    # Map the predicted label to the corresponding tag
    tag = tags[predicted.item()]

    # Calculate probabilities using softmax
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Check if the probability is above a certain threshold for confidence
    if prob.item() > 0.8:
        # Select a random response from the intents for the identified tag
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    # Fallback response if the confidence threshold is not met
    return "I do not understand..."

# Main interaction loop
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # Get user input
        sentence = input("You: ")
        if sentence == "quit":
            break

        # Get and print the model's response
        resp = get_response(sentence)
        print(resp)
