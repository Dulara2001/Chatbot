# Chatbot using PyTorch and Deep Learning

#### Watch Demo: https://drive.google.com/file/d/14qaZ3sS6U-xaZ-5iis7STCN8hjSTCWh1/view?usp=sharing

This project demonstrates the development of a chatbot using deep learning techniques, specifically focusing on a feed-forward neural network implemented with PyTorch. Below is an overview of the key steps and methodologies followed in this project:

---

## Overview of Techniques

### 1. **Preprocessing Pipeline**
![6](https://github.com/user-attachments/assets/48b5f0a1-fd56-4b7e-9e0c-4453de16e512)

The input text is preprocessed to convert it into a numerical format suitable for the neural network model. The steps include:

- **Tokenization**: Breaking down the input sentence into individual words.
- **Lowercasing and Stemming**: Converting words to lowercase and reducing them to their root forms to handle variations (e.g., "running" → "run").
- **Removing Punctuation**: Excluding non-alphanumeric characters to focus on meaningful tokens.
- **Bag of Words Representation**: Representing the sentence as a binary vector where each position indicates the presence (1) or absence (0) of a specific word in the vocabulary.

**Example**:  
Input Sentence: *"Is anyone there?"*  
- After tokenization: `["Is", "anyone", "there", "?"]`  
- After stemming and removing punctuation: `["is", "anyon", "there"]`  
- Bag of Words vector: `[0, 0, 0, 1, 0, 1, 0, 1]`

This vector is fed into the neural network as input.

---

### 2. **Feed-Forward Neural Network Architecture**
![3](https://github.com/user-attachments/assets/08862ff9-2b77-43f1-b5e1-bb9cf4b52137)

- **Input Layer**: Accepts the bag-of-words vector representing the input sentence.
- **Hidden Layers**: Processes the input using multiple layers of neurons to capture patterns and features.
- **Output Layer**: Produces a probability distribution over the possible classes (e.g., intents or responses) using the Softmax activation function.

**Example**:  
Input: *"How are you?"* → Bag of Words: `[1, 0, 0, 1, 0]`  
Output: Probability vector for intents: `[0.91, 0.01, 0.02, 0.01]`  
The model predicts the intent with the highest probability (e.g., 0.91 corresponds to "greeting").

---

## Project Highlights

1. **Model Training**:
   - The model is trained on labeled data, where each sentence is mapped to a specific intent or response category.
   - The loss function used is cross-entropy, and optimization is performed using gradient descent.

2. **Inference**:
   - Given a new sentence, the chatbot preprocesses it using the pipeline, feeds the bag-of-words vector into the trained model, and identifies the corresponding intent or response.

3. **Technologies Used**:
   - **PyTorch**: For building and training the deep learning model.
   - **Natural Language Processing (NLP)**: For text preprocessing.
   - **Softmax Activation**: For output probability normalization.

---

## Demo of system
![7](https://github.com/user-attachments/assets/a763ff21-ade6-4bbb-aba9-eb4743812b58)

