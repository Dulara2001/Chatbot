from flask import Flask, render_template, request, jsonify
from chat import get_response  # Import the get_response function from chat.py

app = Flask(__name__)

# Define the route for the home page
@app.route('/')
def index():
    # Render the base.html template
    return render_template('index.html')

# Define the route for handling predictions based on user input
@app.route('/predict', methods=['POST'])
def predict():
    # Extract the user input message from the JSON payload
    text = request.get_json().get("message")

    # Get a response from the chatbot based on the user input
    response = get_response(text)

    # Create a JSON response containing the chatbot's answer
    message = {"answer": response}

    # Return the JSON response
    return jsonify(message)

# Run the Flask app if this script is executed directly
if __name__ == "__main__":
    app.run(debug=True)
