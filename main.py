from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FissionAI")

# Rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# Load pre-trained model and tokenizer
MODELS = {
    "gpt2": "gpt2",
    "blenderbot": "facebook/blenderbot-400M-distill",
}
current_model = "gpt2"  # Default model
tokenizer = AutoTokenizer.from_pretrained(MODELS[current_model])
model = AutoModelForCausalLM.from_pretrained(MODELS[current_model])

# Create a text generation pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Conversation history
conversation_history = {}

# Home route
@app.route("/")
def home():
    return "Welcome to Fission AI! Use the /chat endpoint to interact."

# Chat route
@app.route("/chat", methods=["POST"])
@limiter.limit("5 per minute")  # Rate limit: 5 requests per minute
def chat():
    global conversation_history

    # Get user input from the request
    data = request.json
    user_input = data.get("message", "")
    user_id = data.get("user_id", "default_user")  # Unique ID for each user

    if not user_input:
        logger.warning(f"User {user_id} sent an empty message.")
        return jsonify({"error": "No message provided"}), 400

    # Retrieve conversation history for the user
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    # Add user input to conversation history
    conversation_history[user_id].append(f"User: {user_input}")

    # Generate a response using the chatbot pipeline
    try:
        prompt = "\n".join(conversation_history[user_id]) + "\nAI:"
        response = chatbot(prompt, max_length=100, num_return_sequences=1)
        bot_response = response[0]["generated_text"].split("AI:")[-1].strip()

        # Add AI response to conversation history
        conversation_history[user_id].append(f"AI: {bot_response}")

        # Log the interaction
        logger.info(f"User {user_id}: {user_input} -> AI: {bot_response}")

        # Return the response as JSON
        return jsonify({"response": bot_response})

    except Exception as e:
        logger.error(f"Error generating response for user {user_id}: {e}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

# Model selection route
@app.route("/set_model", methods=["POST"])
def set_model():
    global current_model, tokenizer, model, chatbot

    data = request.json
    new_model = data.get("model", "")

    if new_model not in MODELS:
        return jsonify({"error": "Invalid model specified"}), 400

    # Update the model
    current_model = new_model
    tokenizer = AutoTokenizer.from_pretrained(MODELS[current_model])
    model = AutoModelForCausalLM.from_pretrained(MODELS[current_model])
    chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

    logger.info(f"Model changed to {current_model}")
    return jsonify({"message": f"Model updated to {current_model}"})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)