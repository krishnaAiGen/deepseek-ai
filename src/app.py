from langchain_ollama.llms import OllamaLLM
from flask import Flask, jsonify, request
from threading import Lock

# Initialize the Flask app
app = Flask(__name__)

# Initialize the model
model = OllamaLLM(model="mistral", temperature=1)

# Create a lock for thread safety
request_lock = Lock()

@app.route('/chat', methods=['POST'])
def chat():
    with request_lock:  # Ensure sequential processing
        try:
            print("Processing request")
            data = request.get_json()
            input_text = data.get('text')

            if not input_text:
                return jsonify({"error": "No text provided"}), 400

            # Get response from the model
            output = model.invoke(input_text)

            return jsonify({
                "response": output
            })

        except Exception as e:
            print(f"Request processing error: {e}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, threaded=True)  # Ensure the server can handle multiple threads
