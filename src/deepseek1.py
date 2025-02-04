from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from flask import Flask, jsonify, request
from threading import Lock

# Initialize the Flask app
app = Flask(__name__)

# Define base templates for different tasks
TEMPLATES = {
    "POST_SUMMARY": """
    You are a helpful assistant that summarizes Polkadot governance posts.
    Focus on key points like:
    - The proposal's main objective
    - Requested funding amounts (if any)
    - Technical changes proposed
    - Expected impact on the ecosystem
    - Key beneficiaries

    IMPORTANT: Respond ONLY with the markdown summary. Do not include any introductory text, acknowledgments, or additional commentary.
    Keep the summary concise and technical. Use blockchain terminology appropriately but do not overdo it.
    Format your response in markdown with appropriate headers and bullet points.

    Post: {content}

    Answer:
    """,
    "COMMENTS_SUMMARY": """
    You are a helpful assistant that summarizes discussions on Polkadot governance proposals.
    Analyze the sentiment and provide a breakdown in the following format:

    Overall X% of users are feeling optimistic. [Summarize main positive points]
    Overall Y% of users are feeling neutral. [Summarize neutral/questioning points]
    Overall Z% of users are feeling against it. [Summarize main concerns]

    Important technical points raised:
    - [List key technical discussions]

    Key questions from the community:
    - [List main questions]

    IMPORTANT: Respond ONLY with the markdown formatted analysis. Do not include any introductory text, acknowledgments, or additional commentary.

    Discussion: {content}

    Answer:
    """,
    "CONTENT_SPAM_CHECK": """
    You are a helpful assistant that evaluates Polkadot governance content for spam.
    Return only 'true' if the content matches any spam criteria, or 'false' if it's legitimate content.

    Check for:
    - Irrelevant promotional content
    - Off-topic discussions
    - Duplicate proposals
    - Malicious links
    - Impersonation attempts
    - Low-quality or automated content
    - Cryptocurrency scams or unauthorized token promotions
    - Phishing attempts
    - Excessive cross-posting
    - Unrelated commercial advertising

    Content: {content}

    Answer:
    """
}

# Initialize the model
model = OllamaLLM(model="mistral", temperature=1)

# Create a lock for thread safety
request_lock = Lock()

# Basic Chat Endpoint
@app.route('/chat', methods=['POST'])
def basic_chat():
    with request_lock:  # Ensure sequential processing
        try:
            print("Processing basic chat request")
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

# Task-Specific Chat Endpoint
@app.route('/chat1', methods=['POST'])
def task_chat():
    with request_lock:
        try:
            data = request.get_json()
            input_text = data.get('text')
            task_type = data.get('task')  # Specify the task type

            if not input_text or not task_type:
                return jsonify({"error": "Both 'text' and 'task' are required."}), 400

            # Get the corresponding template
            template_str = TEMPLATES.get(task_type)
            if not template_str:
                return jsonify({"error": f"Invalid task type '{task_type}'."}), 400

            # Create the prompt template
            prompt = ChatPromptTemplate.from_template(template_str)

            # Create the chain: prompt → model
            chain = prompt | model

            # Invoke the chain
            output = chain.invoke({"content": input_text})

            return jsonify({
                "response": output
            })

        except Exception as e:
            print(f"Request processing error: {e}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, threaded=True)
