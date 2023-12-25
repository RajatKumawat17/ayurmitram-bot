from flask import Flask, url_for,redirect,render_template, request, jsonify
from get_intent import get_response, get_intent

app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    try:
        text = request.get_json().get("message")
        print(f"Received message: {text}")
        response = get_response(text)
        print(f"Response from get_response: {response}")
        message = {"answer": response}
        print(message, '^^^^^^^'),
        return jsonify(message)
    except Exception as e:
        print(f"Error in prediction route: {e}")
        return jsonify({"error": "Internal Server Error"}), 500




if __name__ == "__main__":
    app.run(debug=True)