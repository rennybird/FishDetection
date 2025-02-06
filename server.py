from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Fish count storage
fish_counts = {"left": {}, "right": {}}

@app.route("/fish_counts", methods=["GET"])
def get_fish_counts():
    return jsonify(fish_counts)

@app.route("/update_fish_counts", methods=["POST"])
def update_fish_counts():
    data = request.json
    direction = data.get("direction")
    class_id = data.get("class_id")
    count = data.get("count")
    if direction in fish_counts:
        fish_counts[direction][class_id] = count
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)  # Run on port 5000