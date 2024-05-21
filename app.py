from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import inference

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        try:
            file.save(file_path)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_' + filename)
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
        
        try:
            print("INFERENCE RAN")
           #inference.run_inference(file_path, output_file_path)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        # Provide URLs for the frontend
        input_video_url = f"/uploads/{filename}"
        output_video_url = f"/outputs/output_{filename}"

        return jsonify({"input_video": input_video_url, "output_video": output_video_url}), 200

@app.route('/uploads/<filename>')
def get_upload(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/outputs/<filename>')
def get_output(filename):
    try:
        print(filename)
        print(app.config['OUTPUT_FOLDER'])
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=True)
