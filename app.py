from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import ffmpeg
import inference

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
COMPRESSED_FOLDER = 'compressed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['COMPRESSED_FOLDER'] = COMPRESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB limit


def compress_video(input_path, output_path):
    try:
        ffmpeg.input(input_path).output(output_path, vcodec='libx264', crf=28).run()
        return True
    except Exception as e:
        print(f"Error compressing video: {e}")
        return False


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
        compressed_file_path = os.path.join(app.config['COMPRESSED_FOLDER'], 'compressed_' + filename)
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
        os.makedirs(app.config['COMPRESSED_FOLDER'], exist_ok=True)

        try:
            print("INFERENCE RAN")
            # Uncomment the following line to run the inference
            # inference.run_inference(file_path, output_file_path)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        if not os.path.exists(compressed_file_path):
            if not compress_video(output_file_path, compressed_file_path):
                return jsonify({"error": "Failed to compress video"}), 500

        # Provide URLs for the frontend
        input_video_url = f"/uploads/{filename}"
        output_video_url = f"/compressed/compressed_{filename}"

        return jsonify({"input_video": input_video_url, "output_video": output_video_url}), 200


@app.route('/uploads/<filename>')
def get_upload(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route('/compressed/<filename>')
def get_compressed(filename):
    try:
        return send_from_directory(app.config['COMPRESSED_FOLDER'], filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(COMPRESSED_FOLDER, exist_ok=True)
    app.run(debug=True)
