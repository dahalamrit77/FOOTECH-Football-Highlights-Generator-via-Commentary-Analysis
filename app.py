import threading
from flask import Flask, request, jsonify
import os
import uuid
import time
from werkzeug.utils import secure_filename
from audio_processing import extract_audio
import ffmpeg
import shutil
import subprocess
from faster_whisper import WhisperModel
from pitch_analysis import save_high_pitch_analysis
from extract_goal_clips import extract_goal_clips  
from sentence_transformers import SentenceTransformer, util
import json  
from goal_keywords import goal_keywords  

app = Flask(__name__)

UPLOAD_FOLDER = "D:/FOOTECH/backend/uploaded"
OUTPUT_FOLDER = "D:/FOOTECH/backend/OUTPUT"
TRANSCRIPTION_FOLDER = "D:/FOOTECH/backend/TRANSCRIPTIONS"
GOAL_CLIPS_FOLDER = "D:/FOOTECH/backend/Goal_Clips"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)
os.makedirs(GOAL_CLIPS_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mkv", "mov", "flv", "wmv"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024 * 1024  # 5GB max file size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TIMEOUT'] = 60 * 60

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_audio(input_file, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)  # Automatically overwrite existing file
    ffmpeg.input(input_file).output(output_file, ac=1, ar=16000).run(overwrite_output=True)
    print(f"Audio preprocessed and saved to {output_file}")

def transcribe_audio_chunked(audio_file, model_size="small"):
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, _ = model.transcribe(audio_file, word_timestamps=True)
    
    transcription = []
    for segment in segments:
        transcription.append({
            "start": f"{segment.start:.2f}s",
            "end": f"{segment.end:.2f}s",
            "sentence": segment.text.strip()
        })
    
    return transcription

def save_transcription_to_json(transcriptions, output_file):
    with open(output_file, "w") as json_file:
        json.dump(transcriptions, json_file, indent=4)
    print(f"Transcription saved to {output_file}")

def detect_goals_using_sbert(transcription_file):
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    
    with open(transcription_file, "r") as file:
        transcriptions = json.load(file)
    
    goal_embeddings = model.encode(goal_keywords, convert_to_tensor=True).to('cuda')
    
    goal_timestamps = []
    for entry in transcriptions:
        sentence = entry["sentence"]
        sentence_embedding = model.encode([sentence], convert_to_tensor=True).to('cuda')
        
        similarity_scores = util.pytorch_cos_sim(sentence_embedding, goal_embeddings)
        
        if similarity_scores.max().item() > 0.6:
            goal_timestamps.append(entry)
    
    goal_timestamps_file = os.path.join(TRANSCRIPTION_FOLDER, "detected_goals.json")
    with open(goal_timestamps_file, "w") as json_file:
        json.dump(goal_timestamps, json_file, indent=4)
    print(f"Goal-related sentences and timestamps saved to {goal_timestamps_file}")
    
    return goal_timestamps_file

def process_audio_and_transcription(audio_path, transcription_folder, pitch_analysis_event):
    try:
        processed_audio_path = os.path.join(transcription_folder, "processed_audio.wav")
        preprocess_audio(audio_path, processed_audio_path)
        transcriptions = transcribe_audio_chunked(processed_audio_path)
        transcription_file_path = os.path.join(transcription_folder, "transcription_with_timestamps.json")
        save_transcription_to_json(transcriptions, transcription_file_path)
        pitch_analysis_file = save_high_pitch_analysis(processed_audio_path)
        
        detected_goals_file = detect_goals_using_sbert(transcription_file_path)
        
        pitch_analysis_event.set()
        return pitch_analysis_file, detected_goals_file
    except Exception as e:
        print(f"Audio processing failed: {str(e)}")
        pitch_analysis_event.set()

def extract_audio_in_background(video_path, output_folder, transcription_folder, pitch_analysis_event):
    try:
        audio_filename = "extracted_audio.wav"
        audio_path = os.path.join(output_folder, audio_filename)
        
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        extracted_audio_path = extract_audio(video_path, output_folder)
        process_audio_and_transcription(extracted_audio_path, transcription_folder, pitch_analysis_event)
    except Exception as e:
        print(f"Audio processing failed: {str(e)}")

def merge_clips(clip_paths, output_path):
    temp_file = os.path.join(os.path.dirname(output_path), "concat_list.txt")
    with open(temp_file, "w") as f:
        for clip in clip_paths:
            f.write(f"file '{clip}'\n")
    
    if os.path.exists(output_path):
        os.remove(output_path)
    
    command = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", temp_file,
        "-c", "copy", output_path
    ]
    
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error concatenating clips: {e}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

@app.route("/upload/", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        allowed_formats = ", ".join(f".{ext}" for ext in ALLOWED_EXTENSIONS)
        return jsonify({"error": f"Invalid file type. Allowed types are: {allowed_formats}"}), 400
    
    original_ext = file.filename.rsplit(".", 1)[1].lower()
    fixed_filename = f"uploaded_video.{original_ext}"
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], fixed_filename)
    file.save(video_path)
    
    pitch_analysis_event = threading.Event()
    threading.Thread(target=extract_audio_in_background, args=(video_path, OUTPUT_FOLDER, TRANSCRIPTION_FOLDER, pitch_analysis_event)).start()
    
    pitch_analysis_event.wait()
    
    pitch_analysis_path = os.path.join(TRANSCRIPTION_FOLDER, "processed_audio_high_pitch_analysis.json")
    goal_clips_folder = os.path.join("D:/FOOTECH/backend", "Goal_Clips")
    os.makedirs(goal_clips_folder, exist_ok=True)
    
    extracted_clips = extract_goal_clips(video_path, pitch_analysis_path, goal_clips_folder, clip_duration=20)
    
    final_clip_path = ""
    if extracted_clips:
        final_clip_path = os.path.join(goal_clips_folder, "extracted_clip.mp4")
        merge_clips(extracted_clips, final_clip_path)
        for clip in extracted_clips:
            if os.path.exists(clip):
                os.remove(clip)
    
    return jsonify({
        "message": "File uploaded successfully.",
        "video_path": video_path,
        "final_clip": final_clip_path
    }), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000, threaded=True, use_reloader=False)
