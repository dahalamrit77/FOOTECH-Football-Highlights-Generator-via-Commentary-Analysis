from faster_whisper import WhisperModel
import os
import ffmpeg
import json  # Import the json module to handle JSON file operations


def preprocess_audio(input_file, output_file):
    """
    Preprocess the input audio file by normalizing and converting it to mono with a sampling rate of 16kHz.
    """
    ffmpeg.input(input_file).output(output_file, ac=1, ar=16000).run(overwrite_output=True)
    print(f"Audio preprocessed and saved to {output_file}")


def transcribe_audio_with_timestamps(audio_file, model_size="small", output_json_file="transcription.json"):
    """
    Transcribe the audio file using Faster-Whisper with CUDA and save the transcription to a JSON file.
    """
    model = WhisperModel(model_size, device="cuda", compute_type="float16")  # Enable GPU
    segments, _ = model.transcribe(audio_file, word_timestamps=True)


    transcription_with_timestamps = []
    current_sentence = ""
    start_time = None


    for segment in segments:
        if not current_sentence:
            start_time = segment.start


        current_sentence += segment.text + " "


        if segment.text.strip().endswith("."):
            transcription_with_timestamps.append({
                "start": f"{start_time:.2f}s",  # Ensure the timestamp is in string format with "s"
                "end": f"{segment.end:.2f}s",  # Format end time similarly
                "sentence": current_sentence.strip()
            })
            current_sentence = ""
            start_time = None


    if current_sentence:
        transcription_with_timestamps.append({
            "start": f"{start_time:.2f}s",  # Format start time
            "end": f"{segment.end:.2f}s",  # Format end time
            "sentence": current_sentence.strip()
        })


    # Save the transcription to a JSON file
    with open(output_json_file, "w", encoding="utf-8") as json_file:
        json.dump(transcription_with_timestamps, json_file, indent=4)


    print(f"Transcription saved to {output_json_file}")
   
    return transcription_with_timestamps  # Optional return if needed for further processing
