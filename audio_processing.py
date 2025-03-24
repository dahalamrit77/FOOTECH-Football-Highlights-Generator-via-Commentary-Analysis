import os
import subprocess

def extract_audio(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, "extracted_audio.wav")

    
    command = f"ffmpeg -i \"{video_path}\" -q:a 0 -map a \"{audio_path}\""


    subprocess.run(command, shell=True, check=True)

    
    return audio_path
import os
import subprocess

