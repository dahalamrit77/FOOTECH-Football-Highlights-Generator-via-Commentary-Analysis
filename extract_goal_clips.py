import os
import json
import subprocess

def get_video_duration(video_path):
    """
    Retrieves the total duration of the video in seconds using ffprobe.
    
    :param video_path: Path to the video file.
    :return: Duration of the video in seconds as a float, or None if unavailable.
    """
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", video_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"Error obtaining video duration: {e}")
        return None

def load_pitch_analysis(pitch_analysis_file):
    """
    Load pitch analysis data from a JSON file.
    
    :param pitch_analysis_file: Path to the JSON file containing pitch analysis.
    :return: List of dictionaries with pitch analysis data.
    """
    with open(pitch_analysis_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_detected_goals(detected_goals_file):
    """
    Load detected goal events from a JSON file.
    
    :param detected_goals_file: Path to the JSON file containing detected goal segments.
    :return: List of dictionaries with goal event data, with timestamps converted to floats.
    """
    with open(detected_goals_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for entry in data:
        if isinstance(entry.get("start"), str):
            entry["start"] = float(entry["start"].replace("s", "").strip())
        if isinstance(entry.get("end"), str):
            entry["end"] = float(entry["end"].replace("s", "").strip())
    return data

def extract_goal_clips(video_path, pitch_analysis_file, output_folder, clip_duration=20, use_ffmpeg=False):
    """
    Extracts video clips from the given video based on high pitch timestamps that match goal-related commentary.
    Cross matches high pitch timestamps with goal commentary timestamps (from detected_goals.json) using a tolerance window.
    
    :param video_path: Path to the uploaded video file.
    :param pitch_analysis_file: Path to the pitch analysis JSON file.
    :param output_folder: Folder where extracted clips will be saved.
    :param clip_duration: Total duration of each extracted clip (in seconds).
    :param use_ffmpeg: Optional parameter to specify whether to use FFmpeg (default False, not used internally).
    :return: List of file paths to the extracted clips.
    """
    # Load pitch analysis data
    pitch_data = load_pitch_analysis(pitch_analysis_file)
    
    # Load detected goal events (from keyword matching) from detected_goals.json located in the same folder as pitch_analysis_file
    detected_goals_file = os.path.join(os.path.dirname(pitch_analysis_file), "detected_goals.json")
    goal_data = load_detected_goals(detected_goals_file)
    
    # Define tolerance window (in seconds) for matching pitch and goal commentary timestamps
    tolerance = 12.0  # Increased from 8.0 to 12.0 seconds for better matching
    
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    clips = []
    half_duration = clip_duration / 2.0  # For 20 seconds, half_duration is 10 seconds
    
    # Get total video duration to ensure clip boundaries are valid
    video_duration = get_video_duration(video_path)
    if video_duration is None:
        print("Could not retrieve video duration. Proceeding without boundary checks.")
    
    # To prevent duplicate extraction for overlapping events
    last_clip_end = -1

    # Loop through each pitch analysis segment
    for idx, segment in enumerate(pitch_data):
        pitch_timestamp = segment.get("timestamp")
        if pitch_timestamp is None:
            continue

        # Check if this pitch timestamp has a matching goal event within the tolerance window
        match_found = False
        matched_goal = None
        for goal in goal_data:
            if (goal["start"] - tolerance) <= pitch_timestamp <= (goal["end"] + tolerance):
                match_found = True
                matched_goal = goal
                break

        if not match_found:
            continue  # Skip this pitch segment if no matching goal event is found

        # Calculate event center using the detected goal segment from commentary (simple midpoint)
        event_center = (matched_goal["start"] + matched_goal["end"]) / 2.0

        # Debug: Print matched goal event and event center
        print(f"Matched goal event: Start = {matched_goal['start']}, End = {matched_goal['end']}")
        print(f"Calculated event center (midpoint): {event_center}")

        # Compute start and end times for clip extraction using the event center (ensuring start_time is not negative)
        start_time = max(event_center - half_duration, 0)
        end_time = event_center + half_duration

        # Adjust clip boundaries if end_time exceeds video duration
        if video_duration and end_time > video_duration:
            start_time = max(video_duration - clip_duration, 0)
            end_time = start_time + clip_duration

        # Prevent duplicate extraction for overlapping clips
        if start_time < last_clip_end:
            continue

        # Update last_clip_end to the end time of the current clip
        last_clip_end = end_time

        # Debug: Print final clip extraction times
        print(f"Extracting clip from {start_time} to {end_time} seconds")

        # Output file name for the individual clip
        clip_filename = os.path.join(output_folder, f"goal_clip_{idx+1}_{start_time:.2f}-{end_time:.2f}.mp4")

        # Run FFmpeg command to extract clip
        command = [
            "ffmpeg", "-y",  # Overwrite existing files without asking
            "-i", video_path,  # Input video file
            "-ss", str(start_time),  # Start time
            "-t", str(clip_duration),  # Duration
            "-c:v", "h264_nvenc", "-c:a", "aac", "-strict", "experimental",  # Encoding options using CUDA (NVIDIA GPU)
            clip_filename  # Output file
        ]

        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            clips.append(clip_filename)
            print(f"Extracted clip {idx+1}: {clip_filename}")
        except subprocess.CalledProcessError:
            print(f"Error extracting clip {idx+1}")

    return clips

def merge_clips(clip_paths, output_path):
    """
    Concatenates multiple video clips into a single video file using FFmpeg.
    The list of clips is written to a temporary file which is then used for concatenation.
    
    :param clip_paths: List of file paths of the clips to merge.
    :param output_path: Path for the final concatenated clip.
    """
    temp_file = os.path.join(os.path.dirname(output_path), "concat_list.txt")
    with open(temp_file, "w") as f:
        for clip in clip_paths:
            # FFmpeg requires the file paths to be prefixed with "file '...'".
            f.write(f"file '{clip}'\n")
    
    if os.path.exists(output_path):
        os.remove(output_path)
    
    command = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", temp_file,
        "-c", "copy", output_path
    ]
    
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"Final concatenated clip saved as: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error concatenating clips: {e}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def main():
    # Define paths based on your project structure:
    video_path = "D:/FOOTECH/backend/uploaded/uploaded_video.mp4"
    pitch_analysis_file = "D:/FOOTECH/backend/TRANSCRIPTIONS/processed_audio_high_pitch_analysis.json"
    output_folder = "D:/FOOTECH/backend/Goal_Clips"

    # Extract clips (each clip is 20 seconds long around the matched event center)
    clips = extract_goal_clips(video_path, pitch_analysis_file, output_folder, clip_duration=20)
    
    # If clips were extracted, concatenate them into a single clip
    if clips:
        final_clip = os.path.join(output_folder, "extracted_clip.mp4")
        merge_clips(clips, final_clip)
        
        # After merging, remove individual clips, leaving only the concatenated clip in the folder
        for clip in clips:
            if os.path.exists(clip):
                os.remove(clip)
                
        print("All goal-related clips have been extracted and merged into a single clip:")
        print(final_clip)
    else:
        print("No clips were extracted.")

if __name__ == "__main__":
    main()
