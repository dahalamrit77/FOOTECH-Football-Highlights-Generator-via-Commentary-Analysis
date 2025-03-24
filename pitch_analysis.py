import librosa
import numpy as np
import json


def determine_dynamic_threshold(pitches, k=2.5):
    
    all_pitches = pitches[pitches > 0]  

    if len(all_pitches) == 0:
        return None  

    mean_pitch = np.mean(all_pitches)
    std_pitch = np.std(all_pitches)
    
    q1 = np.percentile(all_pitches, 25)
    q3 = np.percentile(all_pitches, 75)
    iqr = q3 - q1

    
    ratio = std_pitch / mean_pitch
    if ratio > 0.5:
        adaptive_percentile = 65
    elif ratio < 0.2:
        adaptive_percentile = 80
    else:
        adaptive_percentile = 75
    percentile_value = np.percentile(all_pitches, adaptive_percentile)

    
    adaptive_k = k * (1 + std_pitch / mean_pitch)
    
   
    threshold = min(q3 + adaptive_k * iqr, percentile_value)

    print(f"Dynamic threshold: {round(threshold, 2)} Hz (Mean: {round(mean_pitch, 2)}, Std: {round(std_pitch, 2)}, {adaptive_percentile}th Percentile: {round(percentile_value, 2)})")
    return threshold


def perform_pitch_analysis(audio_path, time_interval=0.5, min_duration=2, min_separation=10):
    """
    Perform pitch analysis on an audio file, capturing pitch data with high pitch moment separation.
    """
    try:
        
        y, sr = librosa.load(audio_path, sr=None)
        print(f"Audio loaded from {audio_path}, sample rate: {sr}")

       
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

        
        pitch_threshold = determine_dynamic_threshold(pitches, k=2.5)
        if pitch_threshold is None:
            print("No pitch data found, skipping analysis.")
            return []

        
        total_duration = len(y) / sr
        num_frames = pitches.shape[1]
        frame_time_step = total_duration / num_frames  
        step_size = max(1, int(round(time_interval / frame_time_step)))  

       
        high_pitch_segments = []
        consecutive_high_pitch_count = 0
        last_high_pitch_timestamp = -min_separation  
        potential_segments = []

        
        for t in range(0, num_frames, step_size):
            pitch_values = pitches[:, t]
            max_pitch = np.max(pitch_values) if pitch_values.any() else 0
            time_in_seconds = round(t * frame_time_step)
            
            
            neighbor_condition = False
            if t > 0:
                prev_max = np.max(pitches[:, t - 1])
                if prev_max > pitch_threshold * 0.8:
                    neighbor_condition = True
            if t < num_frames - 1:
                next_max = np.max(pitches[:, t + 1])
                if next_max > pitch_threshold * 0.8:
                    neighbor_condition = True

            if max_pitch > pitch_threshold or neighbor_condition:
                consecutive_high_pitch_count += 1
                if consecutive_high_pitch_count >= min_duration / time_interval:
                    potential_segments.append({
                        "timestamp": int(time_in_seconds),
                        "pitch": round(float(max_pitch), 2)
                    })
                    consecutive_high_pitch_count = 0
            else:
                consecutive_high_pitch_count = 0

       
        if potential_segments:
            high_pitch_segments.append(potential_segments[0])  
            last_timestamp = potential_segments[0]["timestamp"]
            
            for segment in potential_segments[1:]:
                current_timestamp = segment["timestamp"]
                if current_timestamp - last_timestamp >= min_separation:
                    high_pitch_segments.append(segment)
                    last_timestamp = current_timestamp

        print(f"Detected {len(high_pitch_segments)} high pitch segments after filtering.")
        return high_pitch_segments
    except Exception as e:
        print(f"Error during pitch analysis: {str(e)}")
        return []


def save_high_pitch_analysis(audio_path):
    """
    Runs pitch analysis and saves detected high pitch segments to a JSON file.
    """
    try:
        print(f"Performing pitch analysis on audio file: {audio_path}")

        high_pitch_segments = perform_pitch_analysis(audio_path)

        if not high_pitch_segments:
            print("Warning: No high pitch segments detected.")
            return None

        
        pitch_analysis_file = audio_path.replace(".wav", "_high_pitch_analysis.json")
        with open(pitch_analysis_file, "w") as file:
            json.dump(high_pitch_segments, file, indent=4)

        print(f"Pitch analysis saved to: {pitch_analysis_file}")
        return pitch_analysis_file
    except Exception as e:
        print(f"Error in pitch analysis: {str(e)}")
        return None
