import os
import json
import re
import torch  
from sentence_transformers import SentenceTransformer, util
import numpy as np
from goal_keywords import goal_keywords  




TRANSCRIPTION_FILE = "D:/FOOTECH/backend/OUTPUT/TRANSCRIPTION/transcription_with_timestamps.txt"
OUTPUT_JSON_FILE = "D:/FOOTECH/backend/OUTPUT/TRANSCRIPTION/detected_goal_segments.json"


SIMILARITY_THRESHOLD = 0.65  
SECOND_STAGE_THRESHOLD = 0.75  


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


model = SentenceTransformer("all-mpnet-base-v2")
model.to(device)  # Move model to GPU if available


goal_embeddings = model.encode(goal_keywords, convert_to_tensor=True).to(device)



def parse_transcription_file(file_path):
    segments = []
    if not os.path.exists(file_path):
        print(f"Transcription file not found: {file_path}")
        return segments

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split("]:")
        if len(parts) != 2:
            continue

        timestamps, sentence = parts
        sentence = sentence.strip()

        try:
            timestamps = timestamps.strip("[]")  # Remove brackets
            start_time_str, end_time_str = timestamps.split(" - ")
        
            start_time = float(start_time_str.replace('s', '').strip())
            end_time = float(end_time_str.replace('s', '').strip())
            print(f"Parsed: Start={start_time}, End={end_time}, Sentence={sentence}")
        except ValueError:
            print(f"Error converting time: {timestamps}")
            continue

        segments.append({
            "start": start_time,
            "end": end_time,
            "sentence": sentence
        })

    return segments

def adjust_threshold(similarity_scores, initial_threshold=SIMILARITY_THRESHOLD):
    """
    Adjust the similarity threshold dynamically based on the average similarity score.
    """
    if not similarity_scores:
        return initial_threshold

    avg_similarity = np.mean(similarity_scores)
    print(f"Average similarity: {avg_similarity:.2f}")

    #  (tighten detection)
    if avg_similarity > 0.85:
        adjusted_threshold = initial_threshold + 0.05
    #  (loosen detection)
    elif avg_similarity < 0.55:
        adjusted_threshold = initial_threshold - 0.05
    else:
        adjusted_threshold = initial_threshold

    adjusted_threshold = min(max(adjusted_threshold, 0.5), 1.0)
    print(f"Adjusted threshold: {adjusted_threshold:.2f}")
    return adjusted_threshold

def is_goal_related(sentence):
    """
    Pre-filter to check if the sentence appears to be about a goal.
    - Converts sentence to lowercase.
    - Checks for negation patterns (e.g., "not goal", "no score", "missed goal", "disallowed goal", "not given goal")
      to avoid false positives.
    - Returns True if no negation patterns are detected.
    """
    sentence_lower = sentence.lower()

   
    negation_pattern = r"\b(?:not|no|missed|disallowed|not\s+given)\s+(?:goal|score|strike|header|chance)\b"
    if re.search(negation_pattern, sentence_lower):
        return False

    return True

def find_goal_segments(transcription_segments, similarity_threshold=SIMILARITY_THRESHOLD):
    """
    Analyze each transcription segment using SBERT to detect if it contains goal-related commentary.
    Incorporates an initial filtering step with is_goal_related() to remove clear negations.
    Implements a two-stage filtering approach: for sentences with ambiguous similarity scores (between the initial
    and stricter threshold), an explicit keyword check is applied.
    """
    detected_segments = []
    similarity_scores = []

    for segment in transcription_segments:
        sentence = segment.get("sentence", "")
        if not sentence:
            continue

       
        if not is_goal_related(sentence):
            continue

        
        sentence_embedding = model.encode(sentence, convert_to_tensor=True).to(device)
        
        
        similarity_scores_ = util.pytorch_cos_sim(sentence_embedding, goal_embeddings)
        max_similarity = float(np.max(similarity_scores_.cpu().numpy()))
        similarity_scores.append(max_similarity)

        
        if similarity_threshold < max_similarity < SECOND_STAGE_THRESHOLD:
            if not any(keyword.lower() in sentence.lower() for keyword in goal_keywords):
                continue

        if max_similarity > similarity_threshold:
            segment_info = {
                "start": segment.get("start"),
                "end": segment.get("end"),
                "sentence": sentence,
                "max_similarity": max_similarity
            }
            detected_segments.append(segment_info)

   
    adjusted_threshold = adjust_threshold(similarity_scores, similarity_threshold)
    return detected_segments, adjusted_threshold

def save_detected_segments(detected_segments, output_file):
    """
    Saves the detected goal-related segments to a JSON file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(detected_segments, f, indent=4)
    print(f"Detected goal segments saved to: {output_file}")



def main():
   
    transcription_segments = parse_transcription_file(TRANSCRIPTION_FILE)
    if not transcription_segments:
        print("No transcription segments found. Exiting.")
        return

   
    detected_segments, adjusted_threshold = find_goal_segments(transcription_segments, SIMILARITY_THRESHOLD)

    
    print("Detected goal segments:")
    for seg in detected_segments:
        print(f"Time: {seg['start']} - {seg['end']} | Sentence: {seg['sentence']} | Similarity: {seg['max_similarity']:.2f}")

   
    save_detected_segments(detected_segments, OUTPUT_JSON_FILE)
    print(f"Final adjusted threshold: {adjusted_threshold:.2f}")

if __name__ == "__main__":
    main()
