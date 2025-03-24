# FOOTECH-Football-Highlights-Generator-via-Commentary-Analysis
# 🏟️ FOOTECH: Football Highlights Generator via Commentary Analysis

## 📌 Project Overview

FOOTECH is a standalone software application designed to automatically generate highlight reels from football match footage by analyzing audio commentary metadata. The system focuses on identifying key events, particularly goals, by correlating high-pitch moments in the audio with goal-related keywords detected through speech-to-text analysis.

---

## ⚙️ Core Logic

1. **User Uploads Video**  
   - The video file is stored in the backend for processing.  

2. **Audio Extraction**  
   - The audio is extracted from the video using **FFmpeg**.  

3. **Speech-to-Text Conversion**  
   - The speech is transcribed into text using **Whisper AI**.  

4. **Pitch Analysis**  
   - **Librosa** is used to analyze audio pitch and detect high-pitched moments indicative of important events.  

5. **Keyword Matching**  
   - **SBERT (Sentence-BERT)** is used to match goal-related keywords from the transcribed commentary.  

6. **Combining Pitch and Keyword Analysis**  
   - The system filters potential goal moments by correlating high pitch values with goal-related keywords.  

7. **Extracting Goal Clips**  
   - The corresponding video segments are extracted using **FFmpeg**.  

8. **Generating Highlight Reel**  
   - The extracted clips are compiled into a highlight reel.  

9. **Returning Video**  
   - The final highlight reel is sent back to the frontend for user access.  

---

## 📁 Required Directory Structure

Before running the project, ensure the following directories are manually created in the project root directory:

📦 FOOTECH ├── 📁 uploaded # To store uploaded video files 
            ├── 📁 OUTPUT # To store the final highlight reels 
            ├── 📁 TRANSCRIPTIONS # To store transcribed text files 
            ├── 📁 Goal_Clips # To store individual goal clips


---

## 🔧 Prerequisites

1. **CUDA Installation**  
   - Ensure **CUDA** is installed for GPU acceleration, which is essential for **Whisper AI**.  
2. **Python 3.8 or Higher**  
3. **FFmpeg** for audio extraction.  
4. **Required Python Libraries** — Listed in `requirements.txt`.  

---

## 🚀 Installation

1. **Clone the Repository**  

git clone https://github.com/yourusername/FOOTECH.git
cd FOOTECH

2. **Install Dependencies** 
pip install -r requirements.txt

## 🔧 Goal of the Project
The aim of this project is to automate the creation of football highlight reels by analyzing commentary metadata. The system detects high-pitched moments in the audio and correlates these with goal-related keywords to identify significant match events, automatically extracting and compiling them into concise highlight clips.

## Technologies Used
Frontend: Tkinter (Python)

Backend: Flask (Python)

Speech-to-Text: Whisper AI

Pitch Analysis: Librosa

Semantic Analysis: SBERT

Video Processing: FFmpeg
