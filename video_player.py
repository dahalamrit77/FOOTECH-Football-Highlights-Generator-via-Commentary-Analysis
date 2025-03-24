import tkinter as tk
import cv2
import imageio
from PIL import Image, ImageTk

def play_video(video_path):
    """Function to play the extracted highlight video in a Tkinter window."""
    cap = imageio.get_reader(video_path)

    def update_frame():
        try:
            frame = cap.get_next_data()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(img)
            label.config(image=img)
            label.image = img
            window.after(25, update_frame)  
        except Exception:
            cap.close()
            window.destroy()

    window = tk.Toplevel()
    window.title("Highlight Playback")
    window.geometry("800x450")

    label = tk.Label(window)
    label.pack()
    update_frame()

    window.mainloop()

if __name__ == "__main__":
    play_video("Goal_Clips/extracted_clip.mp4")
