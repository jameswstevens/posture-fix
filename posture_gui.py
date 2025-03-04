import tkinter as tk
from tkinter import ttk
import subprocess
import sys
import os

class PostureDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Posture Detector Control Panel")
        self.root.geometry("400x300")
        
        self.is_running = False
        self.process = None
        self.reminder_time = tk.StringVar(value="30")
        
        self.create_widgets()
        
    def create_widgets(self):
        # Reminder time setting
        reminder_frame = ttk.LabelFrame(self.root, text="Settings", padding="10")
        reminder_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(reminder_frame, text="Reminder Time (seconds):").pack(side="left")
        reminder_entry = ttk.Entry(reminder_frame, textvariable=self.reminder_time, width=10)
        reminder_entry.pack(side="left", padx=5)
        
        # Control buttons
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(fill="x", padx=10, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start Detection", command=self.toggle_detection, style='Black.TButton')
        style = ttk.Style()
        style.configure('Black.TButton', foreground='black')
        self.start_button.pack(fill="x", pady=5)
        
        self.recalibrate_button = ttk.Button(button_frame, text="Recalibrate", command=self.recalibrate)
        self.recalibrate_button.pack(fill="x", pady=5)
        self.recalibrate_button["state"] = "disabled"
        
        # Status label
        self.status_label = ttk.Label(self.root, text="Status: Not Running")
        self.status_label.pack(pady=10)
        
    def toggle_detection(self):
        if not self.is_running:
            try:
                cmd = [sys.executable, "posture_detector.py", self.reminder_time.get()]
                self.process = subprocess.Popen(cmd)
                self.is_running = True
                self.start_button["text"] = "Stop Detection"
                self.recalibrate_button["state"] = "normal"
                self.status_label["text"] = "Status: Running"
            except Exception as e:
                self.status_label["text"] = f"Error: {str(e)}"
        else:
            if self.process:
                self.process.terminate()
            self.is_running = False
            self.start_button["text"] = "Start Detection"
            self.recalibrate_button["state"] = "disabled"
            self.status_label["text"] = "Status: Stopped"
    
    def recalibrate(self):
        if self.is_running and self.process:
            with open("recalibrate.trigger", "w") as f:
                f.write("recalibrate")
            self.status_label["text"] = "Status: Recalibrating..."

if __name__ == "__main__":
    root = tk.Tk()
    app = PostureDetectorGUI(root)
    root.mainloop() 