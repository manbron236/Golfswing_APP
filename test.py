import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

class VideoApp:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.video_running = False  # 비디오 상태를 추적하는 플래그

        # Open video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Create a button and place it on the canvas
        self.btn_toggle = ttk.Button(window, text="Start", command=self.toggle_video)
        self.btn_toggle.place(x=10, y=10)

        self.btn_capture = ttk.Button(window, text="Capture", command=self.capture_image)
        self.btn_capture.place(x=10, y=40)

        self.delay = 15
        self.update()

        self.window.mainloop()

    def toggle_video(self):
        if self.video_running:
            self.stop_video()
        else:
            self.start_video()

    def start_video(self):
        self.video_running = True
        self.btn_toggle.config(text="Stop")

    def stop_video(self):
        self.video_running = False
        self.btn_toggle.config(text="Start")

    def capture_image(self):
        if self.video_running:
            ret, frame = self.vid.read()
            if ret:
                cv2.imwrite("captured_image.jpg", frame)
                print("Image captured and saved!")

    def update(self):
        if self.video_running:
            ret, frame = self.vid.read()
            if ret:
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the VideoApp object
VideoApp(tk.Tk(), "Tkinter and OpenCV")
