import tkinter as tk
from tkinter import filedialog
import os
import detection

class Appalette:
    def __init__(self, root):
        self.root = root
        self.start_bg_pic = tk.PhotoImage(file = "ui_images/start_bg.png")
        self.button_pic = tk.PhotoImage(file = "ui_images/button.png")
        self.processing_bg_pic = tk.PhotoImage(file = "ui_images/processing.png")
        
        self.createLayout()

    # Create the layout for the Start page
    def createLayout(self):
        start_bg = tk.Label(self.root, image = self.start_bg_pic)
        start_bg.pack()

        # When button is clicked, the user can choose the input video
        start_btn = tk.Button(self.root, image = self.button_pic, bd = 0, command = self.selectVideo)
        start_btn.place(x = 88, y = 369)

    def selectVideo(self):
        file_path = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Select a video file", filetypes = [("Video files", "*.mp4")])
        
        # Start object detection after choosing input video
        if file_path:
            self.startDetection(file_path)

    def startDetection(self, file_path):
        self.root.withdraw()

        # Processing page layout
        processing_window = tk.Toplevel(self.root)
        processing_window.title("Processing...")

        processing_label = tk.Label(processing_window, image = self.processing_bg_pic)
        processing_label.pack()

        # Show the Processing page during the detection
        processing_window.update()

        # Once the detection is complete, close the tkinter windows
        def checkIfComplete():
            processing_window.destroy()
            self.root.destroy()

        # Call on the processVideo to start the object detection
        detection.processVideo(file_path, checkIfComplete)

def gui():
    root = tk.Tk()
    root.geometry("800x700")
    root.title("Appalette")
    root.resizable(width = False, height = False)
    
    app = Appalette(root)
    root.mainloop()

if __name__ == "__main__":
    gui()
