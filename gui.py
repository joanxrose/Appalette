import tkinter as tk
from tkinter import filedialog
import os
import detection
import cv2 as cv

class Appalette:
    def __init__(self, root):
        self.root = root
        self.start_bg_pic = tk.PhotoImage(file = "ui_images/start_bg.png")
        self.button_pic = tk.PhotoImage(file = "ui_images/button.png")
        self.processing_bg_pic = tk.PhotoImage(file = "ui_images/processing.png")
        self.watch_pic = tk.PhotoImage(file = "ui_images/watch.png")
        self.view_pic = tk.PhotoImage(file = "ui_images/view.png")
        
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

        # Once the detection is complete, show output window
        def checkIfComplete():
            processing_window.destroy()
            self.checkOutput(file_path)

        # Call on the processVideo to start the object detection
        detection.processVideo(file_path, checkIfComplete)

    def checkOutput(self, file_path):
        self.root.withdraw()

        # Output page layout
        output_window = tk.Toplevel(self.root)
        output_window.title("Check Output")

        output_label = tk.Label(output_window, image = self.start_bg_pic)
        output_label.pack()

        # Play output video when clicked
        watch_btn = tk.Button(output_window, image = self.watch_pic, bd = 0, command = lambda: self.playVideo("output/apples_counted.mp4"))
        watch_btn.place(x = 88, y = 369)

        # Open folder of output frames
        view_btn = tk.Button(output_window, image = self.view_pic, bd = 0, command = self.viewFrames)
        view_btn.place(x = 88, y = 500)

        output_window.update() 
        output_window.protocol('WM_DELETE_WINDOW', self.exit_function)

    def playVideo(self, file_path):
        cap = cv.VideoCapture(file_path)

        # Open video player window
        cv.namedWindow("Video Player", cv.WINDOW_NORMAL)
        cv.resizeWindow("Video Player", 960, 540)

        # Use read() function to play video
        while(cap.isOpened()):
            success, frame = cap.read()
            if success:
                cv.imshow('Video Player', frame)

                # Stop if user presses 'q' on the keyboard or X on the window
                quitButton = cv.waitKey(25) & 0xFF == ord('q')
                closeButton = cv.getWindowProperty('Video Player', cv.WND_PROP_VISIBLE) < 1
                if quitButton or closeButton: 
                    break
            else:
                break

        # Release the video and destroy the window
        cap.release()
        cv.destroyAllWindows()

    # Open folder of output frames
    def viewFrames(self):
        path = "output/frames"
        path = os.path.realpath(path)
        os.startfile(path)

    def exit_function(self):
        self.root.destroy()


def gui():
    root = tk.Tk()
    root.geometry("800x700")
    root.title("Appalette")
    root.resizable(width = False, height = False)
    
    app = Appalette(root)
    root.mainloop()

if __name__ == "__main__":
    gui()
