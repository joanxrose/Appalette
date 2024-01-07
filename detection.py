from imageai.Detection import VideoObjectDetection
from PIL import Image, ImageDraw, ImageFont
import cv2 as cv
import numpy as np
import os

# This function draws lines on the detected apples for texture.
def drawLine(image, start_point, end_point, color):
    if isinstance(image, np.ndarray):
        # Convert NumPy array to Image
        image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(image)

    dot_length = 8
    gap_length = 8
    if start_point[0] == end_point[0]:  # Vertical line
        for y in range(start_point[1], end_point[1], dot_length + gap_length):
            y_end = min(y + dot_length, end_point[1])
            draw.line([(start_point[0], y), (start_point[0], y_end)], fill=color, width=2)
    elif start_point[1] == end_point[1]:  # Horizontal line
        for x in range(start_point[0], end_point[0], dot_length + gap_length):
            x_end = min(x + dot_length, end_point[0])
            draw.line([(x, start_point[1]), (x_end, start_point[1])], fill=color, width=2)

    # Save the modified image back to the NumPy array.
    image_np = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

    return image_np


# This function returns the dominant hue of the region of interest (ROI) in an image.
# It accepts the boxpoints of the ROI and a NumPy array representing the image.
def getDominantHue(box_points, image_np):
    # Extract ROI using box points.
    x_min, y_min, x_max, y_max = box_points
    roi = image_np[y_min:y_max, x_min:x_max]

    # Convert ROI to HSV color space.
    roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    # Calculate histogram of hue values.
    hist_hue = cv.calcHist([roi_hsv], [0], None, [180], [0, 180])

    # Find the dominant hue.
    dominant_hue_bin = np.argmax(hist_hue)
    dominant_hue = int((dominant_hue_bin + 0.5) * 2)

    # Convert the NumPy array to an image for visualization (Beware!).
    # opencv_image = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)
    # cv.imshow("Image", opencv_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return dominant_hue


# This is the function executed after each frame of the video is detected. 
def forFrame(frame_number, output_array, output_count, returned_frame):
    green_apples = 0
    red_apples = 0
    color = {"black": (0, 0, 0), "white": (255, 255, 255)}

    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)

    # Get the set of boxpoints of every apple object.
    apples_box_points = [obj["box_points"] for obj in output_array if obj["name"] == "apple"]
    # print(F"Box Points: {apples_box_points}")

    # For every set of apple boxpoint, get dominant hue inside the box
    for box_points in apples_box_points:
        # Call getDominantHue function.
        dominant_hue = getDominantHue(box_points, returned_frame)
        print(F"Dominant Hue: {dominant_hue}")

        # Identify the type of apple based on dominant hue.
        if dominant_hue in range(30, 90): # TODO: Test hue range.
            green_apples += 1
            label = "Green Apple"
            line_style = "vertical"
        else:
            red_apples += 1
            label = "Red Apple"
            line_style = "horizontal"
    
        # Draw lines on the apples for texture (Optional).
        x_min, y_min, x_max, y_max = box_points
        if line_style == "horizontal":
            start_point = (x_min, (y_min + y_max) // 2)
            end_point = (x_max, (y_min + y_max) // 2)
        else:
            start_point = ((x_min + x_max) // 2, y_min)
            end_point = ((x_min + x_max) // 2, y_max)

        returned_frame = drawLine(returned_frame, start_point, end_point, color["black"])

        # Add text label.
        text_size, _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_w, text_h = text_size
        cv.rectangle(returned_frame, (x_min, y_max - text_h - 8), (x_min + text_w + 8, y_max), color["black"], -1)
        cv.putText(returned_frame, label, (x_min + 6, y_max - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, color["white"], 2, cv.LINE_AA)

    print(F"Red: {red_apples}\nGreen: {green_apples}")

    # Write number of red and green apples on frame.
    text_size_red, _ = cv.getTextSize(F"Red Apples: {red_apples}", cv.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_size_green, _ = cv.getTextSize(F"Green Apples: {green_apples}", cv.FONT_HERSHEY_SIMPLEX, 1, 2)

    text_w_red, text_h_red = text_size_red
    text_w_green, text_h_green = text_size_green
    margin = 8 

    cv.rectangle(returned_frame, (10 - margin, 50 - margin), (10 + text_w_red + 2 * margin, 50 + text_h_red + 2 * margin), color["black"], -1)
    cv.rectangle(returned_frame, (10 - margin, 100 - margin), (10 + text_w_green + 2 * margin, 100 + text_h_green + 2 * margin), color["black"], -1)

    cv.putText(returned_frame, F"Red Apples: {red_apples}", (14, 50 + text_h_red), cv.FONT_HERSHEY_SIMPLEX, 1, color["white"], 2, cv.LINE_AA)
    cv.putText(returned_frame, F"Green Apples: {green_apples}", (14, 100 + text_h_green), cv.FONT_HERSHEY_SIMPLEX, 1, color["white"], 2, cv.LINE_AA)

    # Save frame in a folder.
    output_folder = 'output/frames/'
    os.makedirs(output_folder, exist_ok=True)
    cv.imwrite(F"output/frames/output_frame_{frame_number}.jpg", returned_frame)

    print("------------END OF A FRAME --------------")


def processVideo(file_path, checkIfComplete):
    # Get execution path.
    execution_path = os.getcwd()

    def detectionThread():
        # Detect objects in video using ImageAI.
        video_detector = VideoObjectDetection()
        video_detector.setModelTypeAsYOLOv3()
        video_detector.setModelPath(os.path.join(execution_path, "yolov3.pt"))
        video_detector.loadModel()
        video_detector.detectObjectsFromVideo(
                    input_file_path=os.path.join(execution_path, file_path),
                    output_file_path=os.path.join(execution_path, "output/apples_detected") ,
                    frames_per_second=20,
                    per_frame_function=forFrame,
                    minimum_percentage_probability=30,
                    return_detected_frame=True
                )

        # Use OpenCV VideoWriter class to combine the saved frames.
        # Get the list of saved frames
        frames_folder = 'output/frames/'
        frames_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])

        # Specify the output video file.
        output_video_path = 'output/apples_counted.mp4'

        # Get the first frame to obtain video dimensions.
        first_frame = cv.imread(os.path.join(frames_folder, frames_files[0]))
        height, width, layers = first_frame.shape

        # Initialize VideoWriter object.
        video_writer = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

        # Iterate through frames and write to video.
        for frame_file in frames_files:
            frame_path = os.path.join(frames_folder, frame_file)
            frame = cv.imread(frame_path)
            video_writer.write(frame)

        # Release the VideoWriter object.
        video_writer.release()

        print(f"Video saved at: {output_video_path}")

        # Once the detection is done, update GUI
        checkIfComplete()

    detectionThread()
