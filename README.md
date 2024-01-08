# Appalette (CMSC 165 Lecture Project)
An application that counts the number of green and red apples in a video. 

## Contributors
1. Domingo, Reamonn Lois A.
2. Duño, Zyra Jelic B.
3. Paelden, Joan Rose C.

## Installation
Follow these steps to install and run Appalette:
1. [Download](https://www.python.org/downloads/) and install Python 3.7, Python 3.8, Python 3.9 or Python 3.10. As of January 2024, ImageAI is compatible with these Python versions only.
2. [Download](https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/yolov3.pt/) the pre-trained Yolov3 model.
3. Create and activate your virtual environment.
    - On Windows:
            ```bash
            python -m venv venv 
            venv\Scripts\activate
            ```
    - On macOS and Linux:
        ```bash
        python -m venv venv
        source venv/bin/activate
        ```        
4. Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```
5. Make sure you have tkinter installed. It is not distributed through pip, so you have to get it elsewhere. However, it comes pre-packaged with Python for Windows and macOS.
6. Run Appalette.
    ```
    python gui.py
    ```

