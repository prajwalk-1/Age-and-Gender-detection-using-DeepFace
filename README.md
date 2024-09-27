# Age and Gender Detection using DeepFace

This project demonstrates how to use the DeepFace library along with OpenCV to detect faces from a live webcam feed, determine their gender and age range, and keep track of the counts of detected males and females.

## Features:

- Real-time face detection using Haar Cascades.
- Gender and age range prediction using the DeepFace library.
- Tracking of unique faces to avoid duplicate counting.
- Display of live video feed with annotated age and gender.
- Console output of detected age ranges and gender counts.

## Requirements:

- Python 3.10
- OpenCV
- DeepFace
- TensorFlow 
- Haar Cascade XML file for face detection

## Installation:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/prajwalk-1/age-gender-detection.git
   cd age-gender-detection
   ```

2. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Haar Cascade File:**
   Download the `haarcascade_frontalface_default.xml` file from the [OpenCV GitHub repository](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) and place it in the project directory.

## Usage

1. **Run the Script:**
   ```bash
   python age_gender_detection.py
   ```

2. **Quit the Application:**
   Press the `q` key to exit the video feed.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [DeepFace Library](https://github.com/serengil/deepface)
- [OpenCV](https://opencv.org/)
- [Haar Cascade](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)

---
