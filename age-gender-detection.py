import cv2
from deepface import DeepFace
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Open a connection to the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

haar_cascade_path = r'C:/Users/lSS/Downloads/Compressed/FairFace-master/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

if face_cascade.empty():
    print(f"Error: Failed to load Haar cascade from path: {haar_cascade_path}")
    exit()

# Define age ranges
age_ranges = [
    (0, 5), (6, 10), (11, 15), (16, 20), 
    (21, 25), (26, 30), (31, 35), (36, 40), 
    (41, 50), (51, 60), (61, 120) 
]

def get_age_range(age):
    """Get the age range for the given age."""
    for start, end in age_ranges:
        if start <= age <= end:
            return f"{start}-{end}"
    return "Unknown"  # Default case if age doesn't fit any range

# Initialize tracking variables
tracked_faces = set()  # Set to store tracked face identifiers
male_count = 0
female_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame using Haar cascade
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces are detected
    if len(faces) == 0:
        print("No face detected in the frame.")
        # Display message on the frame
        cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        # Loop through detected faces
        for (x, y, w, h) in faces:
            # Create a unique identifier for the face
            face_id = f"{x}-{y}-{w}-{h}"

            if face_id not in tracked_faces:
                # If the face is not already tracked, analyze it
                face = frame[y:y+h, x:x+w]
                
                try:
                    analysis = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False)
                    
                    # Check if the result contains a face analysis
                    if isinstance(analysis, list):
                        if len(analysis) > 0:
                            analysis = analysis[0]  # Get the first result if a list of results is returned
                    
                    # Ensure 'age' and 'gender' are in the analysis result
                    if 'age' in analysis and 'gender' in analysis:
                        age = analysis['age']
                        gender_confidences = analysis['gender']
                        
                        # Get the gender with the highest confidence
                        gender = max(gender_confidences, key=gender_confidences.get)
                        
                        # Increment the appropriate gender counter
                        if gender == 'Man':
                            male_count += 1
                        elif gender == 'Woman':
                            female_count += 1
                        
                        # Determine the age range
                        age_range = get_age_range(age)
                        
                        # Print the detected age range and gender
                        print(f"Detected Age Range: {age_range}, Detected Gender: {gender}")
                        print(f"Male Count: {male_count}, Female Count: {female_count}")
                        
                        # Store the face in the tracked set to avoid re-detection
                        tracked_faces.add(face_id)
                        
                        # Display the results on the frame without accuracy details
                        label = f"Age Range: {age_range}, Gender: {gender}"
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
                
                except Exception as e:
                    print(f"Analysis error: {e}")

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the counts on the frame
    count_label = f"Males: {male_count}, Females: {female_count}"
    cv2.putText(frame, count_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
