import cv2

# Use 0 for default built-in camera. Try 1 or 2 if using Continuity Camera.
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Could not open video device.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        cv2.imshow('Camera Test', frame)
        
        # Press 'q' to close the window safely
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Crucial steps for macOS stability
cap.release()
cv2.destroyAllWindows()
