import cv2

#This is our face classifier
#Tells us if the image contains a face or not

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#the face detector is based on the VIOLA-JONES Algorithm
#error might pop up saying cv2 has no cascadeclassifier member
#so we go to settings.json and add "python.linting.pylintArgs": ["--generate-members"]

#we need to grab webcam feed / webcam stream / footage now
webcam = cv2.VideoCapture(0)
#when our input is 0 = webcam, but it can be any.mp4 video

#grab current frame
while True:

    #this READS the current frame from the video
    successful_frame_read, frame = webcam.read()

    #abort if error // videofails break
    if not successful_frame_read:
        break

    #change to grayscale to optimize face detection
    #CVT will convert color and we will do rgb to gray
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces now
    #multiscale = multiple sizes of faces
    faces = face_detector.detectMultiScale(frame_grayscale)

    #run smile detection within each of those faces
    for (x, y, w, h) in faces:

        #draw a rectangle within each of those faces
        #we use FRAME not FRAME_GRAYSCALE
        #we only needed the frame_grayscale to get the caluclations
        #so we use the colored image
        cv2.rectangle(frame, (x, y),(x+w, y+h), (100, 200, 50), 4)

    #test if its working
    #will show coordinates of faces
    #and list will increase when finding more faces
    #print(faces)
    #test passed

    #this shows the current frame
    #cv2.imshow('SMILE AI', frame_grayscale)
    cv2.imshow('SMILE AI', frame)

    #display
    cv2.waitKey(1)


#CleanUp
#free up resources and close all windows
webcam.release()
cv2.destroyAllWindows()


#code ran with no errorz
print("Code done")