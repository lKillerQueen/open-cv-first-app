from unicodedata import mirrored
import cv2


def face_capture():
    puth = './haarcascade_frontalface_default.xml'
    clf = cv2.CascadeClassifier(puth)
    camera = cv2.VideoCapture(1)

    while True:
        _, frame = camera.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = clf.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(30, 30),
            flags=(cv2.CASCADE_SCALE_IMAGE)
        )

        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x, y), (x + width,
                          y + height), (130, 0, 0), 2)
        mirror_frame = cv2.flip(frame, 1)
        cv2.imshow('Faces', mirror_frame)

        if cv2.waitKey(1) == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()


face_capture()
