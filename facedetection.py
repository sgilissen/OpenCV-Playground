import cv2

# Stream from the YouTubes.
stream_url = 'https://r8---sn-uxaxoxu-cg0k.googlevideo.com/videoplayback?expire=1576399989&ei=FaD1XZvHKIzi1wK4vY-QBQ&ip=84.192.104.224&id=o-ALAUG3S7aSRwLpTpv8YRRomFRPyYU9jceMJJjW08PVrg&itag=18&source=youtube&requiressl=yes&mm=31%2C29&mn=sn-uxaxoxu-cg0k%2Csn-5hnednlk&ms=au%2Crdu&mv=m&mvi=7&pcm2cms=yes&pl=14&initcwndbps=1582500&mime=video%2Fmp4&gir=yes&clen=34646795&ratebypass=yes&dur=523.006&lmt=1536790339468007&mt=1576378218&fvip=5&fexp=23842630&c=WEB&sparams=expire%2Cei%2Cip%2Cid%2Citag%2Csource%2Crequiressl%2Cmime%2Cgir%2Cclen%2Cratebypass%2Cdur%2Clmt&sig=ALgxI2wwRAIgZWc--HYKT9Lgz7nutfAMcrLGo9LjlBUp-o_-Pau1dRwCIHkSUtYrwNdmUSZxpeqPoqjH8V5l79zIl-joTto8GsG3&lsparams=mm%2Cmn%2Cms%2Cmv%2Cmvi%2Cpcm2cms%2Cpl%2Cinitcwndbps&lsig=AHylml4wRgIhAO5jcALpx1nxKiWMaafdBwa4lbiOMEX51ONXBYTpog4EAiEA5PPKdyhfQusQ5nvGMRrbRkQqnuST9DDNHWpIdYdQu7M%3D'

capt = cv2.VideoCapture(stream_url)

# Trained XML classifiers, found via the OpenCV website.
face_casc = cv2.CascadeClassifier("xmls/haarcascade_frontalface_default.xml")

while True:
    ret, frames = capt.read()

    # Convert to grayscale
    strm_gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_casc.detectMultiScale(
        strm_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print(f'Found {len(faces)} faces.')

    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display window
    cv2.imshow('Face detection', frames)

    # ESC key closes the window
    if cv2.waitKey(33) == 27:
        break

# DESTROY ALL HUMA... err. windows.
cv2.destroyAllWindows()
