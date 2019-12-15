import cv2

# Stream from the YouTubes.
stream_url = 'https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1576399363/ei/o531XZ1y0pOBB7b1jYAK/ip/84.192.104.224/id/3kM_fBVHRCg.1/itag/301/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/goi/160/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D299/hls_chunk_host/r4---sn-uxaxoxu-cg0k.googlevideo.com/playlist_type/DVR/initcwndbps/11510/mm/44/mn/sn-uxaxoxu-cg0k/ms/lva/mv/m/mvi/3/pcm2cms/yes/pl/14/dover/11/keepalive/yes/fexp/23842630/mt/1576377684/disable_polymer/true/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,goi,sgoap,sgovp,playlist_type/sig/ALgxI2wwRAIgb8cVUQom09gsGgmc0EIenP8luEMqZEF5PmoXxIj0c6UCIANjm-pgvbHsUYfqo-ZWV3epheiswcTu5OmYkjevSrsH/lsparams/hls_chunk_host,initcwndbps,mm,mn,ms,mv,mvi,pcm2cms,pl/lsig/AHylml4wRQIhALiF1z3U4JUr5HFCCxHEWrF4a-PGqgrEa6wi4qOZ5vBrAiB0ArpRDL66wTG3zCmYQF40gxMdbgILMO4KCNjygkh6bA%3D%3D/playlist/index.m3u8'

capt = cv2.VideoCapture(stream_url)

# Trained XML classifiers, found on GitHub. Might try different ones later.
car_casc = cv2.CascadeClassifier("xmls/cars.xml")

while True:
    # Read frames
    ret, frames = capt.read()

    # Convert to grayscale
    strm_gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # Detect different sizes of vehicles in the stream image
    vehicles = car_casc.detectMultiScale(strm_gray, 1.1, 1)

    # Draw a rect around each vehicle
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 0, 255), 2)

    for vehicle in vehicles:
        print(f'Found {vehicle}')

    # Display window
    cv2.imshow('Car detection', frames)

    # ESC key closes the window
    if cv2.waitKey(33) == 27:
        break

# DESTROY ALL HUMA... err. windows.
cv2.destroyAllWindows()