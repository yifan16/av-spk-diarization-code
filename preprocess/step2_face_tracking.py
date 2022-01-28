import cv2
import sys, datetime
from time import sleep

import numpy as np
import dlib
import glob
import os
import multiprocessing
import subprocess
import pdb
import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

detector=dlib.get_frontal_face_detector()

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)

pdb.set_trace = lambda: None

def draw_boxes(frame, boxes, color=(0,255,0)):
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
    return frame

def resize_image(image, size_limit=500.0):
    max_size = max(image.shape[0], image.shape[1])
    if max_size > size_limit:
        scale = size_limit / max_size
        _img = cv2.resize(image, None, fx=scale, fy=scale)
        return _img
    return image

class FaceDetector():

    def __init__(self, cascPath="./haarcascade_frontalface_default.xml"):
        self.faceCascade = cv2.CascadeClassifier(cascPath)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces

class FaceTracker():
    
    def __init__(self, frame, face):
        #pdb.set_trace()
        (x,y,w,h) = face
        self.face = (x,y,w,h)
        # Arbitrarily picked KCF tracking
        #self.tracker = cv2.TrackerKCF_create()
        self.tracker =cv2.TrackerMOSSE_create()
        #self.tracker = cv2.TrackerBOOSTING_create()
        self.tracker.init(frame, self.face)
    
    def update(self, frame):
        _, self.face = self.tracker.update(frame)
        return self.face

class Controller():
    
    def __init__(self, event_interval=20):
        self.event_interval = event_interval
        self.last_event = datetime.datetime.now()

    def trigger(self):
        """Return True if should trigger event"""
        return self.get_seconds_since() > self.event_interval
    
    def get_seconds_since(self):
        current = datetime.datetime.now()
        seconds = (current - self.last_event).seconds
        return seconds

    def reset(self):
        self.last_event = datetime.datetime.now()

class Pipeline():

    def __init__(self, event_interval=6):
        self.controller = Controller(event_interval=event_interval)    
        self.detector = FaceDetector()
        self.trackers = []
    
    def detect_and_track(self, frame):
        # get faces 
        #faces = self.detector.detect(frame) #opencv
        if frame is not None:
            #print("start to detect face")
            dets = detector(frame,0)          #dlib
            print("dets shape: ",np.array(dets).shape)
            #faces=np.zeros((1,4)) #asumes that there is only one face after initial face detection step
            if len(dets) >0:
                spk_num=len(dets)
                print(spk_num," persons detected")
                #pdb.set_trace()
                faces=np.zeros((spk_num,4))
                for j in range(spk_num):
                    faces[j,0]=dets[j].left()
                    faces[j,1]=dets[j].top()
                    faces[j,2]=dets[j].right()-dets[j].left()
                    faces[j,3]=dets[j].bottom()-dets[j].top()
                print('faces',faces)
            else:
                print("no face detected")
                return None
        else:
            print("frame is None in detect_and_track func")
            return None
        #faces = tuple((dets[0].left(),dets[0].top(),dets[0].right()-dets[0].left(),dets[0].bottom()-dets[0].top()))
        #print(np.array(faces).shape)

        ## reset timer
        #self.controller.reset()

        # get trackers
        self.trackers = [FaceTracker(frame, face) for face in faces]
        print('self.trackers[0].face',self.trackers[0].face)
        print('self.trackers[1].face',self.trackers[1].face)

        # return state = True for new boxes
        # if no faces detected, faces will be a tuple.
        #new = type(faces) is not tuple
        #if len(faces)==4:
        #    new=True
        #else:
        #    new=False
        new = True

        return faces, new
    
    def track(self, frame):
        pdb.set_trace()
        boxes = [t.update(frame) for t in self.trackers]
        # return state = False for existing boxes only
        return boxes, False
    
    def boxes_for_frame(self, frame):
        #if self.controller.trigger():
        #    return self.detect_and_track(frame)
        #else:
        #    return self.track(frame)
        return self.track(frame)



#def main():
#indir=sys.argv[1]
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='', help='input dir')
opt = parser.parse_args()

indir=opt.input_dir
mp4s=glob.glob(os.path.join(indir,"video","*.mp4"))
for mp4 in mp4s:
    print('mp4',mp4)
    event_interval=100
    if 1:
        #print(mp4)
        spk=mp4.split("/video/")[0]
        fn=mp4.split("/")[-1].split(".mp4")[0]
        outemb_fn=mp4.split("/")[-1].split(".mp4")[0]+".txt"
        print('outemb_fn',outemb_fn)

        mp4_25fps_path=spk+"/video_25fps/"+fn+".mp4"
        mp4_25fps_dir=spk+"/video_25fps"

        outemb_dir=os.path.join(spk,"tracking")
        if not (os.path.isdir(outemb_dir)):
            os.makedirs(outemb_dir)
        if not (os.path.isdir(mp4_25fps_dir)):
            os.makedirs(mp4_25fps_dir)
        outemb_path=os.path.join(outemb_dir,outemb_fn)
        print('outemb_path',outemb_path)

        if not (os.path.exists(outemb_path)) or os.path.getsize(outemb_path)<1.0e-6:
        #if 1:
            cap = cv2.VideoCapture(mp4)
            fps=cap.get(cv2.CAP_PROP_FPS)
            print("ori fps: ",fps)
            if abs(fps-25.0)>1.0e-6:
                print("changing frame rate to 25fps using ffmpeg")
                if not os.path.exists(mp4_25fps_path):
                    subprocess.call(["ffmpeg", "-i", mp4, "-r","25", "-strict", "-2", mp4_25fps_path])
            else:
                if not os.path.exists(mp4_25fps_path):
                    os.symlink(mp4,mp4_25fps_path)
            cap.release()
            video_capture=cv2.VideoCapture(mp4_25fps_path)
            frame_num=int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            print("how many frames: ",int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))

            # exit if video not opened
            if not video_capture.isOpened():
                print('Cannot open video')
                sys.exit()
            else:
                print("video is opened\n")
    
            ## read first frame
            #ok, frame = video_capture.read()
            #if not ok:
            #    print('Error reading video')
            #    #sys.exit()
            #    return None

            # init detection pipeline
            pipeline = Pipeline(event_interval=event_interval)
            
            # hot start detection
            # read some frames to get first detection
            faces = ()
            detected = False
            while not detected:
                _, frame = video_capture.read()
                #print("frame size ", np.array(frame).shape)
                if frame is not None:
                    try:
                        faces, detected = pipeline.detect_and_track(frame)
                        print("hot start; ", faces, type(faces), "size: ", np.array(faces).size)
                    except:
                        print("cannot detect face on this frame, skip to next frame")
                    #    continue
                else:
                    print("frame is None, can not detect face using dlib")
                    break
             

            #draw_boxes(frame, faces)
    
    ##
    ## main loop
    ##
            with open(outemb_path, 'w') as fp:
                pass            
            video_capture.release()
            video_capture = cv2.VideoCapture(mp4_25fps_path) #reopen
            i=0
            while (video_capture.isOpened()):
                i+=1
                #print('i',i)
                if (i>frame_num):
                    break
            # Capture frame-by-frame
                #if frame is not None:
                if True:
                    _, frame = video_capture.read()

                    # update pipeline
                    #try:
                    #
                    boxes, detected_new = pipeline.boxes_for_frame(frame)
                    #except:
                    #    continue

                # logging
                #state = "DETECTOR" if detected_new else "TRACKING"
                #print("[%s] boxes: %s" % (state, boxes))
                #print("shape",np.array(boxes).shape)
                #print("boxes",boxes)
                    with open(outemb_path, 'a+') as fp:
                        fp.write(str(boxes))
                        fp.write("\n")
                #else:
                #    break

            ## quit
            video_capture.release()
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    sys.exit()


        # When everything is done, release the capture
           # video_capture.release()
        #cv2.destroyAllWindows()
        #sys.exit()
#if __name__=="__main__":
#    main()
