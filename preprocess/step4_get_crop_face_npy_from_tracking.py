import os
import sys
import glob
import cv2
import numpy as np
import subprocess
from scipy.misc import imsave
import argparse


#indir=sys.argv[1]
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='', help='input dir')
opt = parser.parse_args()

indir=opt.input_dir
tracks=glob.glob(os.path.join(indir,"tracking","*.txt"))
track_failed_dir=os.path.join(indir,"tracking","failed")
if not os.path.exists(track_failed_dir):
    os.makedirs(track_failed_dir)
for track in tracks:
    
    fn=track.split("tracking/")[-1].split(".txt")[0]
    mp4=track.split("tracking/")[0]+"video/"+fn+".mp4"
    mp4_25fps_path=track.split("tracking/")[0]+"video_25fps/"+fn+".mp4"
    mp4_25fps_dir=track.split("tracking/")[0]+"video_25fps"
    crop_face_112_dir=track.split("tracking/")[0]+"crop_face_112"
    print(crop_face_112_dir)
    crop_face_112_path=os.path.join(crop_face_112_dir,fn+"_person"+str(0)+".npy")   
    if os.path.exists(crop_face_112_path):
        continue
    #sys.exit()
    if not os.path.exists(mp4_25fps_dir):
        os.makedirs(mp4_25fps_dir)
    if not os.path.exists(crop_face_112_dir):
        os.makedirs(crop_face_112_dir)
    #get tracking bounding box for each frame: [(966.0, 471.0, 180.0, 180.0), (1566.0, 445.0, 216.0, 216.0), (279.0, 422.0, 216.0, 216.0)]
    with open(track,"r") as fd:
        data=fd.readlines()
    all_bounding_box=[]
    for d in data:
        d=d.replace("[","")
        d=d.replace("(","")
        d=d.replace("]","")
        d=d.replace(")","")
        d=d.replace(" ","")
        d=d.replace("\n","")
        bounding_box=d.split(",")
        all_bounding_box.append(bounding_box)
    cap=cv2.VideoCapture(mp4)
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
    cap=cv2.VideoCapture(mp4_25fps_path)
    video_frame_num=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not(video_frame_num == len(all_bounding_box)):
        print("Error: video frame num is not equal to tracking frame num ", video_frame_num, len(all_bounding_box))
        sys.exit()
    i=0
    cur_bounding_box=all_bounding_box[i]
    spk_num=len(cur_bounding_box)/4
    print(np.int(spk_num)," persons exists")
    all_mouths=np.zeros(shape=(np.int(spk_num),112,112,np.int(video_frame_num)),dtype=np.float32)
    old_mouth = np.zeros((5,112,112))
    while(cap.isOpened()):
        
        if i==np.int(video_frame_num):
            break
        ret,frame=cap.read()
        cur_bounding_box=all_bounding_box[i]
        #spk_num=len(cur_bounding_box)/4
        #print("frame ",i)
        #print(np.int(spk_num)," persons exists")
        #all_mouths=np.zeros(shape=(np.int(spk_num),112,112,np.int(video_frame_num)),dtype=np.float32)
        for j in range(np.int(spk_num)):
            
            
            x=np.int(np.float(cur_bounding_box[j*4+0]))
            y=np.int(np.float(cur_bounding_box[j*4+1]))
            w=np.int(np.float(cur_bounding_box[j*4+2]))
            h=np.int(np.float(cur_bounding_box[j*4+3]))
            face=frame[y:y+h,x:x+w] # using imsave, i am sure now it is BGR
            #print(fn)
            #print(face)
            if np.array(face).shape[0]>1 and np.array(face).shape[1]>1:
                #print(np.array(face).shape)
                face=cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # now it is RGB
                half_face=face[np.int(len(face)/2):,:]
                mouth=cv2.cvtColor(half_face, cv2.COLOR_RGB2GRAY)
                mouth=cv2.resize(mouth,(112,112))
                old_mouth[j,:,:] = mouth
                all_mouths[j,:,:,i]=mouth
            elif i == 0:
                print("warning: no face tracked for this person",j," at this frame", i, " of file ",fn)
                failed_track_txt=os.path.join(track_failed_dir,fn+".txt")
                subprocess.call(["mv",track,failed_track_txt])
                #sys.exit()
            else:
                all_mouths[j,:,:,i]=old_mouth[j,:,:]

                #sys.exit()
            #print(all_mouths[j,:,:,i])
            #print(mouth)
            #imsave("mouth.jpg",face)
            #cv2.imshow("mouth",mouth)
            #cv2.waitKey(0)
            #sys.exit()
        i+=1 #frame index increased
        #if i>10:
        #    break
    cap.release()
    #save into npy
    all_mouths=np.array(all_mouths,dtype="uint8")
    for j in range(np.int(spk_num)):
        crop_face_112_path=os.path.join(crop_face_112_dir,fn+"_person"+str(j)+".npy")   
        print(crop_face_112_path)
        with open(crop_face_112_path, 'wb') as fp:
            cur_person_mouths=all_mouths[j,:,:,:]
            #print(cur_mouth)
            #print(cur_mouth.shape)
            #cv2.imshow("mouth",cur_mouth[:,:,9])
            #cv2.waitKey(0)
            #sys.exit()
            np.save(fp, cur_person_mouths)
        

