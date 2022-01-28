import pdb
import datetime
import numpy as np

def convert_time_format_to_video_frame(str_date):
    
    d_date = datetime.datetime.strptime(str_date.replace('\n','') , '%M:%S.%f')
    #pdb.set_trace()
    frame = int(round((d_date.timetuple().tm_min*60+d_date.timetuple().tm_sec + d_date.microsecond/1000000. )*25))
    return frame

def convert_time_format_to_video_frame_vad(str_date):
    
    frame = int(round(float(str_date)*25))
    return frame

def read_dia_vad(txt_file,num_frames): 
 vad = np.zeros(num_frames)
 #pad.set_trace()
 for line in open(txt_file):
  # ignore empty line and comment line 
   time1, time2 = line.split()
   st_v_fr = convert_time_format_to_video_frame_vad(time1)
   et_v_fr = convert_time_format_to_video_frame_vad(time2)
   vad[st_v_fr:et_v_fr] = 1
 return vad

def read_dia_GT(txt_file): 
 from collections import defaultdict
 dict_name_time = defaultdict(list)

 for line in open(txt_file):
  # ignore empty line and comment line
  if line.startswith("total_time"):
   tmp, total_time = line.split('=')
   total_fr= convert_time_format_to_video_frame(total_time)
  elif ( line.strip() and (not line.startswith("#")) ):   
   # remove the comments after #
   if '#' in line:
    line, tmp=line.split('#')   
   time, name = line.split()
   st, et = time.split('__')
   st_v_fr = convert_time_format_to_video_frame(st)
   et_v_fr = convert_time_format_to_video_frame(et)
   segment = [st_v_fr, et_v_fr]
   dict_name_time[name].append(segment)

 print( 'total number of speakers: ', len(dict_name_time.keys()) )
 print( 'total number of frames: ', total_fr )
 return dict_name_time, total_fr


def read_dia_GT_all(txt_file): 
 from collections import defaultdict
 dict_name_time = defaultdict(list)

 for line in open(txt_file):
  # ignore empty line and comment line
 
  if ( line.strip() and (not line.startswith("#")) ):   
   # remove the comments after #
   if '#' in line:
    line, tmp=line.split('#')   
   time, name = line.split()
   st, et = time.split('__')
   st_v_fr = convert_time_format_to_video_frame(st)
   et_v_fr = convert_time_format_to_video_frame(et)
   segment = [st_v_fr, et_v_fr]
   dict_name_time[name].append(segment)

 print( 'total number of speakers: ', len(dict_name_time.keys()) )
 #print( 'total number of frames: ', total_fr )
 return dict_name_time 

def get_ours_label(txt_file,people_dic = {'austin':2,'raymond':1,'yong':3}):
  #pdb.set_trace()
  dic, num_frames = read_dia_GT(txt_file)
  label = np.zeros(num_frames)
  
  for people in people_dic.keys():
    people_time = dic[people]
    for time_seg in people_time:
      time1,time2 = time_seg
      label[time1:time2] = people_dic[people]
  return label.astype(int)

#pdb.set_trace()
#aaa = get_ours_label('../data/ours_gt.txt')

def get_ours_label1(txt_file,txt_file_vad,people_dic = {'austin':2,'raymond':1,'yong':3}):
  #pdb.set_trace()
  dic, num_frames = read_dia_GT(txt_file)
  label = np.zeros(num_frames)
  vad = read_dia_vad(txt_file_vad,num_frames)
  
  for people in people_dic.keys():
    people_time = dic[people]
    for time_seg in people_time:
      time1,time2 = time_seg
      label[time1:time2] = people_dic[people]
  return label.astype(int),vad


def get_ours_label_all(txt_file,num_frames,people_dic = {'person0':1,'person1':2,'person2':3,'person3':4}):
  #pdb.set_trace()
  dic = read_dia_GT_all(txt_file)
  label = np.zeros(num_frames)
  #vad = read_dia_vad(txt_file_vad,num_frames)
  
  for people in people_dic.keys():
    people_time = dic[people]
    for time_seg in people_time:
      time1,time2 = time_seg
      label[time1:time2] = people_dic[people]
  return label.astype(int)


 