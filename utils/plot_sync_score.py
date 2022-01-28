import numpy as np
import argparse
import pickle
import ntpath
import matplotlib.pyplot as plt
from scipy import signal
#plt.switch_backend('agg')


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_one_file(npy_file, a1, a2):
	w=0.1
	b=-3.0
	a=np.load(npy_file)
	print ('a',a.shape)
	a = a[[2,0,1],...]
	
	P=1/(1+np.exp(w*a+b))
	for each_face in range(a.shape[0]):
		y=a[each_face, :]

		y = smooth(y,39)
		
		yp=P[each_face, :]
		x=np.arange(len(y))
		a1.plot(x, y)
		a2.plot(x, yp)
	#plt.show()

def set_color_cycle(self, clist=None):
    if clist is None:
        clist = rcParams['axes.color_cycle']
    self.color_cycle = itertools.cycle(clist)

def plot_GT(st_et_list, a3):
	AUDIO_VIDEO_FRAME_RATIO = 40
	nspk=len(st_et_list)
	T=st_et_list[-1][-1] # NOTE: we assume the file is ended with speecch NOT sil/noise
	T_frame=int(T*1000/AUDIO_VIDEO_FRAME_RATIO)
	for each_spk in range(nspk):
		st, et = st_et_list[each_spk]
		st_frame = int(st*1000/AUDIO_VIDEO_FRAME_RATIO)
		et_frame = int(et*1000/AUDIO_VIDEO_FRAME_RATIO)		
		y=np.zeros(T_frame)
		y[st_frame:et_frame]=1
		x=np.arange(len(y))
		
		#ax = plt.plot(x, y)
		a3.fill_between(x, 0, y)

	#plt.show()

def plot_GT_fromdict_with_mult_seg(dict_name_time, total_fr, a3):
	
	for each_spk in dict_name_time.keys():
		y=np.zeros(total_fr)
		for seg in dict_name_time[each_spk]:
			st_v_fr, et_v_fr = seg
			y[st_v_fr:et_v_fr]=1
		x=np.arange(len(y))
		a3.fill_between(x, 0, y)


def convert_time_format_to_video_frame(timestr):
	AUDIO_VIDEO_FRAME_RATIO = 40
	if len(timestr.split(':'))==2:
		mstr, secstr = timestr.split(':')
		mm = float(mstr)
		sec = float(secstr)
		v_fr= int((mm*60+sec)*1000/40)
	
	if len(timestr.split(':'))==3:
		hstr, mstr, secstr = timestr.split(':')
		hh = float(hstr)
		mm = float(mstr)
		sec = float(secstr)
		v_fr= int((hh*3600+mm*60+sec)*1000/40)
	return v_fr


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
	return dict_name_time, total_fr

def plot_simple(input_file):
	f, (a1, a2, a3) = plt.subplots(3,1, gridspec_kw={'height_ratios': [3, 3, 1]})
	a1.set_xlim([000,3000])
	a2.set_xlim([000,3000])
	a3.set_xlim([000,3000])

	plot_one_file(input_file, a1, a2)
	plt.show()
	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("input_file", help="input a npy file (numpy array)")
	parser.add_argument("-d", "--info_dict", help="OPTIONAL: information dict")
	parser.add_argument("-t", "--GT_txt", help="OPTIONAL: GT time segment")
	args = parser.parse_args()	

	#ax1 = plt.subplot(2, 1, 1)
	f, (a1, a2, a3) = plt.subplots(3,1, gridspec_kw={'height_ratios': [3, 3, 1]})
	a1.set_xlim([000,3000])
	a2.set_xlim([000,3000])
	a3.set_xlim([000,3000])

	plot_one_file(args.input_file, a1, a2)

	#plt.show()
	if args.GT_txt:
		dict_name_time, total_fr = read_dia_GT(args.GT_txt)
		plot_GT_fromdict_with_mult_seg(dict_name_time, total_fr, a3)
	
	if args.info_dict: 
		file_dict = pickle.load(open(args.info_dict, 'rb'))
		#print(ntpath.basename(args.input_file))
		utt_id = ntpath.basename(args.input_file).split('_')[0]
		st_et_list = file_dict[utt_id]["time_idx"]
		#plt.hold(True)
		#ax2 = plt.subplot(2, 1, 2)
		plot_GT(st_et_list, a3)
		#ax1.get_shared_x_axes().join(ax1, ax2)
		#ax1.set_xticklabels([])

	plt.show()
	plt.savefig(args.input_file+'.png')

	
	
