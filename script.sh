python preprocess/step1_change_fps.py --input_dir=/media/yifan/data/Dropbox/Dropbox/01_projects/synchronize/code/av-spk-diarization-code/data/test_videos
python preprocess/step2_face_tracking.py --input_dir=/media/yifan/data/Dropbox/Dropbox/01_projects/synchronize/code/av-spk-diarization-code/data/test_videos
python preprocess/step3_get_audio.py --input_dir=/media/yifan/data/Dropbox/Dropbox/01_projects/synchronize/code/av-spk-diarization-code/data/test_videos
python preprocess/step4_get_crop_face_npy_from_tracking.py --input_dir=/media/yifan/data/Dropbox/Dropbox/01_projects/synchronize/code/av-spk-diarization-code/data/test_videos
python preprocess/step5_get_audio_feat.py --input_dir=/media/yifan/data/Dropbox/Dropbox/01_projects/synchronize/code/av-spk-diarization-code/data/test_videos


