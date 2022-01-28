import numpy as np
def mask_short_dur(err_seq, label, tfr=4):
 prevstate=0
 errmask = np.ones(len(err_seq))
 for t in range(len(label)):
  if prevstate!=label[t]:   
   t_left = max(0, t-tfr)
   t_right = min(len(err_seq), t+tfr+1)
   errmask[t_left:t_right] = 0
  prevstate = label[t]

 return errmask * err_seq

def comp_DER_onepair(fdoa_decision_spk, GT_spk, tfr=5):
 len_fdoa = len(fdoa_decision_spk)
 len_GT = len(GT_spk)
 if len_fdoa > len_GT:
  print( "WARNING ---------- length of fdoa={}, length of GT={}".format(len_fdoa, len_GT) )
  err_seq = fdoa_decision_spk[:len_GT]-GT_spk
  err_seq = mask_short_dur(err_seq, GT_spk, tfr=4)
 else:
  err_seq = fdoa_decision_spk-GT_spk[:len_fdoa]
  err_seq = mask_short_dur(err_seq, GT_spk[:len_fdoa], tfr=4)
 d = np.count_nonzero(err_seq)
 return d


def comp_DER_onefile(GT_dict_name_time, fdoa_decision, tfr=3):
 total_fr=GT_dict_name_time["total_fr"]
 GTspk_to_fdoaid = {}
 err_count = 0.0
 frm_count = 0.0
 #euclidean_norm = lambda a, b: np.abs(a - b)
 for each_spk in GT_dict_name_time.keys():
  # spk in GT
  if each_spk != "total_fr":
   y=np.zeros(total_fr)   
   for seg in GT_dict_name_time[each_spk]:
    st_v_fr, et_v_fr = seg
    y[st_v_fr:et_v_fr]=1   
  
   # find spk in fdoa_dict
   min_d = 999
   for spk_id in fdoa_decision.keys():
    len_fdoa = len(fdoa_decision[spk_id])  # Note: GT may be longer than face_tracking results 
    #d, cost_matrix, acc_cost_matrix, path = mydtw(fdoa_decision[spk_id], y[:len_fdoa])
    #d, path = fastdtw(y[:len_fdoa]+1, fdoa_decision[spk_id]+1) # dist=euclidean
    d = np.count_nonzero(fdoa_decision[spk_id]-y[:len_fdoa])        
    #if spk_id>2:
    # print(d, y, fdoa_decision[spk_id])
    # DTW_visual(y[:len_fdoa]+1, fdoa_decision[spk_id]+1)    
    if d < min_d:
     min_d = d
     GTspk_to_fdoaid[each_spk] = spk_id     
   print("speaker {} in the GT is the speaker id {} in face tracking, with dist={}".format(each_spk, GTspk_to_fdoaid[each_spk], min_d))

   # compute err
   # print(len(fdoa_decision[GTspk_to_fdoaid[each_spk]]), len(y))
   # Note, lengths of fdoa and GT may be different
   len_fdoa = len(fdoa_decision[GTspk_to_fdoaid[each_spk]])
   #err_count += np.count_nonzero(fdoa_decision[GTspk_to_fdoaid[each_spk]]-y[:len_fdoa])
   err_count += comp_DER_onepair(fdoa_decision[GTspk_to_fdoaid[each_spk]], y, tfr)   
   frm_count += len(y)

 DER = err_count/frm_count
 return DER, GTspk_to_fdoaid, err_count, frm_count