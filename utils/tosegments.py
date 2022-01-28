import pdb
import simpleder
def to_segments(input_list):
    tolerance = 0
    last = 0
    segment = 0
    output_list = []
    start = -1
    end = 0
    i = 0
    count_frame = 0
    #for i in range(len(input_list)):
    while i < len(input_list):
        #
        if start <0 and not (input_list[i] == 0):
            start = i
            #last = input_list[i]
            segment = input_list[i]
        elif start >= 0 and not input_list[i] == segment:
            
            #pdb.set_trace()
            if tolerance > 4:
                #if count_frame > 4:
                end = i - tolerance - 1
                if count_frame > (end-start) * 0.7:
                    i = start + 1
                    #count_frame = 0
                elif end - start >= 0.1:
                    output_list.append((str(segment),float(start)/25.,float(end)/25.))
                    i = i - tolerance
                count_frame = 0
                tolerance = 0 
                #last = 0 
                start = - 1
                end = 0
                segment = 0               
            else:
                tolerance += 1
        else:
            count_frame += tolerance
            tolerance = 0

        if i == len(input_list) - 1 and start > 0:
            end = i - tolerance
            output_list.append((str(segment),float(start)/25.,float(end)/25.))
        i += 1
    return output_list




'''
a = [0,3,3,3,3,3,3,4,4,3,3,3,0,0,0,4,1,1,1,4,2,3,1,4,1,0,3,3,3,3,3,3,4,4,3,3,3,0,0,0,1,1,1,1,2,3,1,4,1]
#a = [0,3,3,3,3,3,3,3,3,3,3,3,3,0,0,4,4,4,4,4,0,0,0,0,3,3,3,3,3,3,3,3,3,3,3,0,0,0,1,1,1,1,1,1,1,1,1,1]
hyp = to_segments(a)
print('pred',hyp)
b = [0,3,3,3,3,3,3,3,3,3,3,3,3,0,0,4,4,4,4,4,4,0,0,0,0,3,3,3,3,3,3,3,3,3,3,3,0,0,0,1,1,1,1,1,1,1,1,1,1]
c = [0,3,3,3,3,a,3,3,3,3,a,3,3,0,0,a,4,4,4,4,a,0,0,0,0,a,3,3,3,3,a,3,3,3,3,a,0,0,0,1,a,1,1,1,1,a,1,1,1]
ref = to_segments(b)
print('label',ref)

error = simpleder.DER(ref, hyp)
print("DER={:.3f}".format(error))
'''