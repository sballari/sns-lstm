import numpy as np
from functools import reduce
from operator import itemgetter
import os

predLen =12
obsLen = 8

def lis2ind(lisfilepath,destpath):
    
    f = open(lisfilepath)
    lisDS = f.readlines()
    for r in range(len(lisDS)) : 
        lisDS[r] = lisDS[r].split('\t')
    lisDS=np.asarray(lisDS,dtype=np.float32)
    #print(lisDS)
    #print('shape listDs: ',lisDS.shape)
    lisDS = sorted(lisDS,key=itemgetter(1)) 
    lisDS=np.asarray(lisDS,dtype=np.float32)
    
    wf = open(destpath,"w")
    #print(lisDS.shape)
    for pid in range(int(lisDS[-1][1])):

        l_pid = filter(lambda x : (x[1])==pid,lisDS)
        count = 0
        listPid=[]
        for i in  l_pid:
            count-=-1
            listPid.append(i)
            if count >= predLen+obsLen : break
            
        #print (listPid)
        if (count == predLen+obsLen):
            l_pid = sorted(listPid,key=itemgetter(0)) 
            srow=""
            for i in l_pid:
                for e in i : 
                    srow = srow + ("%f " %e) 
                srow = srow + '\n'

            wf.write(srow)
    wf.close()


def explore_directories():
    paths = [ "test/", "train/", "val/"]
    for path in paths:
        for filename in os.listdir("lisotto/datasets/"+path):
            lis2ind("lisotto/datasets/"+path+filename, "datasets_lis2quan/"+path+filename )
    print("ack")



#lis2ind("../sns-lstm/datasets/train/biwi_hotel_train.txt", "../sns-lstm/datasets/train/biwi_hotel_train_indian.txt")

explore_directories()   

