import numpy as np
import matplotlib.pyplot as plt

def euc_dist(x1,y1,x2,y2):
    return ((x1-x2)**2 + (y1-y2)**2)**(0.5)


coord = np.load("coordinates/ethpkl.npy", allow_pickle = True)
print ("(examples,frame,pedestrians,coordinate)")
print('GT shape: ' + str(coord['groundTruth'].shape ))
print('pedsInSequence len: ' + str(len(coord['pedsInSequence'] )))
print('predicted shape: ' + str(coord['predicted'].shape ))

gt = coord['groundTruth']
pd = coord['predicted']
pis = coord['pedsInSequence']

#FDE
pedone = 0
esempio = 100
print("\nesempio %d, frame 20, ped %d" %(esempio,pedone))
print("Bgt (%f,%f) - Bpd (%f,%f)" %(gt[esempio,19,pedone,0], gt[esempio,19,pedone,1], pd[esempio,19,pedone,0], pd[esempio,19,pedone,1]))
print("fde: %f" %(euc_dist(gt[esempio,19,pedone,0], gt[esempio,19,pedone,1], pd[esempio,19,pedone,0], pd[esempio,19,pedone,1])))
print("distanza AZgt: %f" %euc_dist(gt[esempio,0,pedone,0], gt[esempio,0,pedone,1], gt[esempio,19,pedone,0], gt[esempio,19,pedone,1]))
print("distanza AZpred: %f" %euc_dist(pd[esempio,0,pedone,0], pd[esempio,0,pedone,1], pd[esempio,19,pedone,0], pd[esempio,19,pedone,1]))
print("Agt: (%f,%f)" %(gt[esempio,0,pedone,0], gt[esempio,0,pedone,1]))
print("Apd: (%f,%f)" %(pd[esempio,0,pedone,0], pd[esempio,0,pedone,1]))

def plot_path(es,ped):
    esempio = es
    pedone = ped
    ave,fde = ave_fde_traj(pd,gt,es,ped)
    plt.suptitle("esempio: %d, pedone %d of %d\n aveTraj: %f , fdeTraj %f" %(esempio,ped,pis[es]-1,ave,fde))
    plt.subplot(3,1,1)
    
    plt.plot(gt[esempio,:,pedone,0],gt[esempio,:,pedone,1], color='green',label='gt')
    plt.plot(pd[esempio,:,pedone,0],pd[esempio,:,pedone,1], color='red',label='pd')
    for i in range(pd.shape[1]):
        plt.text(pd[esempio,i,pedone,0],pd[esempio,i,pedone,1], str(i))

    for i in range(gt.shape[1]):
        plt.text(gt[esempio,i,pedone,0],gt[esempio,i,pedone,1], str(i))
    
    for i in range(pis[es]):
        if i != pedone : 
            plt.plot(gt[esempio,:,i,0],gt[esempio,:,i,1], color='grey')

    plt.scatter(pd[esempio,7,pedone,0],pd[esempio,7,pedone,1])
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(gt[esempio,:,pedone,0],gt[esempio,:,pedone,1], color='white')
    plt.plot(pd[esempio,:,pedone,0],pd[esempio,:,pedone,1], color='red',label='pd')
    for i in range(pd.shape[1]):
        plt.text(pd[esempio,i,pedone,0],pd[esempio,i,pedone,1], str(i))
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(pd[esempio,:,pedone,0],pd[esempio,:,pedone,1], color='white')
    plt.plot(gt[esempio,:,pedone,0],gt[esempio,:,pedone,1], color='green',label='gt')
    for i in range(gt.shape[1]):
        plt.text(gt[esempio,i,pedone,0],gt[esempio,i,pedone,1], str(i))
    plt.legend()

    plt.show()


print("\n\npedsInSequence CHECK")

def pedInExample(gt_e):
    #gt_e = [frame,ped,coord] 
    count = 0
    for p in range(gt_e.shape[1]):
        if (sum(gt_e[:,p,0]) + sum(gt_e[:,p,1])) > 0 : 
            count = count + 1
    return count

print ("peds in ex_pd %d : %d" %(esempio, pedInExample(pd[esempio,:,:,:])))
print ("peds in ex_gt %d : %d" %(esempio, pedInExample(gt[esempio,:,:,:])))
print ("peds in ex %d for pis : %d" %(esempio,pis[esempio]))

# plot_path(100,14)

#AVE
def ave(pd,gt):
    totAve = 0
    for es in range(gt.shape[0]):
        pNum = pedInExample(gt[es])
        deltaX = sum(sum( ((pd[es,8:,:,0] - gt[es,8:,:,0])**2)))
        deltaY = sum(sum( ((pd[es,8:,:,1] - gt[es,8:,:,1])**2)))
        
        ave = ((deltaX + deltaY)**(1/2) )/(pNum)
        totAve += ave
    mean_ave = totAve/(12*gt.shape[0])
    return mean_ave

print("mean ave : "+str(ave(pd,gt)))

def ave_fde_traj(pd,gt,es,ped):
    deltaXq = ((pd[es,8:,ped,0] - gt[es,8:,ped,0])**2)
    deltaYq = ((pd[es,8:,ped,1] - gt[es,8:,ped,1])**2)
    e = (deltaXq+ deltaYq)**0.5
    ave = sum(e)/12
    fde = (pd[es,19,ped,0] - gt[es,19,ped,0])**2 + (pd[es,19,ped,1] - gt[es,19,ped,1])**2 
    fde = fde**0.5
    return ave,fde


def ave_a(pd,gt):
    totAve = 0
    for es in range(gt.shape[0]):
        pNum = pedInExample(gt[es])
        deltaX = sum(sum( ((pd[es,8:,:pis[es],0] - gt[es,8:,:pis[es],0])**2)))
        deltaY = sum(sum( ((pd[es,8:,:pis[es],1] - gt[es,8:,:pis[es],1])**2)))
        
        ave = ((deltaX + deltaY)**(1/2) )/(pis[es])
        totAve += ave
    mean_ave = totAve/(12*gt.shape[0])
    return mean_ave

print("mean ave con numpedsinseq: "+str(ave_a(pd,gt)))


def final(pd,gt):
    fsum = 0
    for es in range(gt.shape[0]):
        deltaX = sum((pd[es,19,:,0] - gt[es,19,:,0])**2)
        deltaY = sum((pd[es,19,:,1] - gt[es,19,:,1])**2)
        f = (deltaX+deltaY)**(1/2)/pedInExample(gt[es])
        fsum += f
    return fsum/(gt.shape[0])

def final_a(pd,gt):
    fsum = 0
    for es in range(gt.shape[0]):
        deltaX = sum((pd[es,19,:pis[es],0] - gt[es,19,:pis[es],0])**2)
        deltaY = sum((pd[es,19,:pis[es],1] - gt[es,19,:pis[es],1])**2)
        f = ((deltaX+deltaY)**(1/2))/pis[es]
        fsum += f
    return fsum/(gt.shape[0])

print("final : "+str(final(pd,gt)))
print("final pis: "+str(final_a(pd,gt)))


def max_diff(es,ped):
    deltaX = (pd[es,:,ped,0] - gt[es,:,ped,0])**2
    deltaY = (pd[es,:,ped,1] - gt[es,:,ped,1])**2
    tmp = (deltaX + deltaY)# print (tmp )**(0.5)
    return tmp.max()

def nzError(es):
    count =0
    for ped in range(gt.shape[2]):
        if max_diff(es,ped)!=0 : count+=1
    return count


print ("\n\n\n")
es = 188
ess = [10,22,66,188,189,250]
ped =2
mx = max_diff(es,ped)
print("max dist ped %d es %d : %f" %(ped,es,mx))
print("ped not perfect in es %d : %d" %(es,nzError(es)))
print ("pis in %d : %d" %(es,pis[es]))



for es in ess :
    for i in range(pis[es]) : 
        plot_path(es,i)

