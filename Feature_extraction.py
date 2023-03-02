import numpy as np
from skimage.filters import gabor
import cv2
import os
import csv
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops

label = []

data_dir=os.path.expanduser(r'C:\Users\user\Desktop\sobel&hog')

files=[]
labels=[]
for r,d,f in os.walk(data_dir):
    for file in f:
        if '.jpg' in file:
            label=r.split('\\')[-1]
            labels.append(label)
            files.append(os.path.join(r,file))
with open('features_sobel&hog.csv', "a+", newline="") as wr:
    writer = csv.writer(wr)
    i=0
    for f in files:
        label=f.split('\\')[-1]
        img=cv2.imread(f)  
        img1= cv2.resize(img, (400,400))
        img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        
        # LBP
        feat_lbp = local_binary_pattern(img,5,2,'uniform')
        lbp_hist,_ = np.histogram(feat_lbp,8)
        lbp_hist = np.array(lbp_hist,dtype=float)
        lbp_prob = np.divide(lbp_hist,np.sum(lbp_hist))
        lbp_energy = np.nansum(lbp_prob**2)
        lbp_entropy = -np.nansum(np.multiply(lbp_prob,np.log2(lbp_prob)))  
        
        lbphist_features = np.reshape(np.array(lbp_hist).ravel(),(1,len(np.array(lbp_hist).ravel())))
        lbpprob_features=np.reshape(np.array(lbp_prob).ravel(),(1,len(np.array(lbp_prob).ravel())))
        lbpenrgy_features=np.reshape(np.array(lbp_energy).ravel(),(1,len(np.array(lbp_energy).ravel())))
        lbpento_features=np.reshape(np.array(lbp_entropy).ravel(),(1,len(np.array(lbp_entropy).ravel())))


        # GLCM 
        gCoMat = greycomatrix(img, [1], [0],256,symmetric=True, normed=True)
        contrast = greycoprops(gCoMat, prop='contrast')
        dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
        homogeneity = greycoprops(gCoMat, prop='homogeneity')    
        energy = greycoprops(gCoMat, prop='energy')
        correlation = greycoprops(gCoMat, prop='correlation')    
        
        contrast_features = np.reshape(np.array(contrast).ravel(),(1,len(np.array(contrast).ravel())))
        dissimilarity_features=np.reshape(np.array(dissimilarity).ravel(),(1,len(np.array(dissimilarity).ravel())))
        homogeneity_features=np.reshape(np.array(homogeneity).ravel(),(1,len(np.array(homogeneity).ravel())))
        energy_features=np.reshape(np.array(energy).ravel(),(1,len(np.array(energy).ravel())))
        correlation_features=np.reshape(np.array(correlation).ravel(),(1,len(np.array(correlation).ravel())))
    

        # Gabor filter
        gaborFilt_real,gaborFilt_imag = gabor(img,frequency=0.6)
        gaborFilt = (gaborFilt_real**2+gaborFilt_imag**2)//2
        gabor_hist,_ = np.histogram(gaborFilt,8)
        gabor_hist = np.array(gabor_hist,dtype=float)
        gabor_prob = np.divide(gabor_hist,np.sum(gabor_hist))
        gabor_energy = np.nansum(gabor_prob**2)
        gabor_entropy = -np.nansum(np.multiply(gabor_prob,np.log2(gabor_prob)))
        
        gabor_hist_features = np.reshape(np.array(gabor_hist).ravel(),(1,len(np.array(gabor_hist).ravel())))
        gabor_prob_features=np.reshape(np.array(gabor_prob).ravel(),(1,len(np.array(gabor_prob).ravel())))
        gabor_ener_features=np.reshape(np.array(gabor_energy).ravel(),(1,len(np.array(gabor_energy).ravel())))
        gabor_entr_features=np.reshape(np.array(gabor_entropy).ravel(),(1,len(np.array(gabor_entropy).ravel())))
       
        features=np.concatenate((lbpenrgy_features,lbpento_features,contrast_features,dissimilarity_features,homogeneity_features,energy_features,correlation_features,gabor_ener_features,gabor_entr_features),axis=1);
        ff=features[0].tolist()     
        writer.writerow(ff+[labels[i]])
        i+=1
    wr.close()
