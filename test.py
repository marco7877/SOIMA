#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:53:02 2020

@author: marco
"""

def main():
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    def most_common(lst, n):# sirve para plotear, busca más cumun
        if len(lst) == 0: return -1
    #print(lst)
        counts = np.zeros(shape=n+1, dtype=np.int)
        for i in range(len(lst)):
            winner=int(lst[i])
            counts[winner]+=1
        return np.argmax(counts)
    
    def dist(v1, v2):
        v1=np.array(v1)
        v2=np.array(v2)
        return math.sqrt(sum((v1 - v2) ** 2))
    def closest_node(data, t, map, m_rows, m_cols):
        result=(0,0)
        small_dist = dist(data[t],map[0][0])#número inicial de distancia pequeña
        for i in range(int(m_rows)):
            for j in range(int(m_cols)):
                ed = dist(map[i][j], data[t])#esto saca la distancia euclideana entre
# el elemento t y el nodo por analizar i,j#####################################
                if ed < small_dist:# compra distancia euclideana para saber su fitness
                    small_dist = ed# si sí, la menor distancia se reescribe
                    result = (i, j)# si sí, resultado se sobrescriba con el nodo + fit
        return result
    def winner_modalnode(stimuli, SOM, side):
        result=(0,0)
        small_dist = dist(stimuli,SOM[0][0])#número inicial de distancia pequeña
        for i in range(int(side)):
            for j in range(int(side)):
                ed = dist(SOM[i][j], stimuli)#esto saca la distancia euclideana entre
# el elemento t y el nodo por analizar i,j#####################################
                if ed < small_dist:# compra distancia euclideana para saber su fitness
                    small_dist = ed# si sí, la menor distancia se reescribe
                    result = (i, j)# si sí, resultado se sobrescriba con el nodo + fit
        return result
    def charge_som(som_file):
        weights=np.loadtxt(som_file,dtype=float,delimiter=",")
        area,dim=weights.shape
        side=np.sqrt(area)
        som=np.empty(shape=(int(side),int(side),int(dim)),dtype=float)
        index=0
        for i in range(int(side)):
            for j in range(int(side)):
                som[i][j]=np.array(weights[index])
                index+=1
        return som,side
    def charge_hebbian(hebbian_file,sidesom,sidemmr):
        hebbianMatrix=np.loadtxt(hebbian_file,dtype=float,delimiter=",")
        from itertools import product
        keys1=list(product((list(range(int(sidesom)))),repeat=2))
        keys2=list(product((list(range(int(sidemmr)))),repeat=2))
        dictRows={ keys1[i] : i for i in range(0, len(keys1) ) }
        dictCols={ keys2[i] : i for i in range(0, len(keys2) ) }
        return hebbianMatrix,dictRows,dictCols
    def charge_files(filename,labels=True):
        file=np.loadtxt(filename,dtype=float,delimiter=",")
        R,C=file.shape
        if labels==True:
            data=np.loadtxt(filename,dtype=float,delimiter=",", usecols=range(0,(C-1)))
            labels=np.loadtxt(filename,dtype=np.float,delimiter=",", usecols=(C-1))
            labels=labels.astype("int")
            #sli=list(range(C-1))
            #data=file[:,[sli]]
            #data=np.reshape(data,(R,(C-1)))
            #labels=file[:,[(C-1)]]
            #labels=np.reshape(labels,(R))
            #labels=int(labels)
            return data,labels
        else:
            return file
    
    def testSOM(SOM,trainfile,trainlabels,testfile,testlabels,somside,plot=True):
        mapping = np.empty(shape=(int(somside),int(somside)), dtype=object)
        for i in range(int(somside)):
            for j in range(int(somside)):
                mapping[i][j] = []
        for t in range(len(trainfile)):
            (m_row, m_col) = closest_node(trainfile, t, SOM, somside, somside)
            mapping[m_row][m_col].append(trainlabels[t])
        label_map = np.zeros(shape=(int(somside),int(somside)), dtype=np.int)
        for i in range(int(somside)):
            for j in range(int(somside)):
                label_map[i][j] = int(most_common(mapping[i][j], 5))
        predicted=[]
        for i in range(len(testfile)):
            (m_row,m_col)=closest_node(testfile,i,SOM,int(somside),int(somside))
            value=label_map[m_row][m_col]
            predicted.append(int(value))
        predicted=np.array(predicted)
        targetnames=[1,2,3]
        from sklearn.metrics import classification_report, confusion_matrix
        if plot==True:
            print(classification_report(testlabels, predicted,labels=targetnames))
        conf=confusion_matrix(testlabels, predicted,labels=targetnames)
        return conf
    def winnerNodesArray(result1,result2,side1,side2):
        r1=(np.array(result1))/side1
        r2=(np.array(result2))/side2
        rfinal=np.concatenate([r1,r2])
        return rfinal
    def normalizearray(array):
        mini=np.min(array)
        maxi=np.max(array)
        array=(array-mini)/(maxi-mini)
        return array
    
    def testArquitechture(SOM1,SOM2,side1,side2,sidemmr,MMR,hebbian1,hebbian2,stimuli1,stimuli2,
                          dict1rows,dict2rows,labels):
        mapping=np.empty((int(sidemmr),int(sidemmr)),dtype=object)
        for n in range(int(sidemmr)):
            for m in range(int(sidemmr)):
                mapping[n][m]=[]
        for i in range(len(stimuli1)):
            winner1=winner_modalnode(stimuli1[i], SOM1, side1)
            winner2=winner_modalnode(stimuli2[i], SOM2, side2)
            row_1hebb=dict1rows[winner1]#hebbian conections
            row_2hebb=dict2rows[winner2]#hebbian conections
            multimodalstimuli=winnerNodesArray(winner1,winner2,side1,side2)
            activation_map=np.zeros(shape=(int(sidemmr),int(sidemmr)),dtype=float)
            ind1=hebbian1[row_1hebb]#crossmodal activation SOM1
            ind2=hebbian2[row_2hebb]#crossmodal activation SOM2
            ind=np.vstack((ind1,ind2))
            ind=np.sum(ind,axis=0)#learned multimodal from crossmodal
            ind=normalizearray(ind)
            counter=0
            for r in range(int(sidemmr)):
                for c in range(int(sidemmr)):
                    directactivation=dist(MMR[r][c],multimodalstimuli)
                    indirectactivation=ind[counter]
                    if indirectactivation==1:
                        indirectactivation=0.99
                    activation=directactivation*(1-indirectactivation)
                    activation=np.array(activation)
                    activation_map[r][c]=activation
                    counter+=1
            result=np.where(activation_map==np.amin(activation_map))
            x=int(result[0])
            y=int(result[1])
            mapping[x][y].append(labels[i])
        return mapping
    
    def umatrix(SOM,side,plottile):
        mapping=np.zeros((int(side),int(side)),dtype=float)
        for i in range(int(side)):
            for j in range(int(side)):
                interestnode=SOM[i][j]
                u=0
                amount=0
                up=i-1
                down=i+1
                left=j-1
                right=j+1
                if up>=0:
                    u+=dist(interestnode,SOM[up][j])
                    amount+=1
                if down<side:
                    u+=dist(interestnode,SOM[down][j])
                    amount+=1
                if right<side:
                    u+=dist(interestnode,SOM[i][right])
                    amount+=1
                if left>=0:
                    u+=dist(interestnode,SOM[i][left])
                    amount+=1
                if right<side and up>=0:
                    u+=dist(interestnode,SOM[up][right])
                    amount+=1
                if right<side and down<side:
                    u+=dist(interestnode,SOM[down][right])
                    amount+=1
                if down<side and left>=0:
                    u+=dist(interestnode,SOM[down][left])
                    amount+=1
                if up>=0 and left>=0:
                   u+=dist(interestnode,SOM[up][left])
                   amount+=1
                mapping[i][j]=u/amount
        from matplotlib.pyplot import imshow, title, colorbar,show
        imshow(mapping,cmap="autumn")
        title(plottile)
        colorbar()
        show()

        
    def mappingSOM(SOM,data,labels,side):
        mapping = np.empty(shape=(int(side),int(side)), dtype=object)
        for i in range(int(side)):
            for j in range(int(side)):
                mapping[i][j] = []
        for t in range(len(data)):
            (m_row, m_col) = closest_node(data, t, SOM, int(side), int(side))
            mapping[m_row][m_col].append(labels[t])
        return mapping
                    
    def plot_labelsSOM(ocurrencesMap,SOMside,imageName):
        label_map = np.zeros(shape=(int(SOMside),int(SOMside)), dtype=np.int)
        for i in range(int(SOMside)):
            for j in range(int(SOMside)):
                label_map[i][j] = most_common(ocurrencesMap[i][j], 50)
        plt.imshow(label_map, cmap=plt.cm.get_cmap('terrain_r'))
        plt.title(imageName)
        plt.colorbar()
        plt.show()
                    
    def plotpiechart(ocurrencesMap,side,labelslist):
        label=np.zeros((int(side),int(side),10),dtype=int)
        for i in range(int(side)):
            for j in range(int(side)):
                data=ocurrencesMap[i][j]
                counts=np.sum(data)
                count,uniqu=np.unique(data, return_counts=True)
                if counts!=0:
                    counti=uniqu/counts
                else:
                    counti=np.zeros(len(labelslist)+1,dtype=float)
                    counti[0]=100
                    uniqu=[0]
                    count=[0]
                array=np.zeros(10,dtype=float)
                for e in range(len(uniqu)):
                    indx=count[e]
                    #print(indx)
                    array[indx]=counti[e]
                array=array*100
                array.astype("int")
                label[i][j]=array
        c=0
        for i in range(int(side)):
            for j in range(int(side)):
                c+=1
                lislable=list(label[i][j])
                plt.subplot(int(side),int(side),int(c),aspect = 'equal')
                plt.tight_layout()
                plt.axis('equal')
                plt.pie(lislable)  
        plt.legend( ["NaN","ba","ga","da","McGurk"],loc="best", bbox_to_anchor=(0, 0), ncol=4)
        plt.subplots_adjust( wspace=0.001, hspace=0.001)
        plt.show()         
    def activationMatrix(SOM,somside,stimulidata,stimulilabels,title,plot=False):
        activationdictionary={}
        labelslist=list(np.unique(stimulilabels))
        for label in labelslist:
            mask=stimulilabels==label
            maskeddata=stimulidata[mask]
            counter=0
            for stimuli in maskeddata:
                activationmap=np.zeros((int(somside),int(somside)),dtype=float)
                for row in range(int(somside)):
                    for col in range(int(somside)):
                        activationmap[row][col]=dist(SOM[row][col],stimuli)
                activationmap=1-(normalizearray(activationmap))#inverse activation
                if counter==0:
                    result=activationmap
                    counter+=1
                else:
                    result=activationmap+result
            activationmap=normalizearray(activationmap)
            if plot==True:
                if label==1:
                    lab="Ba"
                elif label ==2:
                    lab="Ga"
                elif label==3:
                    lab="Da"
                elif label==4:
                    lab="McGurk"
                elif label==5:
                    lab="BaGa"
                elif label==6:
                    lab="BaDa"
                elif label==7:
                    lab="GaDa"
                elif label==8:
                    lab="DaBa"
                elif label==9:
                    lab="DaGa"
                else:
                    print("label "+str(label)+" is not recognized")
                    lab=label
                #fig= plt.figure(figsize=(15,15))
                plt.imshow(activationmap,cmap="rainbow")
                plt.tight_layout()
                plt.colorbar()
                plt.title(title+" activation map for "+str(lab)+" label")
                plt.show()
            activationmap=np.reshape(activationmap,(int(somside)*int(somside)))
            activationdictionary[lab]=activationmap
        activationdictionary=pd.DataFrame.from_dict(activationdictionary,orient="index")
        if plot==True:
            #fig= plt.figure(figsize=(15,15))
            pltlabels=activationdictionary.index.values.tolist()
            plt.plot(activationdictionary)
            plt.ylabel("Activation per neuron")
            plt.xlabel("Syllable")
            plt.title(title+" activation behavior")
            plt.show()
            actdictarr=activationdictionary.values
            #actdictarr=actdictarr[:3][:]#
            #pltlabels=pltlabels[:3]#
            plt.imshow(actdictarr,cmap="rainbow",aspect='auto',vmin=0,vmax=1)
            #plt.title(title)
            plt.colorbar()
            plt.yticks(np.arange(len(pltlabels)), pltlabels)
            plt.ylabel("Stimuli class")
            plt.tight_layout()
            plt.show()
            x=np.corrcoef(actdictarr)
            #fig= plt.figure(figsize=(10,10))
            plt.imshow(x,cmap="plasma",vmin=0,vmax=1)
            for i in range(len(pltlabels)):
                for j in range(len(pltlabels)):
                    text = plt.text(j, i, round(x[i, j],2), ha="center", va="center", color="k")
            plt.xticks(np.arange(len(pltlabels)), pltlabels)
            plt.xticks(rotation=90)
            plt.yticks(np.arange(len(pltlabels)), pltlabels)
            plt.title(title)#correlations title
            plt.colorbar()
            plt.tight_layout()
            plt.show()
        path="./"+title+".csv"
        activationdictionary.to_csv(path,index=True,header=True)
        return activationdictionary             
########
    def testArquitechture2(SOM1,SOM2,side1,side2,sidemmr,MMR,hebbian1,hebbian2,stimuli1,stimuli2,
                          dict1rows,dict2rows,labels,title,plot=True):
        activationdictionary={}
        crossmodaldictionary1={}
        crossmodaldictionary2={}
        labelslist=list(np.unique(labels))
        for label in labelslist:
            mask=labels==label
            stimuli1masked=stimuli1[mask]
            stimuli2masked=stimuli2[mask]
            mapping=np.empty((int(sidemmr),int(sidemmr)),dtype=object)
            for n in range(int(sidemmr)):
                for m in range(int(sidemmr)):
                    mapping[n][m]=[]
            count=0
            for i in range(len(stimuli1masked)):
                winner1=winner_modalnode(stimuli1masked[i], SOM1, side1)
                winner2=winner_modalnode(stimuli2masked[i], SOM2, side2)
                row_1hebb=dict1rows[winner1]#hebbian conection
                row_2hebb=dict2rows[winner2]#hebbian conection
                multimodalstimuli=winnerNodesArray(winner1,winner2,side1,side2)
                activation_map=np.zeros(shape=(int(sidemmr),int(sidemmr)),dtype=float)
                ind1=hebbian1[row_1hebb]#Crossmodal SOM1
                crossmodal1=normalizearray(ind1)
                ind2=hebbian2[row_2hebb]#Crossmodal SOM2
                crossmodal2=normalizearray(ind2)
                ind=np.vstack((ind1,ind2))
                ind=np.sum(ind,axis=0)
                ind=normalizearray(ind)#Multimodal learned
                counter=0
                actmapping=np.empty((int(sidemmr),int(sidemmr)),dtype=float)
                for r in range(int(sidemmr)):
                    for c in range(int(sidemmr)):
                        directactivation=dist(MMR[r][c],multimodalstimuli)
                        actmapping[r][c]=directactivation
                actmapping=1-(normalizearray(actmapping))
                indirectactivation=np.reshape(ind,(int(sidemmr),int(sidemmr)))
                activation=actmapping+indirectactivation
                activation=normalizearray(activation)
                if label==1:
                    lab="Ba"
                elif label ==2:
                    lab="Ga"
                elif label==3:
                    lab="Da"
                elif label==4:
                    lab="McGurk"
                elif label==5:
                    lab="BaGa"
                elif label==6:
                    lab="BaDa"
                elif label==7:
                    lab="GaDa"
                elif label==8:
                    lab="DaBa"
                elif label==9:
                    lab="DaGa"
                if videofiles==True:
                    outputpath="./"+lab+"/"
                    if not os.path.exists(outputpath):
                        os.makedirs(outputpath)

                    name=outputpath+"fig{0:04d}.png"
                    fig= plt.figure(figsize=(5,5))
                    plt.imshow(activation,cmap="rainbow")
                    fig.tight_layout()
                    fig.savefig(name.format(count),dpi=100)

                if count==0:
                    activationresult=activation
                    crossmodalresult1=crossmodal1
                    crossmodalresult2=crossmodal2
                    count+=1
                    
                else:
                    count+=1
                    activationresult+=activation
                    crossmodalresult1+=crossmodalresult1
                    crossmodalresult2+=crossmodalresult2
                result=np.where(activation==np.amax(activation))
                x=int(result[0])
                y=int(result[1])
                mapping[x][y].append(label)
            crossmodalresult1=np.reshape(crossmodalresult1,(int(sidemmr),int(sidemmr)))
            crossmodalresult1=normalizearray(crossmodalresult1)
            crossmodalresult2=np.reshape(crossmodalresult2,(int(sidemmr),int(sidemmr)))
            crossmodalresult2=normalizearray(crossmodalresult2)
            activationresult=normalizearray(activationresult)
            if plot==True:
                if label==1:
                    lab="BaBa"
                    crosslab1="-Ba"
                    crosslab2="Ba-"
                    print("Ploting crossmodal activations")
                    fig=plt.figure(figsize=(10,10))
                    plt.imshow(crossmodalresult1,cmap="rainbow")
                    plt.tight_layout()
                    plt.colorbar()
                    plt.title("Crossmodal activation map for "+str(crosslab1))
                    plt.show()
                    fig=plt.figure(figsize=(10,10))
                    plt.imshow(crossmodalresult2,cmap="rainbow")
                    plt.tight_layout()
                    plt.colorbar()
                    plt.title("Crossmodal activation map for "+str(crosslab2))
                elif label ==2:
                    lab="GaGa"
                    crosslab1="-Ga"
                    crosslab2="Ga-"
                    print("Ploting crossmodal activations")
                    fig=plt.figure(figsize=(10,10))
                    plt.imshow(crossmodalresult1,cmap="rainbow")
                    plt.tight_layout()
                    plt.colorbar()
                    plt.title(title+" crossmodal activation map for "+str(crosslab1))
                    plt.show()
                    fig=plt.figure(figsize=(10,10))
                    plt.imshow(crossmodalresult2,cmap="rainbow")
                    plt.tight_layout()
                    plt.colorbar()
                    plt.title(title+" crossmodal activation map for "+str(crosslab2))
                elif label==3:
                    lab="DaDa"
                    crosslab1="-Da"
                    crosslab2="Da-"
                    print("Ploting crossmodal activations")
                    fig=plt.figure(figsize=(10,10))
                    plt.imshow(crossmodalresult1,cmap="rainbow")
                    plt.tight_layout()
                    plt.colorbar()
                    plt.title(title+" crossmodal activation map for "+str(crosslab1))
                    plt.show()
                    fig=plt.figure(figsize=(10,10))
                    plt.imshow(crossmodalresult2,cmap="rainbow")
                    plt.tight_layout()
                    plt.colorbar()
                    plt.title(title+" crossmodal activation map for "+str(crosslab2))
                elif label==4:
                    lab="McGurk"
                elif label==5:
                    lab="BaGa"
                elif label==6:
                    lab="BaDa"
                elif label==7:
                    lab="GaDa"
                elif label==8:
                    lab="DaBa"
                elif label==9:
                    lab="DaGa"
                else:
                    print("label "+str(label)+" is not recognized")
                    lab=label
                fig= plt.figure(figsize=(10,10))
                plt.imshow(activationresult,cmap="rainbow")
                plt.tight_layout()
                plt.colorbar()
                plt.title(title+" activation map for "+str(lab)+" label")
                plt.show()
            #if label<4:
                #activationdictionary[crosslab1]=list(np.reshape(crossmodalresult1,(int(sidemmr*int(sidemmr)))))
                #activationdictionary[crosslab2]=list(np.reshape(crossmodalresult2,(int(sidemmr*int(sidemmr)))))
            activationdictionary[lab]=list(np.reshape(activationresult,(int(sidemmr)*int(sidemmr))))
        activationdictionary=pd.DataFrame.from_dict(activationdictionary,orient="index")
        if plot==True:
            pltlabels=activationdictionary.index.values.tolist()
            fig= plt.figure(figsize=(10,10))
            plt.plot(activationdictionary)
            plt.ylabel("Activation per neuron")
            plt.xlabel("Syllable")
            plt.title(title+"activation behavior")
            plt.show()
            actdictarr=activationdictionary.values
            #actdictarr=actdictarr[:3][:]#
            #pltlabels=pltlabels[:3]#
            plt.imshow(actdictarr,cmap="rainbow",aspect='auto',vmin=0,vmax=1)
            plt.yticks(np.arange(len(pltlabels)), pltlabels)
            #plt.title(title)
            plt.tight_layout()
            plt.ylabel("Stimuli class")
            plt.colorbar()
            plt.show()
            x=np.corrcoef(actdictarr)
            fig= plt.figure(figsize=(10,10))
            plt.imshow(x,cmap="rainbow",vmin=-1,vmax=1)
            for i in range(len(pltlabels)):
                for j in range(len(pltlabels)):
                    text = plt.text(j, i, round(x[i, j],2), ha="center", va="center", color="k")
            plt.xticks(np.arange(len(pltlabels)), pltlabels)
            plt.xticks(rotation=90)
            plt.yticks(np.arange(len(pltlabels)), pltlabels)
            #plt.title(title)#correlations
            plt.colorbar()
            plt.tight_layout()
            plt.show()
        path="./"+title+".csv"
        activationdictionary.to_csv(path,index=True,header=True)
        return mapping,activationdictionary

    #############################-MAIN-########################################
    #---------charging SOMS-------------------#
    videofiles=False
    print("Charging SOMS")
    MFCCfile="/media/marco/MarcoHDD/github/SOIMA/SOIMA_alpha0.3_20Steps/6613MFCCSOM_0.3alpha_20steps.csv"
    Lipfile="/media/marco/MarcoHDD/github/SOIMA/SOIMA_alpha0.3_20Steps/6620LipReadingSOM_0.3alpha_20steps.csv"
    MMRfile="/media/marco/MarcoHDD/github/SOIMA/SOIMA_alpha0.3_20Steps/10104MMRSOM_0.3alpha_20steps.csv"
    MFCC,mfcc_side=charge_som(MFCCfile)
    LipReading,lipreading_side=charge_som(Lipfile)
    MMR,mmr_side=charge_som(MMRfile)
    #---------charging hebbian maps-----------#
    print("Charging hebbian weights")
    mfcc_mmr="/media/marco/MarcoHDD/github/SOIMA/SOIMA_alpha0.3_20Steps/MFCC-MMRHebbianConections_0.3alpha_20steps.csv"
    lip_mmr="/media/marco/MarcoHDD/github/SOIMA/SOIMA_alpha0.3_20Steps/LipReading-MMRHebbianConections_0.3alpha_20steps.csv"
    Hebb_MFCC_MMR,mfcc_mmr_rows,mfcc_mmr_cols=charge_hebbian(mfcc_mmr, mfcc_side,mmr_side)
    Hebb_LipRead_MMR,lip_mmr_rows,lip_mmr_cols=charge_hebbian(lip_mmr,mfcc_side,mmr_side)
    #--------charging training and testing stimuli#
    sound_train= "/media/marco/MarcoHDD/github/stimuli/output_centralTendencies/output_StimulifromMultivariate/badaga_sound_train.csv"
    image_train="/media/marco/MarcoHDD/github/stimuli/output_centralTendencies/output_StimulifromMultivariate/badaga_train.csv"
    sound_test="/media/marco/MarcoHDD/github/stimuli/output_centralTendencies/output_StimulifromMultivariate/complete_sound_test.csv"
    image_test="/media/marco/MarcoHDD/github/stimuli/output_centralTendencies/output_StimulifromMultivariate/complete_image_test.csv"
    sound_traindata,sound_trainlabels=charge_files(sound_train)
    image_traindata,image_trainlabels=charge_files(image_train)
    sound_testdata,sound_testlabels=charge_files(sound_test)
    image_testdata,image_testlabels=charge_files(image_test)
    #--------------------Testing modal SOMS--------#
    MFCC_confussionmatrix=testSOM(MFCC, sound_traindata, sound_trainlabels, 
                                  sound_testdata, sound_testlabels, mfcc_side)
    LipReading_confussionmatrix=testSOM(LipReading, image_traindata, 
                                        image_trainlabels, image_testdata, 
                                        image_testlabels, lipreading_side)
    
    _,dict_trainSOIMA=testArquitechture2(MFCC,LipReading,mfcc_side,
                                       lipreading_side,mmr_side,MMR,
                                       Hebb_MFCC_MMR,
                                       Hebb_LipRead_MMR,sound_traindata,
                                       image_traindata,lip_mmr_rows,
                                       mfcc_mmr_rows,image_trainlabels,"MMR training set")
    _,dict_testSOIMA=testArquitechture2(MFCC,LipReading,mfcc_side,
                                       lipreading_side,mmr_side,MMR,
                                       Hebb_MFCC_MMR,
                                       Hebb_LipRead_MMR,sound_testdata,
                                       image_testdata,lip_mmr_rows,
                                       mfcc_mmr_rows,image_testlabels,"MMR testing set")
    TD_trainSOIMA=testArquitechture(MFCC,LipReading,mfcc_side,
                                       lipreading_side,mmr_side,MMR,
                                       Hebb_MFCC_MMR,
                                       Hebb_LipRead_MMR,sound_traindata,
                                       image_traindata,lip_mmr_rows,
                                       mfcc_mmr_rows,image_trainlabels)
    TD_testSOIMA=testArquitechture(MFCC,LipReading,mfcc_side,
                                       lipreading_side,mmr_side,MMR,
                                       Hebb_MFCC_MMR,
                                       Hebb_LipRead_MMR,sound_testdata,
                                       image_testdata,lip_mmr_rows,
                                       mfcc_mmr_rows,image_testlabels)
    print("plotting SOM-node piechart for training")
    plotpiechart(TD_trainSOIMA, mmr_side, [0,1,2,3])
    plotpiechart(TD_testSOIMA, mmr_side, [0,1,2,3,4,5,6,7,8,9])
    mfcc_mapping_tr=mappingSOM(MFCC, sound_traindata, sound_trainlabels, mfcc_side)
    lipreading_mapping_tr=mappingSOM(LipReading, image_traindata,
                                     image_trainlabels, lipreading_side)
    plotpiechart(mfcc_mapping_tr,mfcc_side,[0,1,2,3])
    plotpiechart(lipreading_mapping_tr, lipreading_side, [0,1,2,3,4])
    print("plotting U-matrixes")
    umatrix(MMR, mmr_side, "U-matrix for MMR")
    umatrix(LipReading,lipreading_side,"U-matrix for LipReading")    
    umatrix(MFCC,mfcc_side,"U-matrix for Syllable")
    print("calculating activation matrixes for modal congruent stimuli")
    sound_trainactivation=activationMatrix(MFCC,mfcc_side,sound_traindata,sound_trainlabels,"Auditory training set",plot=True)
    sound_testactivation=activationMatrix(MFCC,mfcc_side,sound_testdata,sound_testlabels,"Auditory testing set",plot=True)
    image_trainactivation=activationMatrix(LipReading,lipreading_side,image_traindata,image_trainlabels,"Visual training set",plot=True)
    image_testactivation=activationMatrix(LipReading,lipreading_side,image_testdata,image_testlabels,"Visual testing set",plot=True)
    
    
    
main()
"""
 MFCC_confussionmatrix=testSOM(MFCC, sound_traindata, sound_trainlabels, 
                                  sound_traindata, sound_trainlabels, mfcc_side)
    LipReading_confussionmatrix=testSOM(LipReading, image_traindata, 
                                        image_trainlabels,
                                        image_traindata, 
                                        image_trainlabels, lipreading_side)

Ba=dict_testSOIMA.loc["Ba",:]
Da=dict_testSOIMA.loc["Da",:]
Ga=dict_testSOIMA.loc["Ga",:]
Mc=dict_testSOIMA.loc["McGurk",:]
Ba=np.array(Ba)
Da=np.array(Da)
Ga=np.array(Ga)
Mc=np.array(Mc)
def prob(x):
    n=sum(x)
    for i in range(len(x)):
        x[i]=x[i]/n
    return x
bap=prob(Ba)
dap=prob(Da)
gap=prob(Ga)
mcp=prob(Mc)
bap=bap+0.001
dap=dap+0.001
gap=gap+0.001
mcp=mcp+0.001
def cross_entropy(p, q):
	return -sum([p[i]*np.log2(q[i]) for i in range(len(p))])

print(cross_entropy(bap, mcp))
print(cross_entropy(dap, mcp))
print(cross_entropy(gap, gap))
print(cross_entropy(mcp, mcp))
"""
