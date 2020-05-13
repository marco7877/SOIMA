#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:53:02 2020

@author: marco
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######
#
# Marco A. Flores-Coronado, Universidad Autónoma del Estado de Morelos (UAEM)
# 2020
#
# This code trains a Self-Organizing Map (SOM);after being trained, weights may
# be saved as .txt
# To plot the SOM, this script search for the winner nodes for each element from
# the training element (normalized in range 0:1), then it determines which type
# of data such node stands for by the most common element (label) identified with
# such node. Thus, que plotting still works for testing data.
#hennian learning only works for joining squared modal and amodal soms
#################### libraries #######################
import datetime
inicio=datetime.datetime.now()
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from math import sqrt
################### functions ########################

def closest_node(data, t, map, m_rows, m_cols):
    result=(0,0)
    small_dist = dist(data[t],map[0][0])#número inicial de distancia pequeña
    for i in range(m_rows):
      for j in range(m_cols):
        ed = dist(map[i][j], data[t])#esto saca la distancia euclideana entre
# el elemento t y el nodo por analizar i,j#####################################
        if ed < small_dist:# compra distancia euclideana para saber su fitness
          small_dist = ed# si sí, la menor distancia se reescribe
          result = (i, j)# si sí, resultado se sobrescriba con el nodo + fit
    return result
#----------------------NEW FUNCTIONS--------------------
def train1step(array,map,m_rows,m_cols,curr_range,curr_rate):
    result=(0,0)
    small_dist=dist(array,map[0][0])
    for i in range(m_rows):
        for j in range(m_cols):
            ed=dist(map[i][j],array)
            if ed <small_dist:
                small_dist=ed
                result=(i,j)
    bmu_row, bmu_col=result
    coord=coord_vec(bmu_row,bmu_col,m_rows,m_cols,curr_range)
    for i in range(coord[0],coord[1]):
        for j in range(coord[2],coord[3]):
            if eucl(bmu_row, bmu_col, i, j) < curr_range:
                map[i][j] = map[i][j] + (curr_rate * (array - map[i][j]))
    return map,result,ed

def winnerNodesArray(result1,result2,Rows1,Rows2):
    r1=(np.array(result1))/Rows1
    r2=(np.array(result2))/Rows2
    rfinal=np.concatenate([r1,r2])
    return rfinal

def hebbianmatrix(som1_rows,som1_cols,som2_rows,som2_cols):
    """
    

    Parameters
    ----------
    som1_rows : modal squared shape som rows.
    som1_cols : modal squared shape som cols
    som2_rows : multimodal squared shape som rows
    som2_cols : multimodal squared shape som cols

    Returns
    -------
    matrix : hebbian matrix to upgrade by coocurrences
    dictRows : rows translation in hebbian matrix of winner node from modal som
    dictCols : colss translation in hebbian matrix of winner node from amodal som

    """
    from itertools import product
    R=som1_cols*som1_rows
    C=som2_cols*som2_rows
    matrix=np.zeros((R,C),dtype=float)
    keys1=list(product((list(range(som1_rows))),repeat=2))
    keys2=list(product((list(range(som2_rows))),repeat=2))
    dictRows={ keys1[i] : i for i in range(0, len(keys1) ) }
    dictCols={ keys2[i] : i for i in range(0, len(keys2) ) }
    return matrix,dictRows,dictCols
def hebbian1step(hebbianMatrix,resultSOM1,resultSOMamodal,dictRows,dictCols,curr_rate,activationSOM1,activationSOM2):
    row=dictRows[resultSOM1]
    col=dictCols[resultSOMamodal]
    hebbianMatrix[row][col]+=curr_rate*activationSOM1*activationSOM2
    return hebbianMatrix
#--------------------------END NEW FUNCTIONS----------------------------------
def dist(v1, v2):
    return math.sqrt(sum((v1 - v2) ** 2))

def eucl (r1,c1,r2,c2):
    n1=np.array((r1,c1))
    n2=np.array((r2,c2))
    eud=sqrt(sum( (n1 - n2)**2 for n1, n2 in zip(n1, n2)))
    return eud

def most_common(lst, n):# sirve para plotear, busca más cumun
    if len(lst) == 0: return -1
    #print(lst)
    counts = np.zeros(shape=n+1, dtype=np.int)
    for i in range(len(lst)):
        winner=int(lst[i])
        counts[winner]+=1
    return np.argmax(counts)

def normal(x,g):
    return x/g


def coord_vec(x,y,Cols,Rows,rango):
    coord=[]
    xpos=int(x+rango)
    if xpos>Cols:
        xpos=Cols        
    xneg=int(x-rango)
    if xneg<0:
        xneg=0        
    ypos=int(y+rango)
    if ypos>Rows:
        ypos=Rows        
    yneg=int(y-rango)
    if yneg<0:
        yneg=0
    coord.append(xneg)
    coord.append(xpos)
    coord.append(yneg)
    coord.append(ypos)
    return coord
def somInitValues(matrix,docname,Cols,Rows):
    import os
    prototipe=[]
    for i in range(Cols):
        for j in range(Rows):
            datito=matrix[i][j]
            prototipe.append(datito)
    print("saving SOMs initial values")
    output_path="./SOMoutput/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    csv=output_path+docname+".csv"
    np.savetxt(csv,prototipe,fmt='%1.10f',delimiter=",")
def plot_som(SOM,Rows,Cols,data_x,data_y,imageName):
    mapping = np.empty(shape=(Rows,Cols), dtype=object)
    for i in range(Rows):
        for j in range(Cols):
            mapping[i][j] = []
    for t in range(len(data_x)):
        (m_row, m_col) = closest_node(data_x, t, SOM, Rows, Cols)
        mapping[m_row][m_col].append(data_y[t])
    label_map = np.zeros(shape=(Rows,Cols), dtype=np.int)
    for i in range(Rows):
        for j in range(Cols):
            label_map[i][j] = most_common(mapping[i][j], 50)
    plt.imshow(label_map, cmap=plt.cm.get_cmap('terrain_r'))
    plt.title(imageName)
    plt.colorbar()
    plt.show()

def plot_mmr(SOM,imageName):
    plt.imshow(SOM,cmap='terrain_r')
    plt.title(imageName)
    plt.colorbar()
    plt.show()

def saveMap (Map,Rows,Cols,Dim,LearnMax,StepsMax,modality,outputpath):
    data=[]
    for i in range(Cols):
        for j in range(Rows):
            datito=Map[i][j]
            data.append(datito)
    docname=str(Rows)+str(Cols)+str(Dim)+modality+"SOM_"+str(LearnMax)+"alpha_"+str(StepsMax)+"steps.csv"
    path=outputpath+"/"+docname
    np.savetxt(path,data,delimiter=",")
    inf=[int(Rows),int(Cols),int(Dim)]
    np.savetxt((outputpath+modality+"guide"),inf,delimiter=",")
    
def savehebbian (hebbianmatrix,modality,outputpath,LearnMax,StepsMax):
    docname=modality+"HebbianConections_"+str(LearnMax)+"alpha_"+str(StepsMax)+"steps.csv"
    path=outputpath+"/"+docname
    np.savetxt(path,hebbianmatrix,delimiter=",")
    x,y=hebbianmatrix.shape
    inf=[int(x),int(y)]
    np.savetxt((outputpath+modality+"guide"),inf,delimiter=",")
# ==================================================================

def main():
    print("charging data")
    Dim1 = 13# dimensiones del vector de entrada
    Rows1 = 6; Cols1 = 6#  tmaño m*n del SOM
    Dim2 = 20# dimensiones del vector de entrada
    Rows2 = 6; Cols2 = 6#  tmaño m*n del SOM
    Dim3 = 4# dimensiones del vector de entrada
    Rows3 = 10; Cols3 = 10#  tmaño m*n del SOM
    RangeMax1 = eucl(0,0,Rows1,Cols1)# cantidad de nodos, AKA= area
    RangeMax2 = eucl(0,0,Rows2,Cols2)# cantidad de nodos, AKA= area
    RangeMax3 = eucl(0,0,Rows3,Cols3)# cantidad de nodos, AKA= area
    LearnMax = 0.3# learning rate
    StepsMax = 10000#cantidad de permutaciones de entrenamiento
    originalvaluesdoc="6*6_SOM_sound_10000_alfa3_badaga_original"
    
    outputpath="./SOIMA_alpha"+str(LearnMax)+"_"+str(StepsMax)+"Steps/"
    savetext=True #T for saving final SOM weights
    plotsom=True
    saveinitial=True

    print("We're woking. With some luck, you won't fuck this shit up")
    data_file1 = "/media/marco/MarcoHDD/github/stimuli/output_centralTendencies/output_StimulifromMultivariate/badaga_sound_train.csv"
    data_file2 ="/media/marco/MarcoHDD/github/stimuli/output_centralTendencies/output_StimulifromMultivariate/badaga_train.csv"
    data_1 = np.loadtxt(data_file1, delimiter=",", usecols=range(0,Dim1),
    dtype=np.floating)# vector por sujeto
    data_2 = np.loadtxt(data_file2, delimiter=",", usecols=range(0,Dim2),
    dtype=np.floating)# vector por sujeto
    data_y1 = np.loadtxt(data_file1, delimiter=",", usecols=(Dim1),
                        dtype=np.float32)# labels
    data_y2 = np.loadtxt(data_file2, delimiter=",", usecols=(Dim2),
                        dtype=np.float32)# labels
    if len(data_1)!=len(data_2):
        print("Stimuli elements from modalities do not match")

    print("Starting training of SOIMA (2 modal soms, 1 amodal)")
    map1 = np.random.random_sample(size=(Rows1,Cols1,Dim1))
    map2=np.random.random_sample(size=(Rows2,Cols2,Dim2))
    map3=np.random.random_sample(size=(Rows3,Cols3,Dim3))
    if saveinitial==True:
        somInitValues(map1,originalvaluesdoc+"SOM1",Cols1,Rows1)
        somInitValues(map2,originalvaluesdoc+"SOM2",Cols2,Rows2)
        somInitValues(map3,originalvaluesdoc+"MMR",Cols3,Rows3)
        
    hebbMatrix1,Hebb1RowsDict,Hebb1ColsDict=hebbianmatrix(Rows1,Cols1,Rows3,Cols3)
    hebbMatrix2,Hebb2RowsDict,Hebb2ColsDict=hebbianmatrix(Rows2,Cols2,Rows3,Cols3)
    for s in range(StepsMax):
        
        pct_left = 1.0 - ((s * 1.0) / StepsMax)#saca la contraria de la prop##
#del código que se ha corrido         ##
        curr_range1 = (int)(pct_left * RangeMax1)#da el valor del vecindario de activación
        curr_range2 = (int)(pct_left * RangeMax2)#da el valor del vecindario de activación
        curr_range3 = (int)(pct_left * RangeMax3)#da el valor del vecindario de activación
        curr_rate = pct_left * LearnMax

        if s % (StepsMax/10) == 0: 
            print("Percent of current training: "+ str((s * 1.0) / StepsMax))
            print("Current learning rate for mod1, mod2, amod: "+str(curr_rate))
        index=list(range(len(data_1)))
        random.shuffle(index)
        for i in range(len(index)):
            t=index[i]
            array1=data_1[t]
            array2=data_2[t]
            map1,result1,activation1=train1step(array1,map1,Rows1,Cols1,curr_range1,curr_rate)
            map2,result2,activation2=train1step(array2,map2,Rows2,Cols2,curr_range2,curr_rate)
            array3=winnerNodesArray(result1, result2, Rows1, Rows2)
            map3,result3,activation3=train1step(array3,map3,Rows3,Cols3,curr_range3,curr_rate)
            #---Updating coactivation matrix ---#
            hebbMatrix1=hebbian1step(hebbMatrix1,result1,result3,Hebb1RowsDict,Hebb1ColsDict,curr_rate, activation1, activation3)
            hebbMatrix2=hebbian1step(hebbMatrix2,result2,result3,Hebb2RowsDict,Hebb2ColsDict,curr_rate, activation2, activation3)
        # --------------------------------------------------------------------#
    trai=datetime.datetime.now()
    print("SOIMA training for "+str(StepsMax)+" lasted: "+ str(trai-inicio))
    if plotsom==True:
        print("Plotting modal SOMs and MMR")
        plot_som(map1, Rows1, Cols1, data_1, data_y1, "Syllable SOM")
        plot_som(map2, Rows2, Cols2, data_2, data_y2, "Lip Reading SOM")
        plot_mmr(map3,"MMR weights")
    if savetext==True:
        print("Saving 2 modal SOMs, 1 amodal SOM and hebbian weights")
        import os
        if not os.path.exists(outputpath):
            os.mkdir(outputpath)
        saveMap(map1, Rows1, Cols1, Dim1, LearnMax, StepsMax, "MFCC", outputpath)
        saveMap(map2, Rows2, Cols2, Dim2, LearnMax, StepsMax, "LipReading", outputpath)
        saveMap(map3, Rows3, Cols3, Dim3, LearnMax, StepsMax, "MMR", outputpath)
        savehebbian(hebbMatrix1, "MFCC-MMR", outputpath, LearnMax, StepsMax)
        savehebbian(hebbMatrix2, "LipReading-MMR", outputpath, LearnMax, StepsMax)
        
if __name__=="__main__":
    main()

   
"""
    def charge_som(som_text):
        area,dim=som_text.shape
        lado=np.sqrt(area)
        som=np.empty(shape=(int(lado),int(lado),int(dim)),dtype=float)
        index=0
        for i in range(int(lado)):
            for j in range(int(lado)):
                som[i][j]=np.array(som_text[index])
                index+=1
        return som,dim,lado
    Rows,Cols,Dim=6,6,20
    data_file = "/media/marco/MarcoHDD/github/stimuli/output_centralTendencies/output_StimulifromMultivariate/badaga_train.csv"
    data_x = np.loadtxt(data_file, delimiter=",", usecols=range(0,Dim),
                        dtype=np.floating)# vector por sujeto
    data_y = np.loadtxt(data_file, delimiter=",", usecols=(Dim),
                        dtype=np.float)# labels
    data_y=data_y.astype('int')
    data_file = "/media/marco/MarcoHDD/github/SOM/SOMoutput/SOM_6-6_20000__alfa3_badaga.csv"
    data_som = np.loadtxt(data_file, delimiter=",",dtype=np.floating)
    map,_,_=charge_som(data_som)
    mapping = np.empty(shape=(Rows,Cols), dtype=object)
    for i in range(Rows):
        for j in range(Cols):
            mapping[i][j] = []
    for t in range(len(data_x)):
        (m_row, m_col) = closest_node(data_x, t, map, Rows, Cols)
        #print(m_row, m_col)
        mapping[m_row][m_col].append(data_y[t])
    label_map = np.zeros(shape=(Rows,Cols), dtype=np.int)
  
    for i in range(Rows):
        for j in range(Cols):
            elements=mapping[i][j]
            elements=list(elements)
            ocurr=elements.count(1)
            label_map[i][j] = ocurr
    #np.savetxt('test_gapdf_pymc3.csv',label_map,delimiter=",")
    print("And now, this is where fun beggins")
    plt.imshow(label_map, cmap=plt.cm.get_cmap('binary',25))
    plt.title("SOM_6-6_20000__alfa3_ga(test)")
    plt.colorbar()
    plt.show()
    
"""
