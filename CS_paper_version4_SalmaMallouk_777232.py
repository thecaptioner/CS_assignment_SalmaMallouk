#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 19:43:48 2025

@author: Salma Mallouk
"""

import json
import re
import numpy as np


def hash_function(v):
    a = np.arange(1,len(v)+1)*6703 + 5717
    hash_value = round(np.dot(a,v)) % 2063
    return hash_value



#%%

"""
In this part the data is extracted from the jsaon file and the model words are converted to binary vectors 
"""

#open the JASON file
with open('TVs-all-merged.json', 'r') as file:
    
    data = json.load(file)

#extract all the titles from the file
titles = []
shop = []
#brand = []
count = 0 
modelID = []
total_products = 1624

for key in data:
    ID = data[key]
    if len(ID) > 1: 
        count += 1
        for i in range(len(ID)): 
            titles.append(ID[i]["title"])
            shop.append(ID[i]["shop"])
            modelID.append(key)
    else:
        titles.append(ID[0]["title"])
        modelID.append(key)
        shop.append(ID[0]["shop"])
        #brand.append(ID[0]["featuresMap"]["Brand"])

#splitting the digits and words from the titles
#hash_1 = re.findall(r'\w+', titles[0])
#hash = titles[0].split()

words_split = []
for i in range(len(titles)):
    words_split.append(re.findall(r'\w+', titles[i]))

#creating binary vector product representations 
word_check = []
for i in range(len(titles)):
    for j in words_split[i]:
        if not j in word_check:
            word_check.append(j)

binary_vectors = np.zeros(shape = (1646,1624), dtype=int)
for i in range(len(words_split)):
    for j in range(len(words_split[i])):
        if words_split[i][j] in word_check:
            index = word_check.index(words_split[i][j])
            binary_vectors[index][i]=1
            
            

     

#%%

"""
In this part, the bootstrap is performed. The columns from binary_vectors are chosen randomly with replacement.
"""

binary_vectors_bootstrap = np.zeros(shape=(1646,1624), dtype=int)
binary_vectors_test = np.zeros(shape=(1646,1624), dtype=int)
corresponding_products = np.array([], dtype =int)

for i in range(total_products):
    selected = np.random.randint(total_products)
    if selected not in corresponding_products:
        corresponding_products = np.append(corresponding_products, selected)
        binary_vectors_bootstrap[:,selected] = binary_vectors[:,selected]
     
test_count = 0
test_products= np.array([], dtype =int)
for i in range(total_products):
    if i not in corresponding_products:
        test_products = np.append(test_products, i)
        binary_vectors_test[:,i] = binary_vectors[:,i]


#%%
"""
In this part, the minhashing is performed for the training set to decrease the size of the matrix. 
520 permutations are chosen because that is approximately half of the 
total number of products from the training set 
"""

index_minhash = np.arange(len(word_check))
column = np.array([], dtype = int)
minhashes = 520
signature_matrix = np.zeros(shape=(minhashes,total_products), dtype = int)
for n in range(minhashes):
    np.random.shuffle(index_minhash)
    for j in range(total_products):
        column = np.array([], dtype = int)
        for i in range(len(word_check)): 
            if binary_vectors_bootstrap[index_minhash[i],j]==1:
                column = np.append(column, index_minhash[i])
                signature_matrix[n,j]=column[0]
                

#print(np.sum(binary_vectors[:,1623]))

#%%

"""
In this part, the minhashing is performed for the test set to decrease the size of the matrix. 
300 permutations are chosen because that is approximately half of the 
total number of products
"""

index_minhash = np.arange(len(word_check))
column = np.array([], dtype = int)
minhashes_test = 520
signature_matrix_test = np.zeros(shape=(minhashes,total_products), dtype = int)
for n in range(minhashes):
    np.random.shuffle(index_minhash)
    for j in range(total_products):
        column = np.array([], dtype = int)
        for i in range(len(word_check)): 
            if binary_vectors_test[index_minhash[i],j]==1:
                column = np.append(column, index_minhash[i])
                signature_matrix_test[n,j]=column[0]
                





#%%

""" 
In this part, the LSH is performed. The parameters b and r are optimized 
For 810 minhashes the numbers could be used for b and r 1,2,3,5,6,9,10,15,18,27,30,45,54,81,90,135,162,270,405,810
"""
b_values = np.array([1, 2, 4, 5, 8, 10, 13, 20,26, 40, 52, 65, 104, 130, 260, 520])

PQ = np.array([])
PC = np.array([])
LSH_candidate_pairs_train = set()
LSH_true_pairs_train = set()
equal = 0
count_comparisons_train = 0

#print("Bands", "PQ", "PC", "F1", "t", sep = "\t\t\t")
for b in b_values:
    r = int(minhashes/b)
    t = (1/b)**(1/r) 
    LSH = np.array_split(signature_matrix, b)
    buckets = {}

    

    
    for j in corresponding_products:
        for band in range(b):
            bucket = hash_function(LSH[band][:,j])
            buckets.setdefault((band, bucket), []).append(int(j))        
    


    for key in buckets:
        length = len(buckets[key])    
        if length > 2:    
            for i in range(length):
                for j in range(i+1,length):
                    if shop[buckets[key][i]] is not shop[buckets[key][j]]:
                        LSH_candidate_pairs_train.add((buckets[key][i],buckets[key][j]))
                    if modelID[buckets[key][i]] == modelID[buckets[key][j]]:
                        LSH_true_pairs_train.add((buckets[key][i],buckets[key][j]))
                        equal += 1
                    #count_comparisons += 1


                        


PQ = np.append(PQ, len(LSH_candidate_pairs_train)/count_comparisons_train)
PC = np.append(PC, len(LSH_candidate_pairs_train)/count)

F1 = (2*PQ*PC)/(PC+PQ)
b_best = int(b_values[F1==np.max(F1)][0])
F1_best = np.max(F1)

print("band ", b_values[F1==np.max(F1)], "bijbehorende F1 score is ", np.max(F1))



#%%


equal = 0
for key in LSH_candidate_pairs_train:
    product1 = key[0]
    product2 = key[1]
    if shop[product1] is not shop[product2]:
        print(modelID[product1]) 
        equal += 1

        
#%%
"""
Use best bands value on test value and calculate F1 score 
"""

b = b_best
r = int(minhashes/b)
t = (1/b)**(1/r) 
LSH = np.array_split(signature_matrix_test, b)
buckets_test = {}
PQ_test = np.array([])
PC_test = np.array([])
LSH_candidate_pairs_test = set()



for j in corresponding_products:
    for band in range(b):
        bucket = hash_function(LSH[band][:,j])
        buckets_test.setdefault((band, bucket), []).append(int(j))        

count_comparisons_test =  0
equal_test = 0 

for key in buckets_test:
    length = len(buckets_test[key])
    if length == 2:
        count_comparisons_test += 1
        if modelID[buckets_test[key][0]] == modelID[buckets_test[key][1]]:
            LSH_candidate_pairs_test.add((modelID[buckets_test[key][0]],modelID[buckets_test[key][1]]))
            equal += 1
    elif length > 3:
        for i in range(length):
            for j in range(i,length):
                count_comparisons_test += 1
                if modelID[buckets_test[key][i]] == modelID[buckets_test[key][j]]:
                    LSH_candidate_pairs_test.add((modelID[buckets_test[key][i]], modelID[buckets_test[key][j]]))
                    equal_test += 1

PQ_test = np.append(PQ_test, equal/count_comparisons_test)
PC_test = np.append(PC_test, equal/count)
F1_test = (2*PQ*PC)/(PC+PQ)
F1_best_test = np.max(F1_test)
print(" F1 score is ", F1_best_test)



#%%

"""
Clustering will be performed with Agglomerative Clustering 
"""


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# N = total number of products
D = np.full((total_products, total_products), np.inf)

np.fill_diagonal(D, 0)   

for i, j in LSH_candidate_pairs:

    # RULE 1: same webshop → forbidden
    if shop[i] == shop[j]:
        continue

    # RULE 2: same brand → forbidden
    if brand[i] == brand[j]:
        continue

    # If we reach here → pair is valid
    sim = cosine_similarity(features[i].reshape(1,-1), features[j].reshape(1,-1))[0,0]
    dist = 1 - sim

    D[i,j] = dist
    D[j,i] = dist
    
model = AgglomerativeClustering(
    affinity='precomputed',
    linkage='complete',

)

labels = model.fit_predict(D)