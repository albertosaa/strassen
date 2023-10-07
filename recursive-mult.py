#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:20:17 2023

@author: asaa
"""

import time
import numpy as np 

def prod_usual(A,B):
    
    s = A.shape[0]
    s2 = int(s/2)
    
    
    if s == 1:
        C = np.matmul(A,B)
        
    else:
        
        A11 = np.empty((s2,s2))    
        A21 = np.empty((s2,s2))    
        A12 = np.empty((s2,s2))    
        A22 = np.empty((s2,s2))    
    
        B11 = np.empty((s2,s2))    
        B21 = np.empty((s2,s2))    
        B12 = np.empty((s2,s2))    
        B22 = np.empty((s2,s2))       
    
        for i in range (0,s2):
            for j in range (0,s2):
            
                A11[i,j] = A[i,j]
                B11[i,j] = B[i,j]
            
                A21[i,j] = A[i+s2,j]
                B21[i,j] = B[i+s2,j]

                A12[i,j] = A[i,j+s2]
                B12[i,j] = B[i,j+s2]

                A22[i,j] = A[i+s2,j+s2]
                B22[i,j] = B[i+s2,j+s2]
                
        C11 = prod_usual(A11,B11) + prod_usual(A12,B21) 
        C12 = prod_usual(A11,B12) + prod_usual(A12,B22) 
        C21 = prod_usual(A21,B11) + prod_usual(A22,B21) 
        C22 = prod_usual(A21,B12) + prod_usual(A22,B22)
    
        C = np.empty((s,s))  

        for i in range (0,s2):
            for j in range (0,s2):
            
                C[i,j] = C11[i,j]
                C[i,j+s2] = C12[i,j]
                C[i+s2,j] = C21[i,j]
                C[i+s2,j+s2] = C22[i,j]            
    
    return C


def prod_strassen(A,B):
    
    s = A.shape[0]
    s2 = int(s/2)
    
    
    if s == 1:
        C = np.matmul(A,B)
        
    else:
        
        A11 = np.empty((s2,s2))    
        A21 = np.empty((s2,s2))    
        A12 = np.empty((s2,s2))    
        A22 = np.empty((s2,s2))    
    
        B11 = np.empty((s2,s2))    
        B21 = np.empty((s2,s2))    
        B12 = np.empty((s2,s2))    
        B22 = np.empty((s2,s2))       
    
        for i in range (0,s2):
            for j in range (0,s2):
            
                A11[i,j] = A[i,j]
                B11[i,j] = B[i,j]
            
                A21[i,j] = A[i+s2,j]
                B21[i,j] = B[i+s2,j]

                A12[i,j] = A[i,j+s2]
                B12[i,j] = B[i,j+s2]

                A22[i,j] = A[i+s2,j+s2]
                B22[i,j] = B[i+s2,j+s2]
                
        P1 = A21 + A22
        P2 = P1 - A11
        P3 = A11 - A21
        P4 = A12 - P2
        
        Q1 = B12 - B11
        Q2 = B22 - Q1
        Q3 = B22 - B12
        Q4 = B21 - Q2

        M1 = prod_strassen(A11,B11)
        M2 = prod_strassen(A12,B21)
        M3 = prod_strassen(P1,Q1)
        M4 = prod_strassen(P2,Q2)
        M5 = prod_strassen(P3,Q3)
        M6 = prod_strassen(P4,B22)
        M7 = prod_strassen(A22,Q4)
        
        R1 = M1 + M4
        R2 = R1 + M5
        R3 = R1 + M3
            
                
        C11 = M1 + M2
        C12 = R3 + M6
        C21 = R2 + M7
        C22 = R2 + M3
    
        C = np.empty((s,s))  

        for i in range (0,s2):
            for j in range (0,s2):
            
                C[i,j] = C11[i,j]
                C[i,j+s2] = C12[i,j]
                C[i+s2,j] = C21[i,j]
                C[i+s2,j+s2] = C22[i,j]            
    
    return C

N = 2**8

A = np.random.rand(N,N)
B = np.random.rand(N,N)

sec0  = time.time()
C1 = np.matmul(A,B)
print(" Numpy matmul finished. ",time.time()-sec0) 

sec0  = time.time()
C2 = prod_usual(A,B) 
print(" Recursive usual multiplication finished. ",time.time()-sec0) 

sec0  = time.time()
C3 = prod_strassen(A,B) 
print(" Recursive Strassen multiplication finished. ",time.time()-sec0) 
 
print(N,'Error usual multiplication = ',np.linalg.norm(C1-C2))
print(N,'Error Strassen multiplication = ',np.linalg.norm(C1-C3))
