# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 02:44:35 2022

@author: lzhangGJ
"""
import torch
import torch.nn as nn
import torch.nn.functional as f

def sampEn(l,m,r):
    n = l.shape[0]
    xi = torch.stack( [f.pad(l,(m-i-1,i)) for i in range(m)])
    xj = xi[:,m-1:xi.shape[1]-1]
    xi = xi[:,m-1:xi.shape[1]-2]
    # compute b
    b = 0
    for i in range(xj.shape[1]):
        xj = torch.cat((xj[:,-1].unsqueeze(1),xj[:,:-1]),axis = 1)
        k = (torch.max(torch.abs(xi - xj[:,:-1]),dim = 0)[0]<=r)
        k =torch.sum(k) 
        b+=k

    b -= xi.shape[1]
    
    # compute a
    m+=1 
    xi = torch.stack( [f.pad(l,(m-i-1,i)) for i in range(m)])
    xj = xi[:,m-1:xi.shape[1]-2]
    xi = xi[:,m-1:xi.shape[1]-2]
    a = 0
    for i in range(xj.shape[1]):
        xj = torch.cat((xj[:,-1].unsqueeze(1),xj[:,:-1]),axis = 1)
        k = (torch.max(torch.abs(xi - xj),dim = 0)[0] <=r)
        k =torch.sum(k)
       
        a+=k
    a-=xj.shape[1]
    return -torch.log(a / b)