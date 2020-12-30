# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 22:01:37 2020

@author: wcfro
"""

import mastanpy.firstorderelastic3D as mp
import numpy as np


coord1 = np.array([0,0,0])
coord2 = np.array([0,80,0])
testnode1 = mp.Node_3d1el(1,coord1)
testnode2 = mp.Node_3d1el(2,coord2)
end_node = np.array([testnode1,testnode2])
print(testnode1)

"""
node class checks
coord  OK
dof    OK
number OK
"""


A,Ayy,Azz=35.3, 8.55, 23.03
Iy,Iz,J=495, 1380, 9.37
E,v=29000,0.3
w=np.array([0,0,0])
weight = 0
release = np.array([0,0])
beta = 0

testmember1 = mp.Element_3d1el(1,'W14x120',end_node,
			A,Ayy,Azz,Iy,Iz,J,E,v,w,weight,release,beta)

B = testmember1.T
C = testmember1.w

"""
element class checks:
length   OK
gamma   OK
k_local   OK
repr   OK
k_global   OK
DOF    OK
FEF    OK
FEF pin      OK
FEF pinpin   OK
FEF    pin   OK
k_pin        OK
k_pinpin     OK
selfweight   OK
diagonal selfweight       OK
gamma complex            OK!!!
beta angle             OK
beta + selfweight        OK
"""