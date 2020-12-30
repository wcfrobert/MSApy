
'''
###############################################################
MSApy - An Easy-to-use Python Package for Structural Analysis
###############################################################
This module include classes and functions for first-order elastic 3D analyses
Classes:
    Node_3d1el
    Element_3d1el
    Structure_3d1el
Function:
    excel_preprocessor - used in conjunction with input excel sheet
'''
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg
import time


class Node_3d1el:
    """
    Class definition for node objects.
    Properties:
        node_coord:             3x1 array containing x,y,z coordinate
        node_number:            tag of every node in the structure
        node_DOF:               6x1 array storing DOFs associated with each node
                                    Node 1 has DOF 1,2,3,4,5,6
                                    Node 2 has DOF 7,8,9,10,11,12
                                    and so on...
        node_disp:              6x1 array of nodal displacement (dx,dy,dz,rotx,roty,rotz)
        node_coord_new:         3x1 array of updated x,y,z coordinate after displacement (obtained after solving)
        node_reaction:          6x1 array of the support reactions if the node is fixed.
    """
    def __init__(self,node_number,node_coord):
        self.node_number = node_number
        self.node_coord = node_coord
        self.node_DOF = self.assign_dof()
        # Obtained after solution
        self.node_disp = np.zeros(6)
        self.node_coord_new = None
        self.node_reaction = 'None'

    def __str__(self):
        print("node number: ",self.node_number)
        print("node coordinate: ({:.2f}, {:.2f}, {:.2f})".format(self.node_coord[0], self.node_coord[1], self.node_coord[2]))
        print("node DOF: ",self.node_DOF)
        if type(self.node_disp) == type(None):
            print("No solution available yet")
        else:
            print("node displacement: {}".format(self.node_disp[0:3]))
            print("node reaction: {}".format(self.node_reaction))
        return '\n'

    def assign_dof(self):
        start_dof = 6*self.node_number - 5
        node_DOF = start_dof + np.array([0,1,2,3,4,5])
        return node_DOF

    def get_node_results(self,node_disp,react):
        # Called from Structure class after solving
        self.node_disp = node_disp
        self.node_coord_new = self.node_coord + self.node_disp[0:3]
        if np.count_nonzero(react)!=0:
            self.node_reaction = react

    def plot(self,fig,freeDOF_set,nodal_load,fixity):
        # plot for visualization
        x = [self.node_coord[0]]
        y = [self.node_coord[1]]
        z = [self.node_coord[2]]
        # give fixed node red, free node green color, and loaded nodes orange color
        nodeDOF_set = set(self.node_DOF)
        load_vector = nodal_load[self.node_number-1,:]
        hovertemplate ='<b>%{text}</b><extra></extra>'
        if len(set(nodeDOF_set & freeDOF_set))<6: # FIXED
            if np.count_nonzero(load_vector)==0:
                node_color = 'red'
                fixity = fixity[self.node_number-1,:]
                load_vector = 'None'
            else:
                node_color = 'red'
                fixity = fixity[self.node_number-1,:]
                load_vector = nodal_load[self.node_number-1,:] 
        else: # FREE
            if np.count_nonzero(load_vector)==0:
                node_color = 'green'
                fixity = 'Free'
                load_vector = 'None'
            else:
                node_color = 'orange'
                fixity = 'Free'
                load_vector = nodal_load[self.node_number-1,:]
        text = 'Node #: {}'.format(self.node_number) +\
                '<br>X: {:.1f}'.format(x[0])+\
                '<br>Y: {:.1f}'.format(y[0])+\
                '<br>Z: {:.1f}'.format(z[0])+\
                '<br>Boundary Condition: {}'.format(fixity)+\
                '<br>Loading: {}'.format(load_vector)
        plot_data = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker = dict(size = 3,color = node_color),
            hovertemplate = hovertemplate,
            text = [text])
        fig.add_trace(plot_data)

    def plot_results(self,fig,freeDOF_set,nodal_load,fixity,SCALE):
        # plot nodes after having solution
        # Plot original structure in grey
        x = [self.node_coord[0]]
        y = [self.node_coord[1]]
        z = [self.node_coord[2]]

        nodeDOF_set = set(self.node_DOF)
        load_vector = nodal_load[self.node_number-1,:]
        hovertemplate ='<b>%{text}</b><extra></extra>'
        if len(set(nodeDOF_set & freeDOF_set))<6: # FIXED
            node_color = 'red'
            text = 'Node #: {}'.format(self.node_number) +\
                '<br>Support node reactions:'+\
                '<br>RX: {:.2f}'.format(self.node_reaction[0])+\
                '<br>RY: {:.2f}'.format(self.node_reaction[1])+\
                '<br>RZ: {:.2f}'.format(self.node_reaction[2])+\
                '<br>MX: {:.2f}'.format(self.node_reaction[3])+\
                '<br>MY: {:.2f}'.format(self.node_reaction[4])+\
                '<br>MZ: {:.2f}'.format(self.node_reaction[5])
        else: # FREE
            text = 'Node #: {}'.format(self.node_number) +\
                '<br>Free node displacements:'+\
                '<br>dX: {:.3f}'.format(self.node_disp[0])+\
                '<br>dY: {:.3f}'.format(self.node_disp[1])+\
                '<br>dZ: {:.3f}'.format(self.node_disp[2])+\
                '<br>rotX: {:.3e}'.format(self.node_disp[3])+\
                '<br>rotY: {:.3e}'.format(self.node_disp[4])+\
                '<br>rotZ: {:.3e}'.format(self.node_disp[5])
            if np.count_nonzero(load_vector)==0:
                node_color = 'green'
            else:
                node_color = 'orange'

        plot_data = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker = dict(size = 3,color = 'grey'),
            hovertemplate = hovertemplate,
            text = [text])
        fig.add_trace(plot_data)

        # Plot deformed structure
        node_coord_new_scaled = self.node_coord + self.node_disp[0:3]*SCALE
        x = [node_coord_new_scaled[0]]
        y = [node_coord_new_scaled[1]]
        z = [node_coord_new_scaled[2]]
        plot_data = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker = dict(size = 2,color = node_color),
            hovertemplate = hovertemplate,
            text = [text])
        fig.add_trace(plot_data)


class Element_3d1el():
    """
    Class definition for element object.
    Properties:
        element_number:         tag of element
        element_name:           unique ID given to the element by the user. 
        element_DOF:            12x1 array containing the element's DOF
        end_nodes:              2x1 array containing i and j node (node object). Where
                                    i node is the beginning node, and j is the end node.
        beta:                   web rotation. Default value is 0. Refer to documentation
                                    for convention and default web direction vector orientation.
                                    User input should be in degrees.
        release:                2x1 integer array. Boolean indicating if beginning node or
                                    end node is pinned (1 = pinned). Currently, pinning a node will 
                                    release moment about both z and y axis. Torsion and axial release 
                                    has not yet been implemented.
                                    [nodei_pin_boolean, nodej_pin_boolean]
        self_weight_factor:     3x1 array with integers denoting gravity factors in the X, Y, and Z direction.
                                    To include self-weight, input -1 for Y-direction.
                                    (i.e. self-weight-factor = [0,-1,0])
        gamma:                  12x12 coordinate transformation matrix [T]
        k_local:                12x12 local stiffness matrix [Ke]
        k_global:               12x12 global stiffness matrix [T]'[Ke][T]
        f_local:                12x1 local element node force vector {F}
        f_global:               12x1 global element node force vector [T]'{F}
        d_local:                12x1 local element node displacement vector  [T]{u}
        d_global:               12x1 global element node displacement vector {u}
        FEF_local:              12x1 local equivalent fixed-end force {FEF}
        FEF_global:             12x1 global equivalent fixed-end force [T]'{FEF}
        w:                      3x1 uniform load in the elements local x,y,z axes{w}
        weight:                 self weight of element in force per unit length
        L:                      length of member    
        A:                      cross-sectional area
        Ayy:                    shear area along y-axis
        Azz:                    shear area along z-axis
        Iy:                     moment of inertia about minor axis
        Iz:                     moment of inertia about major axis
        J:                      torsion constant
        E:                      elastic modulus
        v:                      Poisson's ratio
        G:                      shear modulus
    """
    def __init__(
            self,element_number,element_name,end_nodes,
            A,Ayy,Azz,Iy,Iz,J,E,v,w,weight,release,beta,self_weight_factor):
        self.element_number = element_number
        self.element_name = element_name
        self.end_nodes = end_nodes
        self.A = A
        self.Ayy = Ayy
        self.Azz = Azz
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.E = E
        self.v = v
        self.w = w
        self.weight = weight
        self.G = E/(2*(1+v))
        self.beta = beta
        self.release = release
        self.self_weight_factor = self_weight_factor
        self.element_DOF = self.get_DOF()
        self.L = self.compute_length()
        self.T = self.compute_T()
        self.k_local, self.k_global = self.compute_K()
        self.FEF_local, self.FEF_global = self.uniform_load()
        # to be obtained after solution
        self.f_local = None
        self.f_global = None
        self.d_local = None
        self.d_global = None

    def __str__(self):
        print("Element #: ", self.element_number)
        print("Element ID: ", self.element_name)
        print("i node: N{}\nj node: N{}".format(self.end_nodes[0].node_number,self.end_nodes[1].node_number))
        print("DOF: ", self.element_DOF)
        print("Length: {:.3f}".format(self.L))
        print("A: {:.2f}".format(self.A))
        print("Ayy: {:.2f}".format(self.Ayy))
        print("Azz: {:.2f}".format(self.Azz))
        print("Iy: {:.2f}".format(self.Iy))
        print("Iz: {:.2f}".format(self.Iz))
        print("J: {:.2f}".format(self.J))
        print("E: {:.2f}".format(self.E))
        print("v: {:.2f}".format(self.v))
        print("G: {:.2f}".format(self.G))
        if type(self.d_local) == type(None):
            print("No solution available yet...")
        else:
            print("local end displacements [u]: ")
            print(self.d_local)
            print("local end forces [f]: ")
            print(self.f_local)
        return '\n'

    def get_DOF(self):
        element_DOF = np.zeros(12)
        element_DOF[0:6] = self.end_nodes[0].node_DOF
        element_DOF[6:12] = self.end_nodes[1].node_DOF
        return element_DOF

    def compute_length(self):
        i_coord = self.end_nodes[0].node_coord
        j_coord = self.end_nodes[1].node_coord
        L = np.linalg.norm(j_coord - i_coord)
        return L

    def compute_T(self):
        """
        Approach described in MASTAN textbook. See documentation for default
        web direction vector
        """
        # Direction cosine in x-direction
        i_coord = self.end_nodes[0].node_coord
        j_coord = self.end_nodes[1].node_coord
        position_vector = j_coord - i_coord
        cosinex = np.array([
            position_vector[0]/self.L,
            position_vector[1]/self.L,
            position_vector[2]/self.L])
        # Direction cosine in z
        if np.all(cosinex == [0,1,0]):
            cosinez = np.array([0,0,1])
        elif np.all(cosinex == [0,-1,0]):
            cosinez = np.array([0,0,-1])
        else:
            cosinez = np.cross(cosinex,np.array([0,1,0]))
            cosinez = cosinez/np.linalg.norm(cosinez)
        # Direction cosine in y (web-direction)
        cosiney = np.cross(cosinez,cosinex)
        cosiney = cosiney/np.linalg.norm(cosiney)
        # Beta angle rotation
        beta_rad = self.beta*math.pi/180
        gamma_b = np.array([
            [1,0,0],
            [0, math.cos(beta_rad), math.sin(beta_rad)],
            [0, -math.sin(beta_rad), math.cos(beta_rad)]])
        # Assemble gamma matrix
        gamma_p = np.zeros([3,3])
        gamma_p[0,:] = cosinex
        gamma_p[1,:] = cosiney
        gamma_p[2,:] = cosinez
        gamma = gamma_b@gamma_p
        self.gamma = gamma
        # Assemble transformation matrix
        zero = np.zeros([3,3])
        T = np.block([
            [gamma,zero,zero,zero],
            [zero,gamma,zero,zero],
            [zero,zero,gamma,zero],
            [zero,zero,zero,gamma]])
        return T

    def compute_K(self):
        """
        Shear deformation is included. Format of matrix follows the CEE 421L 
        course note by Dr. Henri Gavin. Theta_y and Theta_z defined here:
        Link: http://people.duke.edu/~hpgavin/cee421/frame-finite-def.pdf

        Pinned element stiffness does not include shear deformation.
        Shear deformation effect is not as pronounced in fixed-pin
        members. Refer to documentation for more details
        """
        A = self.A
        Ayy = self.Ayy
        Azz = self.Azz
        Iz = self.Iz
        Iy = self.Iy
        L = self.L
        J = self.J
        G = self.G
        E = self.E
        theta_y = 12*E*Iz/G/Ayy/L/L 
        theta_z = 12*E*Iy/G/Azz/L/L
        # decoupled stiffness matrices
        k_axial = np.array([
            [E*A/L,-E*A/L],
            [-E*A/L,E*A/L]])
        k_torsion = np.array([
            [G*J/L,-G*J/L],
            [-G*J/L,G*J/L]])
        k_flexureZ = np.array([
            [12*E*Iz/L**3/(1+theta_y), 6*E*Iz/L**2/(1+theta_y), -12*E*Iz/L**3/(1+theta_y), 6*E*Iz/L**2/(1+theta_y)],
            [6*E*Iz/L**2/(1+theta_y), (4+theta_y)*E*Iz/L/(1+theta_y), -6*E*Iz/L**2/(1+theta_y), (2-theta_y)*E*Iz/L/(1+theta_y)],
            [-12*E*Iz/L**3/(1+theta_y), -6*E*Iz/L**2/(1+theta_y), 12*E*Iz/L**3/(1+theta_y), -6*E*Iz/L**2/(1+theta_y)],
            [6*E*Iz/L**2/(1+theta_y), (2-theta_y)*E*Iz/L/(1+theta_y), -6*E*Iz/L**2/(1+theta_y), (4+theta_y)*E*Iz/L/(1+theta_y)]])
        k_flexureY = np.array([
            [12*E*Iy/L**3/(1+theta_z), -6*E*Iy/L**2/(1+theta_z), -12*E*Iy/L**3/(1+theta_z), -6*E*Iy/L**2/(1+theta_z)],
            [-6*E*Iy/L**2/(1+theta_z), (4+theta_z)*E*Iy/L/(1+theta_z), 6*E*Iy/L**2/(1+theta_z), (2-theta_z)*E*Iy/L/(1+theta_z)],
            [-12*E*Iy/L**3/(1+theta_z), 6*E*Iy/L**2/(1+theta_z), 12*E*Iy/L**3/(1+theta_z), 6*E*Iy/L**2/(1+theta_z)],
            [-6*E*Iy/L**2/(1+theta_z), (2-theta_z)*E*Iy/L/(1+theta_z), 6*E*Iy/L**2/(1+theta_z), (4+theta_z)*E*Iy/L/(1+theta_z)]])
        # Adjustment for moment-release Chapter 7 (Kassimali,2011). Note I am using 1e-9 instead of 0 to distinguish
        # released DOF. Otherwise shape error during for COO and CSR sparse matrix construction
        if self.release[0]==1 and self.release[1]==1:
            k_flexureZ = np.zeros([4,4])
            k_flexureY = np.zeros([4,4])
        elif self.release[0]==1:
            k_flexureZ = np.array([
                [3*E*Iz/L**3, 1e-9, -3*E*Iz/L**3, 3*E*Iz/L**2],
                [1e-9, 1e-9, 1e-9, 1e-9],
                [-3*E*Iz/L**3, 1e-9, 3*E*Iz/L**2, -3*E*Iz/L**2],
                [3*E*Iz/L**2, 1e-9, -3*E*Iz/L**2, 3*E*Iz/L]])
            k_flexureY = np.array([
                [3*E*Iy/L**3, 1e-9, -3*E*Iy/L**3, -3*E*Iy/L**2],
                [1e-9, 1e-9, 1e-9, 1e-9],
                [-3*E*Iy/L**3, 1e-9, 3*E*Iy/L**2, 3*E*Iy/L**2],
                [-3*E*Iy/L**2, 1e-9, 3*E*Iy/L**2, 3*E*Iy/L]])
        elif self.release[1]==1:
            k_flexureZ = np.array([
                [3*E*Iz/L**3, 3*E*Iz/L**2, -3*E*Iz/L**3, 1e-9],
                [3*E*Iz/L**2, 3*E*Iz/L, -3*E*Iz/L**2, 1e-9],
                [-3*E*Iz/L**3, -3*E*Iz/L**2, 3*E*Iz/L**3, 1e-9],
                [1e-9, 1e-9, 1e-9, 1e-9]])
            k_flexureY = np.array([
                [3*E*Iy/L**3, -3*E*Iy/L**2, -3*E*Iy/L**3, 1e-9],
                [-3*E*Iy/L**2, 3*E*Iy/L, 3*E*Iy/L**2, 1e-9],
                [-3*E*Iy/L**3, 3*E*Iy/L**2, 3*E*Iy/L**3, 1e-9],
                [1e-9, 1e-9, 1e-9, 1e-9]])
        # Combine into global stiffness matrix
        k_local = np.zeros([12,12])
        k_local[np.ix_([0,6],[0,6])] = k_axial
        k_local[np.ix_([3,9],[3,9])] = k_torsion
        k_local[np.ix_([1,5,7,11],[1,5,7,11])] = k_flexureZ
        k_local[np.ix_([2,4,8,10],[2,4,8,10])] = k_flexureY
        # Convert to global coordinate
        k_global = self.T.transpose()@k_local@self.T
        return k_local,k_global

    def uniform_load(self):
        L = self.L
        w = self.w
        # self-weight adjustment
        gamma = self.T[0:3,0:3]
        w_weight_global = np.array([self.weight,self.weight,self.weight]) * self.self_weight_factor
        w_weight_local = gamma @ w_weight_global
        w = w + w_weight_local
        # Fixed-end forces for fixed-fixed element
        FEF_local = np.array([
            -w[0]*L/2,
            -w[1]*L/2,
            -w[2]*L/2,
            0,
            w[2]*L**2/12,
            -w[1]*L**2/12,
            -w[0]*L/2,
            -w[1]*L/2,
            -w[2]*L/2,
            0,
            -w[2]*L**2/12,
            w[1]*L**2/12,])
        self.FEF_fixed = np.copy(FEF_local)
        # Adjustment for moment-release
        if self.release[0]==1 and self.release[1]==1:
            FEF_local = np.array([
            -w[0]*L/2,
            -w[1]*L/2,
            -w[2]*L/2,
            0,
            0,
            0,
            -w[0]*L/2,
            -w[1]*L/2,
            -w[2]*L/2,
            0,
            0,
            0,])
        elif self.release[0]==1:
            FEF_local = np.array([
            -w[0]*L/2,
            -3*w[1]*L/8,
            -3*w[2]*L/8,
            0,
            0,
            0,
            -w[0]*L/2,
            -5*w[1]*L/8,
            -5*w[2]*L/8,
            0,
            -w[2]*L**2/8,
            w[1]*L**2/8,])
        elif self.release[1]==1:
            FEF_local = np.array([
            -w[0]*L/2,
            -5*w[1]*L/8,
            -5*w[2]*L/8,
            0,
            w[2]*L**2/8,
            -w[1]*L**2/8,
            -w[0]*L/2,
            -3*w[1]*L/8,
            -3*w[2]*L/8,
            0,
            0,
            0,])
        # Convert to global coordinate
        FEF_global = self.T.transpose() @ FEF_local
        return FEF_local,FEF_global

    def force_recovery(self,node_disp):
        # Called from Structure class when solution is requested
        self.d_global = node_disp
        self.d_local = self.T @ node_disp
        # Moment release adjustment
        if self.release[0]==1:#node i released
            self.d_local[4]=0
            self.d_global[4]=0
            self.d_local[5]=0
            self.d_global[5]=0
        if self.release[1]==1:#node j released
            self.d_local[10]=0
            self.d_global[10]=0
            self.d_local[11]=0
            self.d_global[11]=0
        # Force recovery
        self.f_local = self.k_local@self.d_local + self.FEF_local
        self.f_global = self.T.transpose() @ self.f_local
        # Recover rotation at pinned ends
        if self.release[0]==1 or self.release[1]==1:
            self.recover_released_DOF()
        # return to recovered force to structure class
        return self.f_local

    def recover_released_DOF(self):
        """
        this method is called when element is pinned at one or both ends. The 
        rotation at released end is dependent on the solved DOFs. Refer to 
        Kassimali text Chapter 7.1
        """
        # Define variables for ease of writing out equations
        FMb_z=self.FEF_fixed[5]
        FMe_z=self.FEF_fixed[11]
        FVb_z=self.FEF_fixed[1]
        FVe_z=self.FEF_fixed[7]
        u2_z=self.d_local[1]
        u3_z=self.d_local[5]
        u5_z=self.d_local[7]
        u6_z=self.d_local[11]
        FMb_y=self.FEF_fixed[4]
        FMe_y=self.FEF_fixed[10]
        FVb_y=self.FEF_fixed[2]
        FVe_y=self.FEF_fixed[8]
        u2_y=self.d_local[2]
        u3_y=self.d_local[4]
        u5_y=self.d_local[8]
        u6_y=self.d_local[10]
        if self.release[0]==1 and self.release[1]==1:
            self.rot_yi=(u5_y-u2_y)/self.L -self.L/6/self.E/self.Iy*(2*FMb_y-FMe_y)
            self.rot_zi=(u5_z-u2_z)/self.L -self.L/6/self.E/self.Iz*(2*FMb_z-FMe_z)
            self.rot_yj=(u5_y-u2_y)/self.L -self.L/6/self.E/self.Iy*(2*FMe_y-FMb_y)
            self.rot_zj=(u5_z-u2_z)/self.L -self.L/6/self.E/self.Iz*(2*FMe_z-FMb_z)
        elif self.release[0]==1:
            self.rot_yj=self.d_local[10]
            self.rot_zj=self.d_local[11]
            self.rot_yi=3/2/self.L*(u5_y-u2_y) -u6_y/2 -self.L/4/self.E/self.Iy*FMb_y
            self.rot_zi=3/2/self.L*(u5_z-u2_z) -u6_z/2 -self.L/4/self.E/self.Iz*FMb_z
        elif self.release[1]==1:
            self.rot_yi=self.d_local[4]
            self.rot_zi=self.d_local[5]
            self.rot_yj=3/2/self.L*(u5_y-u2_y)-u3_y/2- self.L/4/self.E/self.Iy*FMe_y
            self.rot_zj=3/2/self.L*(u5_z-u2_z)-u3_z/2- self.L/4/self.E/self.Iz*FMe_z

    def plot(self,fig):
        # for visualization prior to solving
        if np.count_nonzero(self.w)==0:
        	member_color = "black"
        else:
        	member_color = "orange"
        # get end node coordinate.
        x = [self.end_nodes[0].node_coord[0], self.end_nodes[1].node_coord[0]]
        y = [self.end_nodes[0].node_coord[1], self.end_nodes[1].node_coord[1]]
        z = [self.end_nodes[0].node_coord[2], self.end_nodes[1].node_coord[2]]
        plot_data = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line = dict(color=member_color),
            hoverinfo='skip')
        fig.add_trace(plot_data)
        # plot an invisible node at midpoint to display element information
        x_mid = [(x[0] + x[1])/2]
        y_mid = [(y[0] + y[1])/2]
        z_mid = [(z[0] + z[1])/2]
        if np.count_nonzero(self.w)==0:
            member_load = 'None'
        else:
            member_load = self.w
        hovertemplate ='<b>%{text}</b><extra></extra>'
        text = 'Element #: {}'.format(self.element_number) +\
                '<br>Name: {}'.format(self.element_name)+\
                '<br>Length: {:.1f}'.format(self.L)+\
                '<br>i node: N{}'.format(self.end_nodes[0].node_number)+\
                '<br>j node: N{}'.format(self.end_nodes[1].node_number)+\
                '<br>End release: i={:.0f}, j={:.0f}'.format(self.release[0],self.release[1])+\
                '<br>Web direction vector: {}'.format(self.T[1,0:3])+\
                '<br>Member load: {}'.format(member_load)
        plot_data = go.Scatter3d(
            x=x_mid,
            y=y_mid,
            z=z_mid,
            mode='markers',
            opacity=0,
            marker = dict(size = 25),
            hovertemplate = hovertemplate,
            text = [text])
        fig.add_trace(plot_data)
        # plot moment releases if applicable
        if self.release[0]==1:
            self.plot_pins(fig,0,'black')
        if self.release[1]==1:
            self.plot_pins(fig,1,'black')

    def plot_pins(self,fig,release_end,user_color):
        # A helper method that plots the end release nodes if applicable
        x = [self.end_nodes[0].node_coord[0], self.end_nodes[1].node_coord[0]]
        y = [self.end_nodes[0].node_coord[1], self.end_nodes[1].node_coord[1]]
        z = [self.end_nodes[0].node_coord[2], self.end_nodes[1].node_coord[2]]
        # Find (x,y,z) at 10% and 90% of length
        x_10 = [0.9*x[0] + 0.1*x[1]]
        y_10 = [0.9*y[0] + 0.1*y[1]]
        z_10 = [0.9*z[0] + 0.1*z[1]]
        x_90 = [0.1*x[0] + 0.9*x[1]]
        y_90 = [0.1*y[0] + 0.9*y[1]]
        z_90 = [0.1*z[0] + 0.9*z[1]]
        # Add pin at these locations
        if release_end == 0:
            xx,yy,zz=x_10,y_10,z_10
        elif release_end ==1:
            xx,yy,zz=x_90,y_90,z_90
        plot_data = go.Scatter3d(
            x=xx,
            y=yy,
            z=zz,
            mode='markers',
            marker = dict(size = 3, color = user_color, symbol='circle-open'),
            hoverinfo = 'skip')
        fig.add_trace(plot_data)

    def plot_results(self,fig,plot_flag,SCALE,force_SCALE):
        # for visualization of global results
        # Plot original structure
        x0 = [self.end_nodes[0].node_coord[0], self.end_nodes[1].node_coord[0]]
        y0 = [self.end_nodes[0].node_coord[1], self.end_nodes[1].node_coord[1]]
        z0 = [self.end_nodes[0].node_coord[2], self.end_nodes[1].node_coord[2]]
        plot_data = go.Scatter3d(
            x=x0,
            y=y0,
            z=z0,
            mode='lines',
            line = dict(color='grey',width = 2),
            hoverinfo='skip')
        fig.add_trace(plot_data)
        # plot moment releases if applicable
        if self.release[0]==1:
            self.plot_pins(fig,0,'grey')
        if self.release[1]==1:
            self.plot_pins(fig,1,'grey')
        # plot an invisible node at midpoint to display element results
        x_mid = [(x0[0] + x0[1])/2]
        y_mid = [(y0[0] + y0[1])/2]
        z_mid = [(z0[0] + z0[1])/2]
        hovertemplate ='<b>%{text}</b><extra></extra>'
        text = 'Element #: {}'.format(self.element_number) +\
                '<br>Member fixed-end forces:' +\
                '<br>Pi: {:.2f};   Pj: {:.2f}'.format(self.f_local[0],self.f_local[6]) +\
                '<br>Ti: {:.2f};   Tj: {:.2f}'.format(self.f_local[3],self.f_local[9]) +\
                '<br>Vyi: {:.2f};   Vyj: {:.2f}'.format(self.f_local[1],self.f_local[7]) +\
                '<br>Mzi: {:.2f};   Mzj: {:.2f}'.format(self.f_local[5],self.f_local[11]) +\
                '<br>Vzi: {:.2f};   Vzj: {:.2f}'.format(self.f_local[2],self.f_local[8]) +\
                '<br>Myi: {:.2f};   Myj: {:.2f}'.format(self.f_local[4],self.f_local[10])
        plot_data = go.Scatter3d(
            x=x_mid,
            y=y_mid,
            z=z_mid,
            mode='markers',
            opacity=0,
            marker = dict(size = 25),
            hovertemplate = hovertemplate,
            text = [text])
        fig.add_trace(plot_data)
        # Plot deformed structure
        self.plot_results_overall(fig,plot_flag.upper(),SCALE)
        
        
    def plot_results_overall(self,fig,flag,SCALE):
        # method used to visualize element results in the global structure
        if flag == "D": # Displacement
            node_coord_new_scaled_i = self.end_nodes[0].node_coord + self.end_nodes[0].node_disp[0:3]*SCALE
            node_coord_new_scaled_j = self.end_nodes[1].node_coord + self.end_nodes[1].node_disp[0:3]*SCALE
            x = [node_coord_new_scaled_i[0], node_coord_new_scaled_j[0]]
            y = [node_coord_new_scaled_i[1], node_coord_new_scaled_j[1]]
            z = [node_coord_new_scaled_i[2], node_coord_new_scaled_j[2]]
            plot_data = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='lines',
                line = dict(color='red'),
                hoverinfo='skip')
            fig.add_trace(plot_data)

            # # Obtained longitudinal and transverse displacement in local coordinate
            # N_POINTS = 25
            # x_local = np.linspace(0,self.L,N_POINTS)
            # x_deformed,dy = self.plot_displacement_y(x_local)
            # _,dz = self.plot_displacement_z(x_local)
            # dx = x_deformed - x_local
            # # Convert to global coordinate
            # dxdydz_local = np.vstack([dx,dy,-dz])
            # dxdydz_global = self.gamma @ dxdydz_local
            # # Calculate (x,y,z) for plotting
            # xi,yi,zi = self.end_nodes[0].node_coord
            # xj,yj,zj = self.end_nodes[1].node_coord
            # x_global = np.linspace(xi,xj,N_POINTS) + dxdydz_global[0,:]*SCALE
            # y_global = np.linspace(yi,yj,N_POINTS) + dxdydz_global[1,:]*SCALE
            # z_global = np.linspace(zi,zj,N_POINTS) + dxdydz_global[2,:]*SCALE
            # # Add trace to plot
            # plot_data = go.Scatter3d(
            #     x=x_global,
            #     y=y_global,
            #     z=z_global,
            #     mode='lines',
            #     line = dict(color='red'),
            #     hoverinfo='skip')
            # fig.add_trace(plot_data)
        elif flag == "A": # Axial forces
            pass
        elif flag == "T": # Torsion
            pass
        elif flag == "VY": # Major-axis shear
            pass
        elif flag == "MZ": # Major-axis bending
            pass
        elif flag == "VZ": # Minor-axis shear
            pass
        elif flag == "MY": # Minor-axis bending
            pass
        else:
            raise RuntimeError('Result flag not found. Valid flag: d, A, T, Vy, Mz, Vz, My')
        fig.add_trace(plot_data)


    def plot_member_results(self,fig):
        N_POINTS = 100
        # plot member reference line
        fig.add_trace(go.Scatter(x=[0,self.L],y=[0,0],
                                mode='lines+markers+text',name="Member",
                                line = dict(color='black',width = 2),
                                marker = dict(size = 7),
                                hoverinfo = 'skip',
                                text=["i node","j node"],
                                textposition="top center",
                                textfont=dict(family="Arial",size=14,color="blue")))
        # for visualization of member-level results
        x = np.linspace(0,self.L,N_POINTS)
        y1=self.plot_axial(x)
        fig.add_trace(go.Scatter(x=x,y=y1,mode='lines',fill='tozeroy',name = 'Axial Force P(x)',visible = "legendonly"))
        y2=self.plot_torsion(x)
        fig.add_trace(go.Scatter(x=x,y=y2,mode='lines',fill='tozeroy',name = 'Torsion T(x)',visible = "legendonly"))
        y3=self.plot_Vy(x)
        fig.add_trace(go.Scatter(x=x,y=y3,mode='lines',fill='tozeroy',name = 'Major Axis Shear Vy(x)',visible = "legendonly"))
        y4=self.plot_Mz(x)
        fig.add_trace(go.Scatter(x=x,y=y4,mode='lines',fill='tozeroy',name = 'Major Axis Bending Mz(x)',visible = "legendonly"))
        y5=self.plot_Vz(x)
        fig.add_trace(go.Scatter(x=x,y=y5,mode='lines',fill='tozeroy',name = 'Minor Axis Shear Vy(x)',visible = "legendonly"))
        y6=self.plot_My(x)
        fig.add_trace(go.Scatter(x=x,y=y6,mode='lines',fill='tozeroy',name='Minor Axis Bending Mz(x)',visible="legendonly"))
        x7,y7=self.plot_displacement_y(x)
        fig.add_trace(go.Scatter(x=x7,y=y7,mode='lines',name = 'Major Axis Deflection',visible = "legendonly"))
        x8,y8=self.plot_displacement_z(x)
        fig.add_trace(go.Scatter(x=x8,y=y8,mode='lines',name = 'Minor Axis Deflection',visible = "legendonly"))

    def plot_axial(self,x):
        Pi = self.f_local[0]
        w = self.w[0]
        y = -x*w-Pi
        return y

    def plot_torsion(self,x):
        Ti = self.f_local[3]
        y = -np.ones(150)*Ti
        return y

    def plot_Vy(self,x):
        Vi = self.f_local[1]
        w = self.w[1]
        y = -x*w+Vi
        return y

    def plot_Mz(self,x):
        Vi = self.f_local[1]
        Mi = self.f_local[5]
        w = self.w[1]
        y = (-Mi+Vi*x+w*x/2)*-1 #plot +M on tension side
        return y

    def plot_Vz(self,x):
        Vi = self.f_local[2]
        w = self.w[2]
        y = -Vi-w*x
        return y

    def plot_My(self,x):
        # note peculiarity of minor axis due to right hand rule. x => right, z=> down
        Vi = self.f_local[2]
        Mi = self.f_local[4]
        w = self.w[2]
        y = (-Mi-Vi*x-w*x/2)*-1 #plot +M on tension side
        return y

    def plot_displacement_y(self,x):
        #Hermite cubic interpolation functions. Major axis deflection
        N1 = 1 - x/self.L
        N2 = 1 - 3*(x/self.L)**2 + 2*(x/self.L)**3
        N6 = x*(1-x/self.L)**2
        N7 = x/self.L
        N8 = 3*(x/self.L)**2 - 2*(x/self.L)**3
        N12 = x*((x/self.L)**2-x/self.L)
        u1=self.d_local[0]
        v2= self.d_local[1]
        rot6=self.d_local[5]
        u7=self.d_local[6]
        v8=self.d_local[7]
        rot12=self.d_local[11]
        axial_deformation = N1*u1 + N7*u7
        x_deformed = x + axial_deformation
        y = N2*v2 + N6*rot6 + N8*v8 + N12*rot12
        return x_deformed,y

    def plot_displacement_z(self,x):
        #Hermite cubic interpolation functions. Minor axis deflection
        N1 = 1 - x/self.L
        N3 = 1 - 3*(x/self.L)**2 + 2*(x/self.L)**3
        N5 = x*(1-x/self.L)**2
        N7 = x/self.L
        N9 = 3*(x/self.L)**2 - 2*(x/self.L)**3
        N11 = x*((x/self.L)**2-x/self.L)
        u1=self.d_local[0]
        v3= self.d_local[2]
        rot5=self.d_local[4]
        u7=self.d_local[6]
        v9=self.d_local[8]
        rot11=self.d_local[10]
        axial_strain = N1*u1 + N7*u7
        x_deformed = x + axial_strain
        z = N3*v3 + N5*rot5 + N9*v9 + N11*rot11
        return x_deformed,z

    


class Structure_3d1el():
    """
    Class definition of an structure object. 3D first-order elastic analysis.

    User Input Matrices:
        node_coord
            N_node x 3 matrix containing x,y,z coordinate of each node
            Row # indicate node ID, columns indicate x,y,z
        fixity
            N_node x 6 matrix containing boundary condition of each node's DOF.
            Row # indicate node ID, columns indicate each node's 6 DOF. Note
            that we can also assign support settlement here.
            0 = fixed, empty = free, <float> = prescribed displacement.
        nodal_load
            N_node x 6 matrix containing external applied loads
            Row # indicate node ID, columns indicate each node's 6 DOF
        member_load
            N_element x 3 matrix containing uniform load which references each
            element's local coordinate system. Row # indicate element ID, 
            column 1,2,3 indicate uniform load component in x,y,z direction, 
            respectively.
        connectivity
            A matrix that establishes member connectivity. The row # indicate 
            element ID. The columns indicate:
                1 - beginning node ID
                2 - end node ID
                3 - beginning node pin boolean
                4 - end node pin boolean
                5 - beta angle (member rotation)
        section
            A matrix that contains various section properties. Row # indicate
            element ID. The columns indicate various section properties
                1 - element name
                2 - self-weight
                3 - A
                4 - Az
                5 - Iz
                6 - Ay
                7 - Iy
                8 - E
                9 - v
                10 - J
        self_weight_factor
            3x1 array with integers denoting gravity factors in the X, Y, and Z direction.
            To include self-weight, input -1 for Y-direction (i.e. self-weight-factor = [0,-1,0])

    Result Matrices:
        DEFL
            N_node x 6 matrix containing nodal displacement results
        REACT
            N_node x 6 matrix containing reaction forces at fixed nodes
        ELE_FOR
            N_element x 12 matrix containing end-node forces of each element
                in local coordinate

    Other Class Properties:
        N_node:         number of nodes in the structure
        N_element:      number of element in the structure
        node_array:     an array holding all node objects
        element_array:  an array holdign all element objects
        freeDOF:        an array holding the DOF # of all free DOFs
        fixedDOF:       an array holding the DOF # of all fixed DOFs
        dispDOF:        an array holding the DOF # of all prescribed DOFs
        K_structure:    final assemble global stiffness matrix in CSR format
        AFLAG:          flag variable indicating if solution was successfully obtained
                            0 = failed or not calculated. 1 = success

        Structure stiffness matrix partition into nine sub-matrices. External 
        load vector, fixed-end forces, and displacement vectors are all 
        partitioned into three sub-vectors
            kff, kfn, kfs, knf, knn, kns, ksf, ksn, kss
            Pf, Pn, Ps
            feff,fefn,fefs
            df, dn, ds
        subscript f = free DOF, s = fixed DOF, n = prescribed DOF

    
    NOTES:
    1.) Node and element ID simply correspond to the row # of each of 
            the input matrices above. For example, the Iz value of element 6 is 
            given in the 6th row of the section matrix.
    2.) Instantiation of an structure class object does not prompt any 
            computation. Rather it simply sets up the nodes, elements, 
            connectivities, forces, and DOFS. Matrix inversion occurs when
            the "solve" method is invoked.
    3.) After the solution is obtained, the user may wish to access the above 
            result matrices directly. Alternatively, there are a variety of options
            for visualization. See documentation for detail.
    """
    def __init__(self,coord,fixity,connectivity,nodal_load,member_load,section,self_weight_factor=[0,0,0],shear_deformation=True,print_flag=True):
        self.print_flag = print_flag
        self.self_weight_factor = self_weight_factor
        self.shear_deformation = shear_deformation
        self.coord = np.array(coord)
        self.fixity = np.array(fixity)
        self.N_node = len(coord)
        self.N_element = len(connectivity)
        self.connectivity = np.array(connectivity)
        self.nodal_load = np.array(nodal_load)
        self.member_load = np.array(member_load)
        self.section = np.array(section)
        self.self_weight_factor = np.array(self_weight_factor)
        self.node_array = self.create_nodes()
        self.element_array = self.create_elements()
        self.freeDOF,self.fixedDOF,self.dispDOF = self.classify_DOF()
        # Results obtained after Structure.solve()
        self.DEFL = None
        self.REACT = None
        self.ELE_FOR = None
        self.AFLAG = 0
        if print_flag:
            print(self)

    def __str__(self):
        print("Structure has been assembled")
        print("Total number of nodes: ", self.N_node)
        print("Total number of elements: ", self.N_element)
        print("Total number of DOF: ", self.N_node*6)
        if self.AFLAG==0:
            print("Not solved yet. Use .solve() to obtain solution")
        else:
            print("Solution stored in matrices: .DEFL, .REACT, .ELE_FOR")
        return '\n'

    def solve(self):
        time_start = time.time()
        self.assemble_stiffness()
        self.partition_stiffness_matrix()
        self.assemble_load_vector()
        self.compute_disp()
        self.compute_reaction()
        self.force_recovery()
        self.recover_node_results()
        time_end = time.time()
        if self.print_flag:
            print("Analysis Completed! Elapsed time: {:.4f} seconds".format(time_end-time_start))
        self.AFLAG=1

    def create_nodes(self):
        node_array = []
        for i in range(self.N_node):
            node_array.append(Node_3d1el(i+1,self.coord[i,:]))
        return node_array

    def create_elements(self):
        element_array = []
        for i in range(self.N_element):
            element_name = self.section[i,0]
            i_node_index=int(self.connectivity[i,0]-1)
            j_node_index=int(self.connectivity[i,1]-1)
            i_node = self.node_array[i_node_index]
            j_node = self.node_array[j_node_index]
            end_nodes = np.array([i_node,j_node])
            A = self.section[i,2]
            Ayy = self.section[i,5]
            Azz = self.section[i,6]
            if self.shear_deformation == False:
                Ayy = np.inf
                Azz = np.inf
            Iy,Iz = self.section[i,4],self.section[i,3]
            J = self.section[i,9]
            E = self.section[i,7]
            v = self.section[i,8]
            w = self.member_load[i,:]
            weight = self.section[i,1]
            release = np.array([self.connectivity[i,2],self.connectivity[i,3]])
            beta = self.connectivity[i,4]
            element_array.append(Element_3d1el(i+1,element_name,end_nodes,
                A,Ayy,Azz,Iy,Iz,J,E,v,w,weight,release,beta,self.self_weight_factor))
        return element_array

    def classify_DOF(self):
        # Find index of free, fixed, and prescribed BC in fixity matrix
        fixity_t = self.fixity.transpose()
        fixed_index = np.nonzero(fixity_t==0)
        free_index = np.nonzero(np.isnan(fixity_t))
        disp_index = np.nonzero((fixity_t!=0) & (fixity_t==False))
        # Convert to linear index to obtain DOF
        freeDOF = []
        fixedDOF = []
        dispDOF = []
        for i in range(len(free_index[0])):
            row = free_index[0][i]+1
            col = free_index[1][i]+1
            freeDOF.append(6*col+row-6)
        for i in range(len(fixed_index[0])):
            row = fixed_index[0][i]+1
            col = fixed_index[1][i]+1
            fixedDOF.append(6*col+row-6)
        for i in range(len(disp_index[0])):
            row = disp_index[0][i]+1
            col = disp_index[1][i]+1
            dispDOF.append(6*col+row-6)
        # indices must be sorted for mesh indexing to work properly
        freeDOF = np.sort(freeDOF)
        fixedDOF = np.sort(fixedDOF)
        dispDOF = np.sort(dispDOF)
        return np.array(freeDOF),np.array(fixedDOF),np.array(dispDOF).astype(int)

    def assemble_stiffness(self):
        # Create matrix in COO sparse format
        COO_i=[]
        COO_j=[]
        COO_v=[]
        for ele in self.element_array:
            non_zero_index = ele.k_global.nonzero()
            for i in range(len(non_zero_index[0])):
                COO_i.append(int(ele.element_DOF[non_zero_index[0][i]])-1)
                COO_j.append(int(ele.element_DOF[non_zero_index[1][i]])-1)
                COO_v.append(ele.k_global[non_zero_index[0][i],non_zero_index[1][i]])
        # Convert to CSR sparse format
        self.K_structure = sp.sparse.csr_matrix( (COO_v,(COO_i,COO_j)))

    def partition_stiffness_matrix(self):
        # Partition K_structure
        f = self.freeDOF - 1
        n = self.dispDOF - 1
        s = self.fixedDOF - 1
        self.kff = self.K_structure[np.ix_(f,f)]
        self.kfn = self.K_structure[np.ix_(f,n)]
        #kfs = self.K_structure[np.ix_(f,s)]
        self.knf = self.K_structure[np.ix_(n,f)]
        self.knn = self.K_structure[np.ix_(n,n)]
        #kns = self.K_structure[np.ix_(n,s)]
        self.ksf = self.K_structure[np.ix_(s,f)]
        self.ksn = self.K_structure[np.ix_(s,n)]
        #kss = self.K_structure[np.ix_(s,s)]

    def assemble_load_vector(self):
        # Assemble external load vectors
        nodal_load_vector = self.nodal_load.reshape(self.N_node*6)
        self.Pf = nodal_load_vector[(self.freeDOF-1)]
        self.Ps = nodal_load_vector[(self.fixedDOF-1)]
        self.Pn = nodal_load_vector[(self.dispDOF-1)]
        # Assemble fixed-end force vectors
        fef = np.zeros(len(nodal_load_vector))
        for i in range(self.N_element):
            ele_DOF = (self.element_array[i].element_DOF - 1).astype(int)
            ele_FEF = self.element_array[i].FEF_global
            fef[np.ix_(ele_DOF)] = fef[np.ix_(ele_DOF)] + ele_FEF
        self.feff = fef[(self.freeDOF-1)]
        self.fefs = fef[(self.fixedDOF-1)]
        self.fefn = fef[(self.dispDOF-1)]
        if self.feff.size == 0:
            self.feff = 0
        if self.fefs.size == 0:
            self.fefs = 0
        if self.fefn.size == 0:
            self.fefn = 0
        
    def compute_disp(self):
        # Prescribed displacement already known
        fixity_vector = self.fixity.transpose().reshape(self.N_node*6)
        self.dn = fixity_vector[(self.dispDOF-1)]
        # Invert Kff to get nodal displacement
        self.df, converge_flag = sp.sparse.linalg.cg(self.kff,self.Pf - self.feff - self.kfn@self.dn)
        if converge_flag>0:
            raise RuntimeError(f"CG solver failed to converge after {converge_flag} iterations.\nStructure may be unstable or ill-conditioned")
        elif converge_flag<0:
            raise RuntimeError(f"Illegal input or breakdown.\nStructure may be unstable")
        # Assemble DEFL matrix
        self.DEFL = np.zeros(self.N_node*6)
        self.DEFL[np.ix_(self.freeDOF-1)] = self.df
        self.DEFL[np.ix_(self.dispDOF-1)] = self.dn
        self.DEFL = self.DEFL.reshape(self.N_node,6)

    def compute_reaction(self):
        if self.knn.size==0 or self.knf.size == 0 or self.ksn == 0:
            self.Rn = np.array([])
            self.Rs = self.ksf @ self.df + self.fefs - self.Ps
        else:
            self.Rn = self.knf @ self.df + self.knn@self.dn + self.fefn - self.Pn
            self.Rs = self.ksf @ self.df + self.ksn@self.dn + self.fefs - self.Ps
        # Assemble REACT matrix
        self.REACT = np.zeros(self.N_node*6)
        self.REACT[np.ix_(self.fixedDOF-1)] = self.Rs
        self.REACT[np.ix_(self.dispDOF-1)] = self.Rn
        self.REACT = self.REACT.reshape(self.N_node,6)

    def force_recovery(self):
        d_vector = self.DEFL.reshape(self.N_node*6)
        self.ELE_FOR = np.zeros([self.N_element,12])
        for i in range(self.N_element):
            DOFs = self.element_array[i].element_DOF.astype(int)
            element_d = d_vector[DOFs-1]
            self.ELE_FOR[i,:] = self.element_array[i].force_recovery(element_d)

    def recover_node_results(self):
        for i in range(self.N_node):
            node_disp = self.DEFL[i,:]
            react = self.REACT[i,:]
            self.node_array[i].get_node_results(node_disp,react)

    def plot(self):
        # Initialize figure
        fig = go.Figure()
        self.plot_origin_mark(fig)
        # Plot nodes
        for i in range(self.N_node):
            self.node_array[i].plot(fig,set(self.freeDOF),self.nodal_load,self.fixity)
        # Plot elements
        for i in range(self.N_element):
            self.element_array[i].plot(fig)
        # Add some annotations
        my_footnote = [dict(xref='paper', yref='paper', x=0.9, y=0, xanchor='right', yanchor='top',showarrow=False,
                              text = 'MSApy. Copyright (c) 2020 RW',
                              font=dict(family='Arial',size=12,color='rgb(150,150,150)'), align = "right"),
                        dict(xref='paper', yref='paper', x=0.15, y=1, xanchor='right', yanchor='top',showarrow=False,
                              text = "Color Legend: <br>Red = fixed, <br>Green = free, <br>Orange = loaded" +
                                '<br>Hover to see more info.',
                              font=dict(family='Arial',size=12,color='rgb(150,150,150)'), align = "left")]
        fig.update_layout(annotations=my_footnote)
        # Setting up camera buttons
        self.camera_buttons(fig)
        return fig

    def plot_results(self, plot_flag='D',plot_scale=64):
        # Initialize Figure
        fig = go.Figure()
        self.plot_origin_mark(fig)
        # Plot nodes
        for i in range(self.N_node):
            self.node_array[i].plot_results(fig,set(self.freeDOF),self.nodal_load,self.fixity,plot_scale)
        # Set up plot scale if plotting force diagrams
        if plot_flag.upper() == "D":
            force_scale = None
        else:
            force_scale = self.get_force_scale(plot_scale)
        # Plot elements
        for i in range(self.N_element):
            self.element_array[i].plot_results(fig, plot_flag, plot_scale, force_scale)
        # Add some annotation
        my_footnote = [dict(xref='paper', yref='paper', x=0.9, y=0, xanchor='right', yanchor='top',showarrow=False,
                              text='MSApy. Copyright (c) 2020 RW',
                              font=dict(family='Arial',size=12,color='rgb(150,150,150)')),
                        dict(xref='paper', yref='paper', x=0.15, y=1, xanchor='right', yanchor='top',showarrow=False,
                              text="Hover to see more info. <br>Plot Scale x{:d}".format(plot_scale),
                              font=dict(family='Arial',size=12,color='rgb(150,150,150)'),align="left")]
        fig.update_layout(annotations=my_footnote)
        # Setting up camera buttons
        self.camera_buttons(fig)
        return fig

    def plot_element_results(self,ele_number):
        fig = go.Figure()
        if ele_number>self.N_element or ele_number<1:
            raise RuntimeError('Invalid element number. Pick element between 1 to {}'.format(self.N_element))
        self.element_array[ele_number-1].plot_member_results(fig)
        my_footnote = [dict(xref='paper', yref='paper', x=1, y=-0.1,
                              xanchor='right', yanchor='top',
                              text='Click on legend to plot turn on and off'+
                                    '<br>+M on tension side. +P is tension',
                              font=dict(family='Arial',
                                        size=12,
                                        color='rgb(150,150,150)'),
                              showarrow=False)]
        myscene = dict(xaxis_title='Element Local x Axis',yaxis_title='Element Results')
        my_title = {'text': "Displaying Results for Element No. {}".format(ele_number),
                    'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}
        fig.update_layout(title=my_title,scene=myscene,
                            hoverlabel=dict(bgcolor="white"),
                            annotations=my_footnote,
                            xaxis_title='Element Local x Axis')
        return fig

    def plot_origin_mark(self,fig):
        """This function plots the cartesian coordinate marks and set the appropriate drawing scale"""
        # Find longest member length to define appropriate axis scale
        dx = (max(self.coord[:,0]) - min(self.coord[:,0]))
        dy = (max(self.coord[:,1]) - min(self.coord[:,1]))
        dz = (max(self.coord[:,2]) - min(self.coord[:,2]))
        dmax = max(dx,dy,dz)
        # Plot X (in blue)
        if dx != 0:
            X = go.Scatter3d(
                x=[0,dmax/14],
                y=[0,0],
                z=[0,0],
                mode='lines+text',
                hoverinfo = 'skip',
                line=dict(color='blue', width=4),
                text=["","X"],
                textposition="middle right",
                textfont=dict(
                    family="Arial",
                    size=8,
                    color="blue"))
            fig.add_trace(X)
        # Plot Y (in red)
        if dy != 0:
            Y = go.Scatter3d(
                x=[0,0],
                y=[0,dmax/14],
                z=[0,0],
                mode='lines+text',
                hoverinfo = 'skip',
                line=dict(color='red', width=4),
                text=["","Y"],
                textposition="top center",
                textfont=dict(
                    family="Arial",
                    size=8,
                    color="red"))
            fig.add_trace(Y)
        # Plot Z (in green)
        if dz != 0:
            Z = go.Scatter3d(
                x=[0,0],
                y=[0,0],
                z=[0,dmax/14],
                mode='lines+text',
                hoverinfo = 'skip',
                line=dict(color='green', width=4),
                text=["","Z"],
                textposition="middle center",
                textfont=dict(
                    family="Arial",
                    size=8,
                    color="green"))
            fig.add_trace(Z)
        # Adjust plot scale to be uniform:
        fig.update_layout(
            scene = dict(
                xaxis = dict(range = [min(self.coord[:,0])-dmax/12, max(self.coord[:,0])+dmax/12]),
                yaxis = dict(range = [min(self.coord[:,0])-dmax/12, max(self.coord[:,0])+dmax/12]),
                zaxis = dict(range = [min(self.coord[:,0])-dmax/12, max(self.coord[:,0])+dmax/12])
            )
        )
        # Other layout adjustments
        myscene = dict(
            xaxis = dict(showticklabels=False,gridcolor='white',backgroundcolor='white',showspikes=False,visible=False),
            yaxis = dict(showticklabels=False,gridcolor='white',backgroundcolor='white',showspikes=False,visible=False),
            zaxis = dict(showticklabels=False,gridcolor='white',backgroundcolor='white',showspikes=False,visible=False))
        if dz == 0:
            my_camera = dict(up=dict(x=0,y=1,z=0),center=dict(x=0,y=0,z=0),eye=dict(x=0,y=0,z=2))
        else:
            my_camera = dict(up=dict(x=0,y=1,z=0),center=dict(x=0,y=0,z=0),eye=dict(x=1.25,y=1.25,z=1.25))
        fig.update_layout(scene=myscene,showlegend=False,hoverlabel=dict(bgcolor="white"),
                            scene_camera=my_camera,scene_dragmode='pan',
                            margin=dict(l=50,r=50,b=100,t=10,pad=4),
                            height = 650)





    def get_force_scale(self,user_scale):
        """ 
        A helper function to set the appropriate size of force diagram plot. The function will determine
        the maximum fixed-end force. Non-max values will be scaled accordining. 

        Returns a vector with scale factors for [A,T,Vy,Mz,Vz,My]. If the maximum value is zero 
        (e.g. no torsion in the system). The length will be set to zero and nothing will be
        plotted. 

        In essence, we are normalizing all force values by a maximum
        For example. A member end has 1200 kip.ft. The maximum in the structure is 2400 kip.ft.
        This end will be normalized to 0.5, which means it will be half of the longest possible line
        """
        # force_scale=[]
        # # determine the maximum length to plot. Similar to the origin mark, use longest member / 9
        # dx = (max(self.coord[:,0]) - min(self.coord[:,0]))/9
        # dy = (max(self.coord[:,1]) - min(self.coord[:,1]))/9
        # dz = (max(self.coord[:,2]) - min(self.coord[:,2]))/9
        # scaling = user_scale/64
        # default_scale = max(dx,dy,dz) * scaling
        # # determine maximum member-end forces
        # for i in range(N_element):
        #     self.ELE_FOR[:,0]+
        # return force_scale
        pass

    def camera_buttons(self, fig):
        # Include a button on the plot that allows user to switch between views
        button1 = dict(
            method = "relayout",
            args=[{"scene.camera.up": {'x':0,'y':1,'z':0},
                     "scene.camera.eye":{'x':1.25,'y':1.25,'z':1.25},
                     "scene.camera.center":{'x':0,'y':0,'z':0}}], 
            label = "Axonometric")
        button2 = dict(
            method = "relayout",
            args=[{"scene.camera.up":{'x':0, 'y':1,'z':0},
                     "scene.camera.eye":{'x':0,'y':2,'z':0},
                     "scene.camera.center":{'x':0,'y':0,'z':0}}], 
            label="Top View XZ")
        button3 = dict(
            method = "relayout",
            args=[{"scene.camera.up":{'x':0, 'y':1,'z':0}, 
                    "scene.camera.eye":{'x':0,'y':0,'z':2},
                    "scene.camera.center":{'x':0,'y':0,'z':0}}], 
            label="Plane View XY")
        button4 = dict(
            method = "relayout",
            args=[{"scene.camera.up":{'x':0, 'y':1,'z':0}, 
                    "scene.camera.eye":{'x':2,'y':0,'z':0},
                    "scene.camera.center":{'x':0,'y':0,'z':0}}], 
            label="Plane View ZY")
        fig.update_layout(updatemenus=[dict(buttons=[button1, button2, button3, button4],
                                                direction="right",
                                                pad={"r": 10, "t": 10},
                                                showactive=True,
                                                x=0.1,
                                                xanchor="center",
                                                y=0.1,
                                                yanchor="top")])


"""HELPER FUNCTIONS BELOW"""
def excel_preprocessor(file_name,print_flag=True):
    """
    This is a preprocessor that reads from the input excel sheet and returns
    the appropriate input matrices
    """
    a=pd.read_excel(file_name,sheet_name='node_coord',skiprows=0,usecols='B:D')
    a.dropna(inplace=True)
    N_node=a.count()[0]
    coord = a.to_numpy()

    b=pd.read_excel(file_name,sheet_name='connectivity',skiprows=0,usecols='B:F')
    b.dropna(inplace=True,how='all')
    b.fillna(0,inplace=True)
    N_element = b.count()[0]
    connectivity = b.to_numpy()

    c=pd.read_excel(file_name,sheet_name='fixity',skiprows=0,usecols='B:G',nrows=N_node)
    fixity = c.to_numpy()

    d=pd.read_excel(file_name,sheet_name='nodal_load',skiprows=0,usecols='B:G',nrows=N_node)
    d.fillna(0,inplace=True)
    nodal_load = d.to_numpy()
    
    e=pd.read_excel(file_name,sheet_name='member_load',skiprows=0,usecols='B:D',nrows=N_element)
    e.fillna(0,inplace=True)
    member_load = e.to_numpy()

    f=pd.read_excel(file_name,sheet_name='section',skiprows=0,usecols='B:K',nrows=N_element)
    f.fillna(0,inplace=True)
    section=f.to_numpy()

    if print_flag:
        print("Input matrices generated")

    return coord, connectivity, fixity, nodal_load, member_load, section
