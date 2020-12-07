Development in progress....

# MSApy - Matrix Structural Analysis in Python

MSApy is a python package built for simple 3-D structural analysis. It is an implementation of the direct stiffness method which allows for the analysis of any three-dimensional truss or frame structures. Perform quick and simple analyses right in your browser (e.g. Jupyter notebook) or any IDE of your choice.


* Arbitrary structure (beams, frames, trusses) in 2D and 3D
* Nodal loads in any global direction
* Member uniform load in three member local axis (wx,wy,wz)
* Moment released Member (pin-fixed or pin-pin)
* Member rotation (web-direction vector)
* Shear deformation included
* Support settlement
* Self straining load
* AISC W and HSS sections available
* Beautiful visualization and web interactivity with Plotly
	* Plot overall structure displacement (straight or cubic spline curves)
	* Force diagrams (P, T, Vy, Mz, Vz, My)
	* Individual detail plots

While the current implementation is limited to elastic analyses with "stick" elements, the lack of complexity is balanced by its simplicity. The intent has always been to create an extremely easy-to-use, barebone analysis framework that anyone with a computer and knowledge of python can use; all free of charge. Ideally MSApy can fill the niche between complicated commercial softwares that are not free and the academic softwares like OpenSees that are hard to learn/use.

# Important Notes and Assumptions  
* MSApy is agnostic when it comes to unit. Stay consistent in terms of units.
* Global Coordinate (X,Y,Z): 
	* Y is the vertical axis (Elevation)
	* X and Z are the axes within the horizontal plane (Plan)
* Local Coordinate (x,y,z):
	* x-axis = longitudinal axis
	* z-axis = major bending axis
	* y-axis = minor bending axis (along depth of element)
* Current implementation of moment-released member do not account for shear deformation but the effect of this is minor (see verification folder for more details)
* The default member orientation (web direction vector) follows the following algorithm.
	1. Local x-axis is defined by i and j node
	2. If +x is aligned with Y (vertical member), then +y is aligned with -X (always strong axis bending in plane)
	3. In other cases, +y will always have positive component with +Y
	4. After defining x and y axis, local z-axis is defined to be consistent with right-hand rule
	5. after initial assignment, the element may be rotated by user-specified beta angle.


# Syntax Example

Importing module:
```python
import msapy.firstorderelastic3D as mp
```

Creating structure:
```python
coord, connectivity, fixity, \
nodal_load, member_load, section = mp.excel_preprocessor('inputfile.xlsx')
self_weight_modifier = [0,0,0]
structure1 = mp.Structure_3d1el(coord,fixity,connectivity,nodal_load,member_load,section,self_weight_modifier)
```

Solving structure:
```python
structure1.solve()
```

Visualization:
```python
structure1.plot()
structure1.plot_results()
structure1.plot_member_results(ele_number)
```

If using IDE like spyder, it may be necessary to change your renderer
```python
import plotly.io as pio
pio.renderers.default = "browser"
fig.structure1.plot()
fig.show()
```


# Verification  
Look in the verification folder for demonstration of the various capabilities of MSApy.


# Analysis Modules 
Currently avaiable:
* firstorderelastic3D.py  - 3D first-order elastic analysis

In the future:
* firstorderelastic2D.py - 2D first-order elastic analysis
* secondorderelastic2D.py - 2D second-order elastic analysis
* secondorderinelastic2D.py  - 2D second-order inelastic analysis

* secondorderelastic3D.py  - 3D second-order elastic analysis
* secondorderinelastic3D.py  - 3D second-order inelastic analysis

* modal2D.py  - 2D modal analysis
* modal3D.py  - 3D modal analysis
* eigen2D.py  - 2D critical load analysis
* eigen3D.py  - 3D critical load analysis

* RSA.py - response spectrum analysis
* TH.py - time history analysis

* codecheck.py - various functions for code checking members (AISC, ACI, ASCE)


# References 
* Kassimali, A. (2012). Matrix analysis of structures. USA: Cengage Learning. 
* McGuire, W., Gallagher, R., Ziemian, R. (2014). Matrix Structural Analysis. Second Edition. 
* Bathe, K. (2014). Finite Element Procedures. Second Edition. Prentice Hall.

The entire technical foundation of this project is based on the matrix analysis textbook written by Richard Gallagher, Ronald Ziemian, and William McGuire, henceforth referred to as MGZ textbook. Furthermore, My graduate course note in CEE 280 Advanced Structural Analysis, and CEE 282 Nonlinear Structural Analysis, both of which are taught by Dr. Greg Deierlein at Stanford, were tremendously helpful in elucidating some of the more complex subjects within the MGZ textbook. I've gone ahead and listed Bathe's FEM textbook too despite not using it so far because I know I'll eventually use it.

The data structure is inspired by MASTAN2, a free MATLAB-based program that accompanies the MGZ textbook. User input data is inspired by the intuitive and user-friendly spreadsheet-style input of RISA 3D. The object-oriented design of three distinct classes (nodes, element, and structure) is entirely the conception of Dr. Reagan Chandramohan during his tenure as a TA for Dr. Deierlein's CEE 280 class. MSApy started off as a simple conversion of MATLAB code from the CEE 280 programming project into python. Later on, many additional features were added. Most notably visualization options offered by the Plotly python packages.

Other resources that were helpful:
* CEE 421L course note by Dr. Henri Gavin. Link: http://people.duke.edu/~hpgavin/cee421/frame-finite-def.pdf
* CEE 281 course note by Dr. Ronaldo Borja
* OpenSeespy Visualization module by Seweryn Kokot: https://github.com/zhuminjie/OpenSeesPy/blob/master/openseespy-pip/openseespy/postprocessing/ops_vis.py



# License Info

MIT License  
Copyright (c) [2020] [Robert Wang]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

DISCLAIMER:  
Considerable time, effort and expense have gone into the development, documentation and testing of MSApy. In using the program, the user accepts and understands that no warranty is expressed or implied by the developers on the accuracy or the relaibility of the program. The user msut explicitly understand the assumptions of the program and must independently verify the results.



Given the nature of our profession, I feel it necessary to emphasize on the above disclaimer. Please note MSApy is a one-man project and can in no way, shape or form compete with the rigor of the debugging process at commercial software companies. Please forward your suggestions, feedbacks, criticisms, and comments on bugs here on Github or send me an email at rwang01 [at] stanford [dot] edu. Thank you!

Robert

Last Updated: 2020-11-12
