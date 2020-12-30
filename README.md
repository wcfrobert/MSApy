# MSApy - Matrix Structural Analysis in Python

MSApy is a free-to-use, open-source python package for matrix structural analysis. It is an implementation of the direct stiffness method. Notable features are listed below. Perform quick analyses right in your Jupyter Notebook or IDE of your choice.



(placeholder for screenshots)













<u>Notable Features:</u>


* Simple to learn and use. Optional user-input Excel sheet

* Analysis of beams, frames, trusses in both 2-D and 3-D

* Nodal loads in any global direction

* Member uniform load in three member local axis (wx, wy, wz)

* Moment released Member (pin-fixed or pin-pin)

* Member rotation (web-direction vector)

* Shear deformation

* Support settlement

* AISC W and HSS sections available

* Beautiful visualization and web interactivity with Plotly.js
	* Plot overall structure displacement (straight or cubic spline curves)
	* Force diagrams (P, T, Vy, Mz, Vz, My)
	* Individual detail plots

Planned in the future:

* Second-order elastic analysis (2-D and 3-D)
* Second-order inelastic analysis (2-D)
* Modal and eigen value analysis (2-D and 3-D)



<br>

# Getting Started

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

If using IDE like spyder or vscode, it may be necessary to change your default renderer

```python
import plotly.io as pio
pio.renderers.default = "browser"
fig.structure1.plot()
fig.show()
```

Refer to the "verification_examples" folder for some sample input spreadsheets.

<br>

# Important Notes and Assumptions  

* MSApy is agnostic when it comes to unit. Stay consistent for valid results
* Global Coordinate (X,Y,Z): 
	* Y is the vertical axis (Elevation)
	* X and Z are the axes within the horizontal plane (Plan)
* Local Coordinate (x,y,z):
	* x-axis = longitudinal axis
	* z-axis = major bending axis
	* y-axis = minor bending axis (along depth of element)
* The default member orientation (web direction vector) follows the following algorithm.
	1. Local x-axis is defined by the element's i and j node
	2. If +x is aligned with Y (vertical member), then +y is aligned with -X (i.e. <-1,0,0>)
	3. In other cases, +y will always have positive component with +Y
	4. After defining x and y axis, local z-axis is defined to be consistent with right-hand rule
	5. after initial assignment, the element may be rotated by user-specified beta angle.



<br>


# References 
* Kassimali, A. (2012). Matrix analysis of structures. USA: Cengage Learning. 
* McGuire, W., Gallagher, R., Ziemian, R. (2014). Matrix Structural Analysis. Second Edition. 

My course note in CEE 280 Advanced Structural Analysis, and CEE 282 Nonlinear Structural Analysis, both of which were taught by Dr. Greg Deierlein at Stanford, were also tremendously helpful. The object-oriented design of MSApy; with 3 distinct classes (nodes, element, and structure) is also entirely based on the course programming project. MSApy started off as a simple conversion of my CEE 280 project from MATLAB to Python. Later on, many additional features and refinements were added; most notably the visualization options via Plotly.

Other resources that I referred to:
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



# Feedbacks

Thank you for using MSApy. Given the nature of our profession, I feel it is necessary to emphasize on the above disclaimer. Please note MSApy is a one-man project and can in no way, shape or form compete with the rigor of the debugging process at commercial software companies. In using the program, the user accepts and understands that no warranty is expressed or implied by the developers on the accuracy or the reliability of the program. The user must explicitly understand the assumptions of the program and must independently verify the results.

Please forward your suggestions, feedbacks, criticisms, etc here on Github or send me an email at rwang01 [/\\] stanford edu. Thank you!



Robert

Last Updated: 2020-12-25

