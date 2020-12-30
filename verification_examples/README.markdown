# Verification Problems

Excel input files and verification of various structural analysis problems. Many of these problems were taken from:

* Russell C. Hibbeler's Structural Analysis Textbook: [Amazon Link](https://www.amazon.com/Structural-Analysis-10th-Russell-Hibbeler/dp/0134610679/ref=sr_1_1?dchild=1&keywords=structural+analysis+hibbeler&qid=1608621933&s=books&sr=1-1)
* Aslam Kassimali's Matrix Analysis Textbook: [Amazon Link](https://www.amazon.com/Matrix-Analysis-Structures-Aslam-Kassimali/dp/1111426201/ref=sr_1_2?dchild=1&keywords=aslam+kassimali&qid=1608621894&s=books&sr=1-2)

[TOC]

## Problem 1: Simple Beam

The first problem will be a simple beam with a concentrated load at mid-span and an uniform load on half of the span. This problem was taken from the Hibbeler Structural Analysis textbook (2014). Since the structure is determinant, the forces do not depend on the section/material properties. Let's use the following assumptions:

* All units in [inches] and [kips]
* E = 29000 ksi
* Use section property of W18x40.
  * A = 11.8 in<sup>2</sup>
  * I<sub>z</sub> = 612 in<sup>4</sup>
  * I<sub>y</sub> = 19.1 in<sup>4</sup>
  * J = 0.810 in<sup>4</sup>
* Neglect self-weight
* Shear deformation included. For wide-flange sections, assume:
  * A<sub>y</sub> = A<sub>web</sub>
  * A<sub>z</sub> = 5/6 * A<sub>flanges</sub>

![](img/P1.png)

| Results        | RISA and MASTAN        | MSApy Output           |
| -------------- | ---------------------- | ---------------------- |
| Reaction       | A: 27 kips, B: 15 kips | A: 27 kips, B: 15 kips |
| Deflection     | - 0.96 inches          | - 0.96 inches          |
| Maximum Shear  | V: 27 kips             | V: 27 kips             |
| Maximum Moment | M: 180 kip.ft          | M: 180 kip.ft          |

<img src="img/P1b.png" style="zoom:75%;" />

<img src="img/P1a.png" style="zoom:75%;" />



## Problem 2: Simple Truss

The second problem is a simple planar truss, again from the Hibbeler text. 

* All units in [N], and [mm]
* E = 200 000 MPa
* Use section property of a 10mm x 10 mm square solid section
  * A = 100 mm<sup>2</sup>
* Neglect self-weight

![](img/P2.png)

| Results           | RISA and MASTAN             | MSApy Output |
| ----------------- | --------------------------- | ------------ |
| Reactions         | A: 1000 N,   B: 1000 N      |              |
| Deflection at C   | dx: 0.24 mm   dy: -1.056 mm |              |
| Axial Force in BC | 1400 N                      |              |
| Axial Force in HG | 1000 N                      |              |
| Axial Force in CG | 800 N                       |              |
| Axial Force in BG | 566 N                       |              |

<img src="img/p2b.png" style="zoom:75%;" />

## Problem 3: Simple Frame

A simple single-bay portal frame fixed base. 

* Units in [kips], [in]
* E = 29 000 ksi
* Consider self-weight, and apply a self-weight modifier of 0.5 to the right
* Consider shear deformation. For wide-flange sections, assume:
  * A<sub>y</sub> = A<sub>web</sub>
  * A<sub>z</sub> = 5/6 * A<sub>flanges</sub>

<img src="img/P3.png" style="zoom:50%;" />

| Results                             | RISA and MASTAN                                  | MSApy Output                                     |
| ----------------------------------- | ------------------------------------------------ | ------------------------------------------------ |
| Left Reaction                       | Fx: 32.6 kips \| Fy: 26 kips \| Mz: -1650 kip.in | Fx: 32.6 kips \| Fy: 26 kips \| Mz: -1650 kip.in |
| Right Reaction                      | Fx: -46.6 kips \| Fy: 44 kips \| Mz: 2461 kip.in | Fx: -46.6 kips \| Fy: 44 kips \| Mz: 2461 kip.in |
| Deflection at Roof Peak             | dx: 0.031 in \| dy: -0.7544 in                   | dx: 0.031 in \| dy: -0.7544 in                   |
| Lateral Deflection at Right Column  | dx: 0.166 in                                     | dx: 0.166 in                                     |
| Max Shear and Moment (Left Column)  | V: 32.6 kips \| M: 2263 kip.ft                   | V: 32.6 kips \| M: 2263 kip.ft                   |
| Max Shear and Moment (Right Column) | V: 46.6 kips \| M: 3131 kip.ft                   | V: 46.6 kips \| M: 3131 kip.ft                   |
| Max Shear and Moment (Left Beam)    | V: 19.09 kips \| M: 2410 kip.ft                  | V: 19.09 kips \| M: 2410 kip.ft                  |
| Max Shear and Moment (Right Beam)   | V: 37.94 kips \| M: 3131 kip.ft                  | V: 37.94 kips \| M: 3131 kip.ft                  |

<img src="img/p3b.png" style="zoom:75%;" />

<img src="img/p3a.png" style="zoom:75%;" />

## Problem 4: Simple Frame w/ Self-Weight Modifiers





## Problem 5: Simple Frame w/ Moment Release

The case of frame element and truss element were proved to be working in the previous examples. In this problem, we want to test the implementation of pinned-fixed elements (i.e. element with a moment release on one end).

* Units in [kips], [in]
* Neglect self-weight and shear deformation

<img src="img/P5.png" style="zoom:50%;" />





## Problem 6: 3-D Truss



## Problem 6: 3-D Frame



## Problem 7: Moment Frame



## Problem 8: Support Settlement



## Problem 9: Ill-conditioned Structure

Two structures will be tested here. One is ill-conditioned (weird stiffness distribution), and the other is not stable. The ideal outcome would be for MSApy to catch the ill-conditioning and provide an error.

<img src="img/P9.png" style="zoom:50%;" />

In the first case, an error exception was added in the code to catch ill-conditioned structures:

<img src="img/P9a.png" style="zoom:50%;" />

The case of the unstable structure was also caught in the same vein as above.



## Problem 10: Simple Beam Rotated to Minor Axis Bending



## Problem 11: Overhang Beam Placed Diagonally









