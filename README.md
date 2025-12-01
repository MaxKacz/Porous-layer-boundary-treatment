Porous layer boundary treatment
=========

This repository contains the OpenFOAM numerical setups, libraries and python post-processing codes used in ADD LINK HERE. 


What is this repository for?
----------------------------

* Supported OpenFoam Versions : v2412plus to latest
* Supported Python Versions : >= 3.13


Installation of the roughWallFunctions library
-----------------------

```bash
cd $FOAM_RUN
git clone git@github.com:MaxKacz/Porous-layer-boundary-treatment.git
cp -rf Porous-layer-boundary-treatment/wallFunctions $WM_PROJECT_USER_DIR/
cd $WM_PROJECT_USER_DIR/wallFunctions/
./Allwclean
./Allwmake
```

Usage
---------------

swak4foam is required for running the numerical cases involving a porous layer.

The 'roughWallFunctions' library, which contains the rough wall boundary condition introduced by Wilcox [2006] and revised by Fuhrman et al. [2010], is located in the /wallFunctions folder. This library is used as a reference in this work. See the section above for installation instructions.

The 'Fuhrman 2010' folder contains the numerical setups for simulating a fully developed boundary layer over a rough surface (see section 3 of the article).   
The '/Kikkert2012' folder contains the numerical setups for simulating the bore-driven swash experiments of Kikkert et al. [2012] in 2D and 3D (see section 4 of the article).   
The '/python' folder contains the post-processing codes used to generate the figures presented in the article.


Bibliography
---------------
Fuhrman et al. [2010] : Fuhrman, D., M. Dixen, and N. Jacobsen, Physically-consistent wall boundary conditions for the k−ω turbulence model, Journal of Hydraulic Research, 48, 793–800, 2010.   
Kikkert et al. [2012] : Kikkert, G., T. O’Donoghue, D. Pokrajac, and N. Dodd, Experimental study of bore-driven swash hydrodynamics onimpermeable rough slopes, Coastal Engineering, 60, 2012.   
Wilcox [2006] : Wilcox, D., Turbulence modeling for cfd, third edition, DCW Industries, 2006.   


