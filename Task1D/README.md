# FEniCSx for polymers
FEniCSx for polymers is simple API for FEniCSx - finite element analysis framework for python. See https://github.com/FEniCS/dolfinx. Fenics package is facade for low level dolfinx lib and contains high level functions for quick set up general structures for FEniCSx solvers. I created docker image with all necessary dependencies. 

This is my university course work of mathematical modeling of diffusion processes in multicomponent photopolymerization system. See for another information about chemical aspects: https://github.com/EnglishBeach/COMSOL_polymers. 

# Structure
* fenics - API forlder:
  * operators - functions, different operators from ufl and nessesary clases for set up study
  * express - constructors and tools for simple build and configure common studies
  * distributions - module collecting different form distributions
* Solver.ipynb - solver for diffusion processes of photopolymerization
* parametrs - parametrs for 1D and another tasks with boundary conditions and so on

# Usage
Easiest way is Docker images:
```shell
docker run -ti englishbeach/fenics:v0.1
```
For network:
```shell
docker run --init -ti -p 8888:8888 englishbeach/fenics:v0.1  # Access at http://localhost:8888
```

# Changes:
* corrected complicated parametrs set up
* unify high and low level variables realisations: ufl.Expressions, fem.Functions, fem.Consstants and pythonic functions, floats
* new tools for efficient solving and viewing results

# Plans:
* Fix database for recording results
* add setup for fenics package
* add 2D support and relevant functions
* add more API elements for another FEniCSx features
* fix some bad errors in dolfinx functions (infite circle trying compile ufl form in function)
