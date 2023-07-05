# FEniCSx_polymers
This is simple common usage API for FEniCSx - finite element analysis framework for python. There is example of complex physical study of photopolymerization here. See for another information:
https://github.com/EnglishBeach/COMSOL_polymers. This study about physical aspects of diffusion processes in solution. I use time-depended solver and solve it on 1D interval. 

# Structure
* fenics - API forlder:
  * operators - functions, different operators from ufl and nessesary clases for set up study
  * express - constructors and tools for simple build and configure common studies
* Solver.ipynb - solver for diffusion processes of photopolymerization
* parametrs - parametrs for 1D and another tasks with boundary conditious and so on

# Usage
Because this project depends on FEniCsx: https://github.com/FEniCS/dolfinx. You neen usesimilar instructions for use thia api. Easiest way is Docker images:
```shell
docker run -ti dolfinx/dolfinx:stable
```
For connecting:
```shell
docker run --init -ti -p 8888:8888 dolfinx/lab:stable  # Access at http://localhost:8888
```

# Changes:
* corrected complicated parametrs set up
* unify high and low level variables realisations: ufl.Expressions, fem.Functions, fem.Consstants and pythonic functions, floats
* new tools for efficient solving and viewing results
