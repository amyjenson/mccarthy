mccarthy
========

Copyright 2010--2024  Ed Bueler

This repository contains slides, notes, and computer programs for numerical glacier and ice sheet modeling.  These materials have been used for the [International Summer School in Glaciology](http://glaciers.gi.alaska.edu/courses/summerschool), McCarthy, AK in years 2010, 2012, 2014, 2016, 2018, 2022, and 2024.

Download all materials by one of these methods:

  * use the [releases page](https://github.com/bueler/mccarthy/releases) for a `.zip`/`.tar.gz` archive
  * clone or shallow clone this repository:

    git clone --depth=1 https://github.com/bueler/mccarthy.git

The optional `depth=1` setting reduces the download size by not getting the history, which most users will not need.

slides and notes
----------------

The PDF slides [slides/slides-2024.pdf](slides/slides-2024.pdf) are new this year.  I believe they cover the essential material in a better way.  They are supported by Python codes in [py/](py/).

The older material is in PDF notes [notes/notes-2024.pdf](notes/notes-2024.pdf), previous slides [slides/slides-2022.pdf](slides/slides-2022.pdf), and Matlab/Octave programs [mfiles/](mfiles/).  My plan is to update the notes soon!

Python programs
---------------

The codes in subdirectory [py/](py/) solve surface kinematic equation (SKE) and shallow ice approximation (SIA) problems.  Python programs [surface1d.py](py/surface1d.py) and [shallowuw.py](py/shallowuw.py) use only the numpy and matplotlib libraries.  You are encouraged to actually run and modify them!  Feel free to email me about how they work.

For now the [slides (slides-2024.pdf)](slides/slides-2024.pdf) are the documentation, along with line comments in the `.py` source codes.  Please report any bugs, either by email or by using the [issues](https://github.com/bueler/mccarthy/issues) for this repository.

The [surface2d.py](py/surface2d.py) program uses Firedrake.  Additional Firedrake codes which solve the glaciological (Glen-law) Stokes model are in the [stokes/](stokes/) directory.  See [stokes/doc.pdf](stokes/doc.pdf) for some documentation.

Matlab/Octave programs (deprecated)
-----------------------------------

The codes in subdirectory [mfiles/](mfiles/) solve SIA and SSA problems.  They should work in either Matlab or [Octave](https://www.gnu.org/software/octave/); if not please report a bug, either by email or by using the [issues](https://github.com/bueler/mccarthy/issues) for this repository.  The [current notes](notes/notes-2024.pdf) and the [older slides](slides/slides-2022.pdf) document these programs, but the programs also have help files (leading comments).

Stokes solver
-------------

The Python tools in [stokes/](stokes/) are primarily for projects.  By default they solve a 2D Glen-law Stokes flow over a bedrock step.  The workflow uses the following tools: [Firedrake](https://www.firedrakeproject.org/) (a finite element library), [Gmsh](http://gmsh.info/) (a mesh generator), [PETSc](http://www.mcs.anl.gov/petsc/) (a solver library), and [Paraview](https://www.paraview.org/) (for visualization).  See [stokes/doc.pdf](stokes/doc.pdf) for more information.

ancient versions
----------------

Older versions (2009, 2010, 2012, 2014) of this material lived in the repo https://github.com/bueler/karthaus
