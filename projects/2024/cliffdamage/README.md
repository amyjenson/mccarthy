# Cliffs, overhangs, damage

## reassurance

The point of the project is not to reach a particular destination.  It is to _help you learn fast from wherever you currently are_.  So please be honest with yourself about what you don't know, and please ask about it!  Another student may know more programming than you, or have seen more differential equations, or have more glaciers background, or better data manipulation skills, or whatever.  That is irrelevant to our goals.  We want to help you move forward as far as possible in one intense week.  Forward progress will be from things _you_ understand into things _you_ do not yet understand.

## description

PROJECT 14: Cliffs, overhangs, damage

ADVISOR: Ed Bueler

DESCRIPTION:  At the surface of a glacier, and especially on steep margins, our viscous-fluid understanding of glaciers can break down.  Near cliffs and overhangs, stresses within the ice can turn into fractures and crevasses.  A numerical Stokes model can address where fractures appear, via a model of damage, that is, a model of how stresses cause the deterioration of polycrystalline structure.  We will try to model the initial damage, starting from some of the relevant literature, using an already-written finite-element solver of the 2D (planar) Glen-Stokes equations.  The solver will connect ice geometry and surface stresses to stresses within the ice, and these stresses can be evaluated as rates of change of damage.  By modifying the solver we will explore different geometries, evolution models, and questions as they arise.

SOFTWARE REQUIREMENTS: You will need a recent version of Python _running locally on your machine_.  Please try to build/install the following: Firedrake, Gmsh, Paraview.  (As backup, I'll bring an extra laptop, pre-loaded.)

STUDENT BACKGROUND: Linear algebra and some differential equations are required.  Optionally, perhaps a bit of numerical methods or the finite element method?

## getting started

As the first step, if you have not done it already, please clone my whole [McCarthy repo](https://github.com/bueler/mccarthy):

    $ git clone --depth=1 https://github.com/bueler/mccarthy.git

Now go to the `stokes/` directory and try all the steps documented in `stokes/README.md`.  You will learn how to use my Firedrake-based Python Stokes solver, with pre- and post-processing using Gmsh and Paraview.  Some of these "first step" steps may involve some learning, in which case the project has already been worthwhile!

As a second step, collect and browse the following documents.  I will provide these on paper in McCarthy, but they are also available electronically:

  * `stokes/doc.pdf`:  This documents how the Firedrake Stokes solver works.  It can be built in `doc/` using LaTeX.
  * `projects/2024/cliffdamage/PralongFunkLuthi2003.pdf`:  This paper contains some key ideas relating the stresses which are modeled in a Stokes solver to the evolution of damage.  We will start with a focus on formula (8), but you should read from the beginning.
