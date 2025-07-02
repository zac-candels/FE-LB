# FE-LB: Finite-Element-Lattice-Boltzmann

In this repo. we will use the finite element method to solve the velocity-discretized Boltzmann equation (aka Lattice Boltzmann equation).
Below we will present some of the background

## Governing Equations 

We start with the lattice Boltzmann equation

$$ \frac{\partial f_i}{\partial t} + \mathbf{c}_i \cdot \nabla f_i = J_i(f_1,...,f_Q) $$

where $J_i$ is the collision operator for the $i^{th}$ distribution function.

## File Naming Convention

(1) First part is time-step TS. Examples include forward-euler (FwE),
backward-euler (BE), Crank-Nicolson (CN) 

(2) Second component is how the nonlinear terms (ie collision and force)
are handled. This is either explicitly (using values at previous time step)
or implicitly, using some linearization procedure.

(3) Third component is stabilization. If not stabilization is used,
we have noStab. If we use least-squares stabilization, LSstab.

(4) Boundary conditions. Equilibrium boundary conditions will be
denoted EqBC, bounceback as BB.

A sample file name is 

TS_BE_CollForce_Explicit_Stab_LSstab_BC_BB.py

This tells us that (a) we are using a backward Euler timestep, 
that (b) the collision and forcing terms are handled explicitly,(c) we use
 least-squares stabilization, and (d), we are using bounceback
boundary conditions. Conversely, we could also have 

TS_CN_CollForce_Implicit_Stab_noStab_BC_EqBC.py

In this case, we use a Crank-Nicolson scheme, we treat collision 
and force terms implicitly, we have no stabilization and 
we use equilibrium boundary conditions.
