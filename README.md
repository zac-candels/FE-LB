# FE-LB: Finite-Element-Lattice-Boltzmann

In this repo. we will use the finite element method to solve the velocity-discretized Boltzmann equation (aka Lattice Boltzmann equation).
Below we will present some of the background

## Governing Equations 

We start with the lattice Boltzmann equation

$$ \frac{\partial f_i}{\partial t} + \mathbf{c}_i \cdot \nabla f_i = J_i(f_1,...,f_Q) $$

where $J_i$ is the collision operator for the $i^{th}$ distribution function.

## File Naming Convention

(1) First part is time-step TS. Examples include forward-euler (FWE),
backward-euler (BE), Crank-Nicolson (CN) 

(2) Second component is how the nonlinear terms (ie collision and force)
are handled. This is either explicitly (using values at previous time step)
or implicitly, using some linearization procedure.

(3) Third component is stabilization. If not stabilization is used,
we have noStab. If we use least-squares stabilization, LSstab.
