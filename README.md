## Lid-Driven Cavity Flow Solver -- Python

A computational fluid dynamics (CFD) solver for the classic lid-driven cavity flow problem, implemented in Python. This solver uses the SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm to solve the incompressible Navier-Stokes equations on a staggered grid.

### Features

- Multiple Discretization Schemes: Implements and compares Central Difference Scheme (CDS), Upwind, Hybrid, and QUICK schemes
- Steady and Unsteady Simulations: Supports both steady-state and time-dependent simulations
- Comprehensive Visualization
- Velocity vector plots
- Streamline visualization
- Velocity contour plots (u, v, and magnitude)
- Velocity profiles with comparison to benchmark data
- Convergence history plots
- Animation Capabilities: Creates GIF animations of velocity fields and contours over time
- Benchmark Comparison: Compares results with Ghia et al. (1982) benchmark data
- Performance Metrics: Tracks solution time and convergence rates for different schemes

### Prerequisites
- Python 3.7+
- NumPy
- Matplotlib
- Pillow (for animations)

### Problem parameters
length = 0.4  # Domain size (m)
nx = ny = 60  # Grid resolution
rho = 1.0     # Density (kg/m³)
nu = 0.004    # Kinematic viscosity (m²/s)
u_lid = 1.0   # Lid velocity (m/s)

### Simulation mode
unsteady = True  # Set to False for steady simulation
dt = 0.1         # Time step size (s)
total_time = 2.0 # Total simulation time (s)
create_animations = True
animation_dir = "animations"

### Problem
The lid-driven cavity is a benchmark problem in CFD where fluid is contained in a square cavity with the top wall (lid) moving at a constant velocity while all other walls remain stationary. This creates a recirculating flow with complex features including:

- Primary vortex in the center
- Secondary vortices in the corners
- Boundary layers along the walls

The Reynolds number (Re = u_lid × length / ν) determines the flow characteristics

- Low Re (< 1000): Laminar flow with simple vortex structure
- High Re (> 1000): More complex flow patterns with multiple vortices


## Numerical Methods

### SIMPLE Algorithm

The solver implements the SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm:

1. Set boundary conditions
2. Solve momentum equations to get intermediate velocities
3. Solve pressure correction equation
4. Correct pressure and velocities
5. Repeat until convergence


### Discretization Schemes

The solver implements four discretization schemes for the convective terms:

1. **Central Difference Scheme (CDS)**: Second-order accurate but can be unstable at high Reynolds numbers
2. **Upwind Scheme**: First-order accurate, very stable but introduces numerical diffusion
3. **Hybrid Scheme**: Combines CDS and Upwind based on local Peclet number
4. **QUICK Scheme**: Third-order accurate, provides better accuracy with reasonable stability


## Results and Visualization

### Velocity Profiles

The solver compares velocity profiles along the geometric center lines with benchmark data from Ghia et al. (1982).

### Convergence History

Convergence history plots show the residual reduction over iterations for each scheme.

### Velocity Contours

Velocity contour plots show the distribution of u-velocity, v-velocity, and velocity magnitude throughout the domain.

### Animations

For unsteady simulations, the solver creates animations showing the evolution of the flow field over time.

## Convergence Problems

If the solution doesn't converge:

1. Adjust relaxation factors (`alpha_u`, `alpha_v`, `alpha_p`)
2. Increase the maximum number of iterations
3. Use a more stable scheme (e.g., Upwind instead of CDS)
4. Refine the grid (increase `nx` and `ny`)


## Future Work

- Implement higher-order schemes (e.g., MUSCL, WENO)
- Add adaptive mesh refinement
- Extend to 3D simulations
- Implement turbulence models
- Add parallel processing capabilities
- Develop a GUI for parameter input and visualization


## References

1. Ghia, U., Ghia, K.N., Shin, C.T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. Journal of Computational Physics, 48, 387-411.
2. Versteeg, H.K., Malalasekera, W. (2007). An Introduction to Computational Fluid Dynamics: The Finite Volume Method. Pearson Education.
3. Patankar, S.V. (1980). Numerical Heat Transfer and Fluid Flow. Hemisphere Publishing Corporation.
4. Ferziger, J.H., Perić, M. (2002). Computational Methods for Fluid Dynamics. Springer.

## Acknowledgments

- The benchmark data from Ghia et al. (1982) is widely used in the CFD community
- The SIMPLE algorithm was originally developed by Patankar and Spalding
- The staggered grid approach helps avoid the checkerboard pressure field issue
