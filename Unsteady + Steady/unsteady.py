import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, PillowWriter
import time
import os
import sys
import traceback

# Set a vibrant colormap for better visualization
COLORMAP = 'plasma'  # More vibrant than viridis

# Global dictionary to store solver data for each scheme
solvers_data = {}
solution_times = {}

# Ghia et al. benchmark data for different Reynolds numbers
def get_ghia_data(re=100, length=0.4):
    # Scale factor to convert from 1x1 cavity to our domain size
    scale = length
    
    # For Re = 100, 400, and 1000
    # y-coordinate along vertical centerline (same for all Re)
    y_ghia_orig = np.array([0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0])
    
    # x-coordinate along horizontal centerline (same for all Re)
    x_ghia_orig = np.array([0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0])
    
    # Scale coordinates to our domain size
    y_ghia = y_ghia_orig * scale
    x_ghia = x_ghia_orig * scale
    
    # Data for Re = 100
    if re <= 100:
        u_ghia = np.array([0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.0])
        v_ghia = np.array([0.0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.0])
        re_label = 100
    
    # Data for Re = 400
    elif re <= 400:
        u_ghia = np.array([0.0, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 0.55892, 0.61756, 0.68439, 0.75837, 1.0])
        v_ghia = np.array([0.0, 0.18109, 0.19791, 0.21090, 0.22965, 0.23176, 0.13612, 0.00332, -0.11477, -0.27805, -0.22600, -0.16304, -0.10648, -0.09077, -0.07456, -0.05803, 0.0])
        re_label = 400
    
    # Data for Re = 1000
    else:
        u_ghia = np.array([0.0, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27533, -0.10150, -0.06080, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117, 0.57492, 0.65928, 1.0])
        v_ghia = np.array([0.0, 0.27485, 0.29012, 0.30174, 0.30533, 0.28124, 0.22965, 0.16391, 0.02526, -0.23827, -0.44993, -0.38598, -0.33050, -0.32393, -0.32235, -0.28643, 0.0])
        re_label = 1000
    
    return {
        'y_ghia': y_ghia,
        'u_ghia': u_ghia,
        'x_ghia': x_ghia,
        'v_ghia': v_ghia,
        're': re_label
    }

def init_solver(length=0.4, nx=80, ny=80, rho=1.0, nu=0.004, u_lid=1.0, 
           scheme="CDS", alpha_u=0.5, alpha_v=0.5, alpha_p=0.3,
           unsteady=False, dt=0.001, total_time=10.0):
    # Create a dictionary to store all solver data
    data = {}

    # Basic parameters
    data['length'], data['nx'], data['ny'] = length, nx, ny
    data['dx'], data['dy'] = length/nx, length/ny
    data['rho'], data['nu'], data['mu'] = rho, nu, rho*nu
    data['u_lid'], data['scheme'] = u_lid, scheme
    data['Re'] = u_lid * length / nu
    data['alpha_u'], data['alpha_v'], data['alpha_p'] = alpha_u, alpha_v, alpha_p
    data['convergence_criterion'] = 1e-5
    
    # Unsteady simulation parameters
    data['unsteady'] = unsteady
    data['dt'] = dt
    data['total_time'] = total_time
    data['current_time'] = 0.0
    
    # Solution fields (staggered grid)
    data['u'] = np.zeros((nx+1, ny+2))  # u-velocity at (i+1/2, j)
    data['v'] = np.zeros((nx+2, ny+1))  # v-velocity at (i, j+1/2)
    data['p'] = np.zeros((nx+2, ny+2))  # pressure at cell centers
    data['p_corr'] = np.zeros((nx+2, ny+2))
    
    # Previous time step fields for unsteady simulation
    data['u_old'] = np.zeros_like(data['u'])
    data['v_old'] = np.zeros_like(data['v'])

    # Coefficients for momentum equations
    data['ae_u'], data['aw_u'], data['an_u'], data['as_u'] = [np.zeros((nx+1, ny)) for _ in range(4)]
    data['ap_u'], data['b_u'] = np.zeros((nx+1, ny)), np.zeros((nx+1, ny))

    data['ae_v'], data['aw_v'], data['an_v'], data['as_v'] = [np.zeros((nx, ny+1)) for _ in range(4)]
    data['ap_v'], data['b_v'] = np.zeros((nx, ny+1)), np.zeros((nx, ny+1))

    # Coefficients for pressure correction
    data['ae_p'], data['aw_p'], data['an_p'], data['as_p'] = [np.zeros((nx, ny)) for _ in range(4)]
    data['ap_p'], data['b_p'] = np.zeros((nx, ny)), np.zeros((nx, ny))

    # Grid for plotting
    data['x_cell'] = np.linspace(data['dx']/2, length-data['dx']/2, nx)
    data['y_cell'] = np.linspace(data['dy']/2, length-data['dy']/2, ny)
    data['X'], data['Y'] = np.meshgrid(data['x_cell'], data['y_cell'])

    # Colors for plotting
    colors = {"CDS": "blue", "Upwind": "red", "Hybrid": "green", "QUICK": "purple"}
    data['color'] = colors.get(scheme, "blue")
    
    # For storing time history data in unsteady simulations
    if unsteady:
        data['time_history'] = []
        data['u_history'] = []
        data['v_history'] = []
        data['p_history'] = []
        data['vel_mag_history'] = []  # For animation

    print(f"Reynolds number: {data['Re']}, Using {scheme} scheme (α_u={alpha_u}, α_v={alpha_v}, α_p={alpha_p})")
    if unsteady:
        print(f"Unsteady simulation: dt={dt}, total_time={total_time}")
    else:
        print("Steady simulation")

    return data

# ============= BOUNDARY CONDITIONS =============

def set_boundary_conditions(data):
    # Extract needed variables
    u, v = data['u'], data['v']
    u_lid = data['u_lid']
    
    # Top wall (lid) - non-zero u
    u[:, -1] = 2*u_lid - u[:, -2]

    # Left, right, and bottom walls - no slip
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    v[0, :] = 0
    v[-1, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0

def calculate_coefficients(data):
    # Extract needed variables
    nx, ny = data['nx'], data['ny']
    dx, dy = data['dx'], data['dy']
    rho, mu = data['rho'], data['mu']
    scheme = data['scheme']
    unsteady = data['unsteady']
    dt = data['dt']

    u, v, p = data['u'], data['v'], data['p']
    ae_u, aw_u, an_u, as_u = data['ae_u'], data['aw_u'], data['an_u'], data['as_u']
    ap_u, b_u = data['ap_u'], data['b_u']
    ae_v, aw_v, an_v, as_v = data['ae_v'], data['aw_v'], data['an_v'], data['as_v']
    ap_v, b_v = data['ap_v'], data['b_v']
    
    # Reset coefficient arrays
    ae_u.fill(0.0)
    aw_u.fill(0.0)
    an_u.fill(0.0)
    as_u.fill(0.0)
    ap_u.fill(0.0)
    b_u.fill(0.0)
    
    ae_v.fill(0.0)
    aw_v.fill(0.0)
    an_v.fill(0.0)
    as_v.fill(0.0)
    ap_v.fill(0.0)
    b_v.fill(0.0)
    
    small = 1e-20  # Small number to avoid division by zero

    # u-momentum coefficients
    for i in range(1, nx):
        for j in range(1, ny+1):
            if j-1 >= ny: continue
            
            # Diffusion terms (same for all schemes)
            ae_u[i, j-1] = aw_u[i, j-1] = mu * dy / dx
            an_u[i, j-1] = as_u[i, j-1] = mu * dx / dy
            
            # Average velocities for convection
            u_e = 0.5 * (u[i+1, j] + u[i, j])
            u_w = 0.5 * (u[i, j] + u[i-1, j])
            v_n = 0.5 * (v[i, j] + v[i+1, j])
            v_s = 0.5 * (v[i, j-1] + v[i+1, j-1])
            
            # Apply scheme-specific convection terms
            if scheme == "CDS":
                ae_u[i, j-1] -= 0.5 * rho * u_e * dy
                aw_u[i, j-1] += 0.5 * rho * u_w * dy
                an_u[i, j-1] -= 0.5 * rho * v_n * dx
                as_u[i, j-1] += 0.5 * rho * v_s * dx
            elif scheme.lower() == "upwind":
                # East face
                if u_e > 0:
                    continue
                else: 
                    ae_u[i, j-1] -= rho * u_e * dy
                
                # West face
                if u_w > 0:
                    aw_u[i, j-1] += rho * u_w * dy
                else:
                    continue
                
                # North face
                if v_n > 0:
                    continue
                else:
                    an_u[i, j-1] -= rho * v_n * dx
                    
                # South face
                if v_s > 0:
                    as_u[i, j-1] += rho * v_s * dx
                else:
                    continue
            elif scheme == "Hybrid":
                # Calculate Peclet numbers
                Pe_e = rho * u_e * dx / mu
                Pe_w = rho * u_w * dx / mu
                Pe_n = rho * v_n * dy / mu
                Pe_s = rho * v_s * dy / mu
                
                # East face
                if abs(Pe_e) < 2: 
                    ae_u[i, j-1] -= 0.5 * rho * u_e * dy
                else:
                    if u_e > 0: 
                        ap_u[i, j-1] += rho * u_e * dy
                    else: 
                        ae_u[i, j-1] -= rho * u_e * dy
                
                # West face
                if abs(Pe_w) < 2: 
                    aw_u[i, j-1] += 0.5 * rho * u_w * dy
                else:
                    if u_w > 0: 
                        aw_u[i, j-1] += rho * u_w * dy
                    else: 
                        ap_u[i, j-1] -= rho * u_w * dy
                
                # North face
                if abs(Pe_n) < 2: 
                    an_u[i, j-1] -= 0.5 * rho * v_n * dx
                else:
                    if v_n > 0: 
                        ap_u[i, j-1] += rho * v_n * dx
                    else: 
                        an_u[i, j-1] -= rho * v_n * dx
                
                # South face
                if abs(Pe_s) < 2: 
                    as_u[i, j-1] += 0.5 * rho * v_s * dx
                else:
                    if v_s > 0: 
                        as_u[i, j-1] += rho * v_s * dx
                    else: 
                        ap_u[i, j-1] -= rho * v_s * dx
            elif scheme == "QUICK":
                # QUICK scheme implementation for u-momentum - FIXED
                # East face
                if u_e >= 0:
                    # Use upwind for cells near boundaries
                    if i+1 >= nx or i-1 < 0:
                        ap_u[i, j-1] += rho * u_e * dy
                    else:
                        # QUICK formula: 3/8 downstream + 6/8 central - 1/8 upstream
                        # For positive flow: downstream = i+1, central = i, upstream = i-1
                        ae_u[i, j-1] += rho * u_e * dy * (3/8)  # Downstream coefficient
                        ap_u[i, j-1] += rho * u_e * dy * (6/8)  # Central coefficient
                        aw_u[i, j-1] -= rho * u_e * dy * (1/8)  # Upstream coefficient
                else:  # u_e < 0
                    # Use upwind for cells near boundaries
                    if i+2 >= nx+1 or i < 0:
                        ae_u[i, j-1] -= rho * u_e * dy
                    else:
                        # QUICK formula for negative velocity
                        # For negative flow: downstream = i, central = i+1, upstream = i+2
                        ap_u[i, j-1] -= rho * u_e * dy * (3/8)  # Downstream coefficient
                        ae_u[i, j-1] -= rho * u_e * dy * (6/8)  # Central coefficient
                        # This would access u[i+2, j], need to ensure it's valid
                        if i+2 < nx+1:
                            # Add coefficient for the point two cells downstream
                            # We need to handle this specially since it's outside our standard stencil
                            # For now, we'll add it to the east coefficient
                            ae_u[i, j-1] -= rho * u_e * dy * (1/8)  # Upstream coefficient
                
                # West face (similar approach)
                if u_w >= 0:
                    # Use upwind for cells near boundaries
                    if i-2 < 0 or i >= nx:
                        aw_u[i, j-1] += rho * u_w * dy
                    else:
                        # QUICK formula
                        # For positive flow: downstream = i-1, central = i, upstream = i+1
                        aw_u[i, j-1] += rho * u_w * dy * (3/8)  # Downstream coefficient
                        ap_u[i, j-1] += rho * u_w * dy * (6/8)  # Central coefficient
                        ae_u[i, j-1] -= rho * u_w * dy * (1/8)  # Upstream coefficient
                else:  # u_w < 0
                    # Use upwind for cells near boundaries
                    if i-1 < 0 or i+1 >= nx:
                        ap_u[i, j-1] -= rho * u_w * dy
                    else:
                        # QUICK formula for negative velocity
                        # For negative flow: downstream = i, central = i-1, upstream = i-2
                        ap_u[i, j-1] -= rho * u_w * dy * (3/8)  # Downstream coefficient
                        aw_u[i, j-1] -= rho * u_w * dy * (6/8)  # Central coefficient
                        # This would access u[i-2, j], need to ensure it's valid
                        if i-2 >= 0:
                            # Add coefficient for the point two cells upstream
                            aw_u[i, j-1] -= rho * u_w * dy * (1/8)  # Upstream coefficient
                
                # North and South faces - use upwind for simplicity
                if v_n >= 0: 
                    ap_u[i, j-1] += rho * v_n * dx
                else: 
                    an_u[i, j-1] -= rho * v_n * dx
                
                if v_s >= 0: 
                    as_u[i, j-1] += rho * v_s * dx
                else: 
                    ap_u[i, j-1] -= rho * v_s * dx
            
            # Ensure coefficients are positive for diagonal dominance
            ae_u[i, j-1] = max(ae_u[i, j-1], 0.0)
            aw_u[i, j-1] = max(aw_u[i, j-1], 0.0)
            an_u[i, j-1] = max(an_u[i, j-1], 0.0)
            as_u[i, j-1] = max(as_u[i, j-1], 0.0)
            
            # Add unsteady term if needed
            if unsteady:
                # Implicit time discretization: ρ*u/dt
                ap_u[i, j-1] = rho * dx * dy / dt + ae_u[i, j-1] + aw_u[i, j-1] + an_u[i, j-1] + as_u[i, j-1] + rho*u_e*dy/2 - rho*u_w*dy/2 + rho*v_n*dx/2 - rho*v_s*dx/2 
                # Add source term from previous time step
                b_u[i, j-1] = rho * dx * dy * data['u_old'][i, j] / dt + (p[i, j] - p[i+1, j]) * dy
            else:
                ap_u[i, j-1] = ae_u[i, j-1] + aw_u[i, j-1] + an_u[i, j-1] + as_u[i, j-1] + rho*u_e*dy/2 - rho*u_w*dy/2 + rho*v_n*dx/2 - rho*v_s*dx/2 
                ap_u[i, j-1] = max(ap_u[i, j-1], small)
                b_u[i, j-1] += (p[i, j] - p[i+1, j]) * dy
                
            
            # Assemble coefficient matrix and source term
            

    # v-momentum coefficients (similar approach as u-momentum)
    for i in range(1, nx+1):
        for j in range(1, ny):
            if i-1 >= nx: continue
            
            # Diffusion terms
            ae_v[i-1, j] = aw_v[i-1, j] = mu * dy / dx
            an_v[i-1, j] = as_v[i-1, j] = mu * dx / dy
            
            # Average velocities
            u_e = 0.5 * (u[i, j] + u[i, j+1])
            u_w = 0.5 * (u[i-1, j] + u[i-1, j+1])
            v_n = 0.5 * (v[i, j+1] + v[i, j])
            v_s = 0.5 * (v[i, j] + v[i, j-1])
            
            # Apply scheme-specific convection terms (similar to u-momentum)
            if scheme == "CDS":
                ae_v[i-1, j] -= 0.5 * rho * u_e * dy
                aw_v[i-1, j] += 0.5 * rho * u_w * dy
                an_v[i-1, j] -= 0.5 * rho * v_n * dx
                as_v[i-1, j] += 0.5 * rho * v_s * dx
            elif scheme.lower() == "upwind":
                if u_e > 0:
                    continue
                else:
                    ae_v[i-1, j] -= rho * u_e * dy
                
                if u_w > 0:
                    aw_v[i-1, j] += rho * u_w * dy
                else:
                    continue
                
                if v_n > 0:
                    continue
                else:
                    an_v[i-1, j] -= rho * v_n * dx
                
                if v_s > 0:
                    as_v[i-1, j] += rho * v_s * dx
                else:
                    continue
            elif scheme == "Hybrid":
                Pe_e = rho * u_e * dx / mu
                Pe_w = rho * u_w * dx / mu
                Pe_n = rho * v_n * dy / mu
                Pe_s = rho * v_s * dy / mu
                
                if abs(Pe_e) < 2: 
                    ae_v[i-1, j] -= 0.5 * rho * u_e * dy
                else:
                    if u_e > 0: 
                        ap_v[i-1, j] += rho * u_e * dy
                    else: 
                        ae_v[i-1, j] -= rho * u_e * dy
                
                if abs(Pe_w) < 2: 
                    aw_v[i-1, j] += 0.5 * rho * u_w * dy
                else:
                    if u_w > 0: 
                        aw_v[i-1, j] += rho * u_w * dy
                    else: 
                        ap_v[i-1, j] -= rho * u_w * dy
                
                if abs(Pe_n) < 2: 
                    an_v[i-1, j] -= 0.5 * rho * v_n * dx
                else:
                    if v_n > 0: 
                        ap_v[i-1, j] += rho * v_n * dx
                    else: 
                        an_v[i-1, j] -= rho * v_n * dx
                
                if abs(Pe_s) < 2: 
                    as_v[i-1, j] += 0.5 * rho * v_s * dx
                else:
                    if v_s > 0: 
                        as_v[i-1, j] += rho * v_s * dx
                    else: 
                        ap_v[i-1, j] -= rho * v_s * dx
            elif scheme == "QUICK":
                # QUICK scheme implementation for v-momentum - FIXED
                # East and West faces - use upwind for simplicity
                if u_e >= 0: 
                    ap_v[i-1, j] += rho * u_e * dy
                else: 
                    ae_v[i-1, j] -= rho * u_e * dy
                
                if u_w >= 0: 
                    aw_v[i-1, j] += rho * u_w * dy
                else: 
                    ap_v[i-1, j] -= rho * u_w * dy
                
                # North face
                if v_n >= 0:
                    # Use upwind for cells near boundaries
                    if j+1 >= ny or j-1 < 0:
                        ap_v[i-1, j] += rho * v_n * dx
                    else:
                        # QUICK formula
                        # For positive flow: downstream = j+1, central = j, upstream = j-1
                        an_v[i-1, j] += rho * v_n * dx * (3/8)  # Downstream coefficient
                        ap_v[i-1, j] += rho * v_n * dx * (6/8)  # Central coefficient
                        as_v[i-1, j] -= rho * v_n * dx * (1/8)  # Upstream coefficient
                else:  # v_n < 0
                    # Use upwind for cells near boundaries
                    if j+2 >= ny+1 or j < 0:
                        an_v[i-1, j] -= rho * v_n * dx
                    else:
                        # QUICK formula for negative velocity
                        # For negative flow: downstream = j, central = j+1, upstream = j+2
                        ap_v[i-1, j] -= rho * v_n * dx * (3/8)  # Downstream coefficient
                        an_v[i-1, j] -= rho * v_n * dx * (6/8)  # Central coefficient
                        if j+2 < ny+1:
                            # Add coefficient for the point two cells downstream
                            an_v[i-1, j] -= rho * v_n * dx * (1/8)  # Upstream coefficient
                
                # South face (similar approach)
                if v_s >= 0:
                    # Use upwind for cells near boundaries
                    if j-2 < 0 or j >= ny:
                        as_v[i-1, j] += rho * v_s * dx
                    else:
                        # QUICK formula
                        # For positive flow: downstream = j-1, central = j, upstream = j+1
                        as_v[i-1, j] += rho * v_s * dx * (3/8)  # Downstream coefficient
                        ap_v[i-1, j] += rho * v_s * dx * (6/8)  # Central coefficient
                        an_v[i-1, j] -= rho * v_s * dx * (1/8)  # Upstream coefficient
                else:  # v_s < 0
                    # Use upwind for cells near boundaries
                    if j-1 < 0 or j+1 >= ny:
                        ap_v[i-1, j] -= rho * v_s * dx
                    else:
                        # QUICK formula for negative velocity
                        # For negative flow: downstream = j, central = j-1, upstream = j-2
                        ap_v[i-1, j] -= rho * v_s * dx * (3/8)  # Downstream coefficient
                        as_v[i-1, j] -= rho * v_s * dx * (6/8)  # Central coefficient
                        if j-2 >= 0:
                            # Add coefficient for the point two cells upstream
                            as_v[i-1, j] -= rho * v_s * dx * (1/8)  # Upstream coefficient
            
            # Ensure coefficients are positive
            ae_v[i-1, j] = max(ae_v[i-1, j], 0.0)
            aw_v[i-1, j] = max(aw_v[i-1, j], 0.0)
            an_v[i-1, j] = max(an_v[i-1, j], 0.0)
            as_v[i-1, j] = max(as_v[i-1, j], 0.0)
            
            # Add unsteady term if needed
            if unsteady:
                # Implicit time discretization: ρ*v/dt
                ap_v[i-1, j] =ae_v[i-1, j] + aw_v[i-1, j] + an_v[i-1, j] + as_v[i-1, j] + rho*u_e*dx/2 - rho*u_w*dx/2 + rho*v_n*dy/2 - rho*v_s*dy/2+ rho * dx * dy / dt
                # Add source term from previous time step
                b_v[i-1, j] =(p[i, j] - p[i, j+1]) * dx+  rho * dx * dy * data['v_old'][i, j] / dt
            else:
                ap_v[i-1, j] = ae_v[i-1, j] + aw_v[i-1, j] + an_v[i-1, j] + as_v[i-1, j] + rho*u_e*dx/2 - rho*u_w*dx/2 + rho*v_n*dy/2 - rho*v_s*dy/2
                ap_v[i-1, j] = max(ap_v[i-1, j], small)
                b_v[i-1, j] += (p[i, j] - p[i, j+1]) * dx
                
            
            # Assemble coefficient matrix and source term
            

def solve_momentum_equations(data):
    # Extract needed variables
    nx, ny = data['nx'], data['ny']
    alpha_u, alpha_v = data['alpha_u'], data['alpha_v']

    u, v = data['u'], data['v']
    ae_u, aw_u, an_u, as_u = data['ae_u'], data['aw_u'], data['an_u'], data['as_u']
    ap_u, b_u = data['ap_u'], data['b_u']
    ae_v, aw_v, an_v, as_v = data['ae_v'], data['aw_v'], data['an_v'], data['as_v']
    ap_v, b_v = data['ap_v'], data['b_v']

    # Solve u-momentum equation
    for _ in range(5):
        for i in range(1, nx):
            for j in range(1, ny+1):
                if j-1 >= ny: continue
                u_old = u[i, j]
                u_new = (ae_u[i, j-1] * u[i+1, j] + 
                        aw_u[i, j-1] * u[i-1, j] +
                        an_u[i, j-1] * u[i, j+1] +
                        as_u[i, j-1] * u[i, j-1] +
                        b_u[i, j-1]) / ap_u[i, j-1]
                u[i, j] = u_old + alpha_u * (u_new - u_old)

    # Solve v-momentum equation
    for _ in range(5):
        for i in range(1, nx+1):
            for j in range(1, ny):
                if i-1 >= nx: continue
                v_old = v[i, j]
                v_new = (ae_v[i-1, j] * v[i+1, j] + 
                        aw_v[i-1, j] * v[i-1, j] +
                        an_v[i-1, j] * v[i, j+1] +
                        as_v[i-1, j] * v[i, j-1] +
                        b_v[i-1, j]) / ap_v[i-1, j]
                v[i, j] = v_old + alpha_v * (v_new - v_old)

def calculate_pressure_correction(data):
    # Extract needed variables
    nx, ny = data['nx'], data['ny']
    dx, dy = data['dx'], data['dy']
    rho = data['rho']

    u, v = data['u'], data['v']
    p_corr = data['p_corr']
    ap_u, ap_v = data['ap_u'], data['ap_v']
    ae_p, aw_p, an_p, as_p = data['ae_p'], data['aw_p'], data['an_p'], data['as_p']
    ap_p, b_p = data['ap_p'], data['b_p']

    small = 1e-20

    # Reset pressure correction arrays
    ae_p.fill(0.0)
    aw_p.fill(0.0)
    an_p.fill(0.0)
    as_p.fill(0.0)
    ap_p.fill(0.0)
    b_p.fill(0.0)
    p_corr.fill(0.0)

    # Calculate pressure correction coefficients
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            if i-1 >= nx or j-1 >= ny: continue
            
            ae_p[i-1, j-1] = rho * dy**2 / max(ap_u[i, j-1], small) if i < nx else 0
            aw_p[i-1, j-1] = rho * dy**2 / max(ap_u[i-1, j-1], small) if i > 1 else 0
            an_p[i-1, j-1] = rho * dx**2 / max(ap_v[i-1, j], small) if j < ny else 0
            as_p[i-1, j-1] = rho * dx**2 / max(ap_v[i-1, j-1], small) if j > 1 else 0
            
            ap_p[i-1, j-1] = ae_p[i-1, j-1] + aw_p[i-1, j-1] + an_p[i-1, j-1] + as_p[i-1, j-1]
            
            # Source term (mass imbalance)
            b_p[i-1, j-1] = rho * ((u[i-1, j] - u[i, j]) * dy + (v[i, j-1] - v[i, j]) * dx)

    # Solve pressure correction equation
    for _ in range(50):
        for i in range(1, nx+1):
            for j in range(1, ny+1):
                if i-1 >= nx or j-1 >= ny: continue
                
                p_corr[i, j] = (
                    ae_p[i-1, j-1] * p_corr[i+1, j] +
                    aw_p[i-1, j-1] * p_corr[i-1, j] +
                    an_p[i-1, j-1] * p_corr[i, j+1] +
                    as_p[i-1, j-1] * p_corr[i, j-1] +
                    b_p[i-1, j-1]
                ) / (ap_p[i-1, j-1] + small)

def correct_pressure_and_velocity(data):
    # Extract needed variables
    nx, ny = data['nx'], data['ny']
    dx, dy = data['dx'], data['dy']
    alpha_p = data['alpha_p']

    u, v, p, p_corr = data['u'], data['v'], data['p'], data['p_corr']
    ae_p, an_p = data['ae_p'], data['an_p']

    # Correct pressure
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            p[i, j] += alpha_p * p_corr[i, j]

    # Correct u-velocity
    for i in range(1, nx):
        for j in range(1, ny+1):
            if j-1 >= ny: continue
            dp_corr = p_corr[i, j] - p_corr[i+1, j]
            u[i, j] += ae_p[i, j-1] * dp_corr / dy

    # Correct v-velocity
    for i in range(1, nx+1):
        for j in range(1, ny):
            if i-1 >= nx: continue
            dp_corr = p_corr[i, j] - p_corr[i, j+1]
            v[i, j] += an_p[i-1, j-1] * dp_corr / dx

def calculate_mass_residual(data):
    # Extract needed variables
    nx, ny = data['nx'], data['ny']
    b_p = data['b_p']

    residual = 0.0
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            if i-1 < nx and j-1 < ny:
                residual += abs(b_p[i-1, j-1])
    return residual

def store_previous_timestep(data):
    """Store current velocity fields for use in the next time step"""
    data['u_old'][:] = data['u']
    data['v_old'][:] = data['v']

def solve_timestep(data, max_iterations=5000):
    """Solve a single time step for unsteady simulation"""
    # Extract needed variables
    scheme = data['scheme']
    convergence_criterion = data['convergence_criterion']
    
    set_boundary_conditions(data)
    iteration, residual, residuals = 0, float('inf'), []
    
    while iteration < max_iterations and residual > convergence_criterion:
        calculate_coefficients(data)
        solve_momentum_equations(data)
        calculate_pressure_correction(data)
        correct_pressure_and_velocity(data)
        set_boundary_conditions(data)
        
        residual = calculate_mass_residual(data)
        residuals.append(residual)
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Residual: {residual:.8e}")
        
        iteration += 1
    
    return residuals

def solve(data, max_iterations=5000):
    """Solve the flow field (steady or unsteady)"""
    # Extract needed variables
    scheme = data['scheme']
    convergence_criterion = data['convergence_criterion']
    unsteady = data['unsteady']
    
    print(f"Starting {'unsteady' if unsteady else 'steady'} SIMPLE algorithm solution with {scheme} scheme...")
    start_time = time.time()
    
    if unsteady:
        # Unsteady simulation
        dt = data['dt']
        total_time = data['total_time']
        current_time = 0.0
        
        # Initialize time history storage
        data['time_history'] = [current_time]
        data['u_history'] = [data['u'].copy()]
        data['v_history'] = [data['v'].copy()]
        data['p_history'] = [data['p'].copy()]
        
        # Calculate initial velocity magnitude for animation
        vel_data = get_velocity_data(data)
        data['vel_mag_history'] = [vel_data['velocity_magnitude']]
        
        # Time stepping loop
        while current_time < total_time:
            # Store current fields for use in unsteady terms
            store_previous_timestep(data)
            
            # Solve for this time step
            print(f"\nTime step: t = {current_time:.3f} s")
            residuals = solve_timestep(data, max_iterations)
            
            # Update time
            current_time += dt
            data['current_time'] = current_time
            
            # Store results for this time step
            data['time_history'].append(current_time)
            data['u_history'].append(data['u'].copy())
            data['v_history'].append(data['v'].copy())
            data['p_history'].append(data['p'].copy())
            
            # Calculate velocity magnitude for animation
            vel_data = get_velocity_data(data)
            data['vel_mag_history'].append(vel_data['velocity_magnitude'])
            
            print(f"Time: {current_time:.3f}/{total_time:.1f} s, Final residual: {residuals[-1]:.8e}")
        
        elapsed = time.time() - start_time
        print(f"Unsteady solution completed in {elapsed:.2f} seconds")
        print(f"Final time: {current_time:.3f} s, Final residual: {residuals[-1]:.8e}")
        
        # Return the final residuals
        return residuals
    else:
        # Steady simulation
        set_boundary_conditions(data)
        iteration, residual, residuals = 0, float('inf'), []
        
        while iteration < max_iterations and residual > convergence_criterion:
            calculate_coefficients(data)
            solve_momentum_equations(data)
            calculate_pressure_correction(data)
            correct_pressure_and_velocity(data)
            set_boundary_conditions(data)
            
            residual = calculate_mass_residual(data)
            residuals.append(residual)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Residual: {residual:.8e}")
            
            iteration += 1
        
        elapsed = time.time() - start_time
        print(f"Steady solution completed in {iteration} iterations, {elapsed:.2f} seconds")
        print(f"Final residual: {residual:.8e}")
        
        # Store the solution time
        solution_times[scheme] = elapsed
        
        return residuals

def get_velocity_data(data, time_index=None):
    """Extract velocity data for plotting, with option for specific time step in unsteady simulation"""
    # Extract needed variables
    nx, ny = data['nx'], data['ny']
    dx, dy = data['dx'], data['dy']
    length = data['length']
    
    # Get velocity fields (either current or from history)
    if time_index is not None and data['unsteady']:
        # Ensure time_index is within bounds
        if time_index < 0:
            # Handle negative indices (e.g., -1 for last frame)
            time_index = len(data['u_history']) + time_index
        # Ensure it's still within bounds after adjustment
        time_index = max(0, min(time_index, len(data['u_history'])-1))
        
        u = data['u_history'][time_index]
        v = data['v_history'][time_index]
    else:
        u, v = data['u'], data['v']

    # Calculate center velocities
    u_center = np.zeros((nx, ny))
    v_center = np.zeros((nx, ny))

    for i in range(nx):
        for j in range(ny):
            u_center[i, j] = 0.5 * (u[i, j+1] + u[i+1, j+1])
            v_center[i, j] = 0.5 * (v[i+1, j] + v[i+1, j+1])

    # Calculate velocity magnitude
    velocity_magnitude = np.sqrt(u_center**2 + v_center**2)

    # Extract velocity profiles at x = 20 cm and y = 20 cm
    x_index = int(0.20 / dx)
    y_index = int(0.20 / dy)

    u_profile_y = np.array([0.5 * (u[x_index, j+1] + u[x_index+1, j+1]) for j in range(ny)])
    v_profile_x = np.array([0.5 * (v[i+1, y_index] + v[i+1, y_index+1]) for i in range(nx)])

    x_profile = np.linspace(0, length, nx)
    y_profile = np.linspace(0, length, ny)

    return {
        'u_center': u_center,
        'v_center': v_center,
        'velocity_magnitude': velocity_magnitude,
        'x_profile': x_profile,
        'v_profile_x': v_profile_x,
        'y_profile': y_profile,
        'u_profile_y': u_profile_y
    }

def plot_streamlines(data, scheme, time_index=None, density=1.0, linewidth=0.8, arrowsize=1.0):
    """Plot streamlines for the flow field"""
    # Get velocity data
    vel_data = get_velocity_data(data, time_index)
    X, Y = np.meshgrid(data['x_cell'], data['y_cell'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot streamlines
    streamlines = ax.streamplot(X, Y, vel_data['u_center'].T, vel_data['v_center'].T, 
                              density=density, color='black', linewidth=linewidth, arrowsize=arrowsize)
    
    # Set labels and title
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    
    # Add time info to title if unsteady
    time_str = ""
    if time_index is not None and data['unsteady']:
        # Ensure time_index is within bounds
        if time_index < 0:
            time_index = len(data['time_history']) + time_index
        time_index = max(0, min(time_index, len(data['time_history'])-1))
        time_str = f" at t = {data['time_history'][time_index]:.3f} s"
    
    ax.set_title(f'Streamlines - {scheme} Scheme (Re = {data["Re"]:.0f}){time_str}')
    
    # Set axis limits
    ax.set_xlim(0, data['length'])
    ax.set_ylim(0, data['length'])
    
    plt.tight_layout()
    return fig

def create_streamline_animation(data, save_path=None, fps=10, density=1.0, linewidth=0.8, arrowsize=1.0):
    """Create an animation of streamlines"""
    if not data['unsteady']:
        print("This function requires unsteady simulation data")
        return None
    
    # Verify we have enough data
    if len(data['time_history']) < 2:
        print("Not enough time steps for animation")
        return None
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Get grid
        X, Y = np.meshgrid(data['x_cell'], data['y_cell'])
        
        # Initial plot - empty
        streamlines = ax.streamplot(X, Y, 
                                  get_velocity_data(data, 0)['u_center'].T, 
                                  get_velocity_data(data, 0)['v_center'].T, 
                                  density=density, color='black', linewidth=linewidth, arrowsize=arrowsize)
        
        # Set labels and title
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        title = ax.set_title(f'{data["scheme"]} Scheme - t = 0.000 s')
        
        # Set axis limits
        ax.set_xlim(0, data['length'])
        ax.set_ylim(0, data['length'])
        
        # Function to update the plot for each frame
        def update(frame):
            # Clear previous streamlines
            ax.collections = []  # Remove all collections (streamlines)
            ax.patches = []      # Remove all patches (arrows)
            
            # Ensure frame is within bounds
            frame_idx = min(frame, len(data['time_history'])-1)
            
            # Get velocity data for this frame
            vel_data = get_velocity_data(data, frame_idx)
            
            # Update streamlines
            streamlines = ax.streamplot(X, Y, 
                                      vel_data['u_center'].T, 
                                      vel_data['v_center'].T, 
                                      density=density, color='black', linewidth=linewidth, arrowsize=arrowsize)
            
            # Update title with current time
            if frame_idx < len(data['time_history']):
                title.set_text(f'{data["scheme"]} Scheme - t = {data["time_history"][frame_idx]:.3f} s')
            
            return streamlines.lines,
        
        # Create animation with proper frame count
        frames = min(len(data['time_history']), len(data['u_history']))
        print(f"Creating streamline animation with {frames} frames")
        
        # Use a more robust animation approach
        anim = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)
        
        # Save animation if requested
        if save_path:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                
                # Save with Pillow writer (more reliable)
                print(f"Saving streamline animation to {save_path}")
                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer, dpi=100)
                print(f"Streamline animation saved successfully")
                
            except Exception as e:
                print(f"Error saving streamline animation: {e}")
                traceback.print_exc()
        
        plt.close(fig)
        return anim
        
    except Exception as e:
        print(f"Error creating streamline animation: {e}")
        traceback.print_exc()
        return None

def plot_velocity_contours(data, scheme, time_index=None):
    """Plot velocity contours with option for specific time step in unsteady simulation"""
    # Get velocity data
    vel_data = get_velocity_data(data, time_index)
    X, Y = np.meshgrid(data['x_cell'], data['y_cell'])

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot u-velocity contours
    u_contour = ax1.contourf(X, Y, vel_data['u_center'].T, cmap='coolwarm', levels=20)
    ax1.set_title('u-velocity')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_aspect('equal')
    fig.colorbar(u_contour, ax=ax1, label='u (m/s)')

    # Plot v-velocity contours
    v_contour = ax2.contourf(X, Y, vel_data['v_center'].T, cmap='coolwarm', levels=20)
    ax2.set_title('v-velocity')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_aspect('equal')
    fig.colorbar(v_contour, ax=ax2, label='v (m/s)')

    # Plot velocity magnitude contours
    vel_contour = ax3.contourf(X, Y, vel_data['velocity_magnitude'].T, cmap=COLORMAP, levels=20)
    ax3.set_title('Velocity Magnitude')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.set_aspect('equal')
    fig.colorbar(vel_contour, ax=ax3, label='|V| (m/s)')

    # Set labels and title
    time_str = ""
    if time_index is not None and data['unsteady']:
        # Ensure time_index is within bounds
        if time_index < 0:
            time_index = len(data['time_history']) + time_index
        time_index = max(0, min(time_index, len(data['time_history'])-1))
        time_str = f" at t = {data['time_history'][time_index]:.3f} s"
    
    fig.suptitle(f'Velocity Contours - {scheme} Scheme{time_str}', fontsize=16)
    plt.tight_layout()

    return fig

def create_velocity_contour_animation(data, save_path=None, fps=10):
    """Create an animation of velocity contours with improved error handling"""
    if not data['unsteady']:
        print("This function requires unsteady simulation data")
        return None
    
    # Verify we have enough data
    if len(data['time_history']) < 2:
        print("Not enough time steps for animation")
        return None
    
    # Print debug info
    print(f"Animation debug info:")
    print(f"  Time history length: {len(data['time_history'])}")
    print(f"  Velocity history length: {len(data['u_history'])}")
    
    try:
        # Create figure with black background for better contrast
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate velocity magnitude for all frames if not already done
        if len(data.get('vel_mag_history', [])) < len(data['time_history']):
            print("Precalculating velocity magnitude for all frames...")
            data['vel_mag_history'] = []
            for i in range(len(data['time_history'])):
                vel_data = get_velocity_data(data, i)
                data['vel_mag_history'].append(vel_data['velocity_magnitude'])
        
        # Get max velocity for consistent colormap
        max_vel = max(np.max(vel_mag) for vel_mag in data['vel_mag_history'])
        
        # Ensure max_vel is not too small (fix for the 1e-15 issue)
        if max_vel < 1e-10:
            max_vel = data['u_lid']  # Use lid velocity as reference
            print(f"Warning: Very small velocity magnitude detected. Using lid velocity ({max_vel}) as reference.")
        
        # Initial plot - empty
        X, Y = np.meshgrid(data['x_cell'], data['y_cell'])
        
        # Create initial contour plot with proper scaling
        contour = ax.contourf(X, Y, data['vel_mag_history'][0].T, 
                            cmap=COLORMAP, levels=30, vmin=0, vmax=max_vel)
        cbar = fig.colorbar(contour)
        cbar.set_label('Velocity Magnitude (m/s)', color='white')
        
        # Add title with time
        title = ax.set_title(f'{data["scheme"]} Scheme - t = 0.000 s', fontsize=16, color='white')
        ax.set_xlabel('x (m)', color='white')
        ax.set_ylabel('y (m)', color='white')
        ax.set_aspect('equal')
        ax.tick_params(colors='white')
        
        # Store the contour object for updating
        contour_artist = [contour]
        
        # Function to update the plot for each frame
        def update(frame):
            # Clear previous contours
            for coll in ax.collections:
                coll.remove()
            
            # Ensure frame is within bounds
            frame_idx = min(frame, len(data['time_history'])-1)
            
            # Update velocity magnitude contour
            contour_artist[0] = ax.contourf(X, Y, data['vel_mag_history'][frame_idx].T, 
                                   cmap=COLORMAP, levels=30, vmin=0, vmax=max_vel)
            
            # Update title with current time
            if frame_idx < len(data['time_history']):
                title.set_text(f'{data["scheme"]} Scheme - t = {data["time_history"][frame_idx]:.3f} s')
            
            # Return a list containing the updated artist
            return contour_artist
        
        # Create animation with proper frame count
        frames = min(len(data['time_history']), len(data['vel_mag_history']))
        print(f"Creating animation with {frames} frames")
        
        # Use a more robust animation approach
        anim = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=True)
        
        # Save animation if requested
        if save_path:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                
                # Save with Pillow writer (more reliable)
                print(f"Saving animation to {save_path}")
                writer = PillowWriter(fps=fps)
                
                # Use a more direct approach to save the animation
                anim.save(save_path, writer=writer, dpi=100, savefig_kwargs={'facecolor': 'black'})
                print(f"Animation saved successfully")
                
            except Exception as e:
                print(f"Error saving animation: {e}")
                traceback.print_exc()
        
        plt.close(fig)
        # Reset style
        plt.style.use('default')
        return anim
        
    except Exception as e:
        print(f"Error creating animation: {e}")
        traceback.print_exc()
        # Reset style
        plt.style.use('default')
        return None

def run_lid_driven_cavity(length=0.4, nx=80, ny=80, rho=1.0, nu=0.004, u_lid=1.0, 
                       max_iterations=2000, unsteady=False, dt=0.001, total_time=10.0,
                       create_animations=False, animation_dir="animations"):
    """Run the lid-driven cavity simulation with specified parameters"""
    # Schemes to compare
    schemes = ["UPWIND"]
    residuals_dict = {}
    
    # Solve for each scheme
    for scheme in schemes:
        print(f"\nSolving with {scheme} scheme...")
        data = init_solver(length, nx, ny, rho, nu, u_lid, scheme=scheme, 
                         unsteady=unsteady, dt=dt, total_time=total_time)
        residuals = solve(data, max_iterations)
        residuals_dict[scheme] = residuals
        solvers_data[scheme] = data
    
    # For steady simulations, plot standard comparisons
    if not unsteady:
        # Plot velocity contours for each scheme
        for scheme in schemes:
            plot_velocity_contours(solvers_data[scheme], scheme)
            plt.show()
            
            # Plot streamlines
            plot_streamlines(solvers_data[scheme], scheme)
            plt.show()
        
        # Plot velocity profiles comparison
        plot_velocity_profiles(solvers_data)
        plt.show()
        
        # Plot convergence comparison
        plot_convergence(residuals_dict)
        plt.show()
        
        # Print comparison summary
        print("\nComparison of schemes:")
        for scheme in schemes:
            print(f"{scheme}: Final residual = {residuals_dict[scheme][-1]:.8e}, Iterations = {len(residuals_dict[scheme])}, Time = {solution_times[scheme]:.2f} sec")
    else:
        # For unsteady simulations
        for scheme in schemes:
            # Plot final velocity contours
            plot_velocity_contours(solvers_data[scheme], scheme, time_index=-1)
            plt.show()
            
            # Plot final streamlines
            plot_streamlines(solvers_data[scheme], scheme, time_index=-1)
            plt.show()
        
        # Plot final velocity profiles comparison
        plot_velocity_profiles(solvers_data, time_index=-1)
        plt.show()
        
        # Create animations if requested
        if create_animations:
            # Create directory if it doesn't exist
            os.makedirs(animation_dir, exist_ok=True)
            
            for scheme in schemes:
                try:
                    # Create velocity contour animation
                    vel_path = os.path.join(animation_dir, f"{scheme}_velocity_contours.gif")
                    print(f"Creating velocity contour animation for {scheme}...")
                    create_velocity_contour_animation(solvers_data[scheme], save_path=vel_path, fps=10)
                    
                    # Create streamline animation
                    stream_path = os.path.join(animation_dir, f"{scheme}_streamlines.gif")
                    print(f"Creating streamline animation for {scheme}...")
                    create_streamline_animation(solvers_data[scheme], save_path=stream_path, fps=10)
                except Exception as e:
                    print(f"Error creating animation for {scheme}: {e}")
                    traceback.print_exc()
                
            print(f"Animations saved to {animation_dir} directory")

def plot_velocity_profiles(data_dict, time_index=None):
    """Plot velocity profiles with option for specific time step in unsteady simulation"""
    # Get Reynolds number from first scheme data
    re = next(iter(data_dict.values()))['Re']
    
    # Get Ghia et al. benchmark data for the closest Reynolds number
    if re <= 100:
        ghia_data = get_ghia_data(100, length=data_dict[next(iter(data_dict))]['length'])
    elif re <= 400:
        ghia_data = get_ghia_data(400, length=data_dict[next(iter(data_dict))]['length'])
    else:
        ghia_data = get_ghia_data(1000, length=data_dict[next(iter(data_dict))]['length'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot for each scheme
    for scheme, data in data_dict.items():
        vel_data = get_velocity_data(data, time_index)
        color = data['color']
        
        # Add time info to label if unsteady
        time_str = ""
        if time_index is not None and data['unsteady']:
            # Ensure time_index is within bounds
            if time_index < 0:
                time_index = len(data['time_history']) + time_index
            time_index = max(0, min(time_index, len(data['time_history'])-1))
            time_str = f" at t = {data['time_history'][time_index]:.3f} s"
        
        label = f"{scheme}{time_str}"
        
        # y-velocity profile
        ax1.plot(vel_data['x_profile'], vel_data['v_profile_x'], color=color, linewidth=2, label=label)
        
        # x-velocity profile
        ax2.plot(vel_data['u_profile_y'], vel_data['y_profile'], color=color, linewidth=2, label=label)
    
    # Plot Ghia et al. benchmark data
    ax1.plot(ghia_data['x_ghia'], ghia_data['v_ghia'], 'ko', markersize=6, label=f'Ghia et al. (Re={ghia_data["re"]})')
    ax2.plot(ghia_data['u_ghia'], ghia_data['y_ghia'], 'ko', markersize=6, label=f'Ghia et al. (Re={ghia_data["re"]})')
    
    # Set labels and titles
    ax1.set_title('y-velocity Profile at y = 20 cm')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('v (m/s)')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_title('x-velocity Profile at x = 20 cm')
    ax2.set_xlabel('u (m/s)')
    ax2.set_ylabel('y (m)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_convergence(residuals_dict):
    """Plot convergence history for each scheme"""
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot for each scheme
    for scheme, residuals in residuals_dict.items():
        color = solvers_data[scheme]['color']
        time_taken = solution_times.get(scheme, 0)
        plt.semilogy(residuals, color=color, linewidth=2, 
                    label=f"{scheme} ({time_taken:.2f} sec)")
    
    # Add convergence criterion line
    plt.axhline(y=1e-5, color='r', linestyle='--')
    
    # Set labels and title
    plt.title('Convergence History')
    plt.xlabel('Iteration')
    plt.ylabel('Mass Source Residual')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def check_animation_requirements():
    """Check if animation requirements are met and provide tips"""
    animation_tips = []
    
    # Check for matplotlib animation support
    try:
        from matplotlib.animation import FuncAnimation
        animation_tips.append("✓ Matplotlib animation support is available")
    except ImportError:
        animation_tips.append("✗ Matplotlib animation support is not available. Install matplotlib with 'pip install matplotlib'")
    
    # Check for Pillow
    try:
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=10)
        animation_tips.append("✓ Pillow writer is available for GIF animations")
    except Exception:
        animation_tips.append("✗ Pillow writer is not available. Install Pillow with 'pip install Pillow'")
    
    # Print tips
    print("\nAnimation Requirements Check:")
    for tip in animation_tips:
        print(tip)
    print("\nTips for successful animations:")
    print("1. Use fewer frames for faster animations (reduce total_time or increase dt)")
    print("2. Use GIF format for better compatibility")
    print("3. Reduce resolution (nx, ny) for faster processing")
    print("4. Ensure you have enough memory for storing all time steps")
    
    return animation_tips

# Main execution
if __name__ == "__main__":
    # Check animation requirements and provide tips
    check_animation_requirements()
    
    # Problem parameters
    length = 0.4  # 40 cm = 0.4 m
    nx = ny = 60  # Number of grid cells (reduced for faster animation)
    rho = 1.0  # Density (kg/m^3)
    nu = 0.004  # Kinematic viscosity (m^2/s)
    u_lid = 1.0  # Lid velocity (m/s)
    
    # Simulation mode
    unsteady = True  # Set to False for steady simulation
    dt = 0.1  # Time step size (s) - increased for fewer frames
    total_time = 2.0  # Total simulation time (s) - reduced for faster animation
    
    # Animation options
    create_animations = True  # Set to True to create animations
    animation_dir = "animations"  # Directory to save animations
    
    # Run the simulation
    run_lid_driven_cavity(length, nx, ny, rho, nu, u_lid, max_iterations=1000,
                        unsteady=unsteady, dt=dt, total_time=total_time,
                        create_animations=create_animations, animation_dir=animation_dir)