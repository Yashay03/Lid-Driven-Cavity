# Import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import time
from scipy.interpolate import interp1d

# Global dictionary to store solver data for each scheme
solvers_data = {}

# Global dictionary to store computation time for each scheme
solution_times = {}

# Ghia et al data for comparision in velocity profile
def get_ghia_data(re = 100, length = 0.4):
    # Scale factor to convert from 1x1 cavity to our domain size, ghia et al data should be dimensionless
    scale = length
    
    # For Re = 100
    # y-coordinate along vertical centerline 
    y_ghia_orig = np.array([0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0])
    
    # x-coordinate along horizontal centerline 
    x_ghia_orig = np.array([0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0])
    
    # Scale coordinates to our domain size
    y_ghia = y_ghia_orig * scale
    x_ghia = x_ghia_orig * scale
    
    # Data for Re = 100
    if re <= 100:
        u_ghia = np.array([0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.0])
        v_ghia = np.array([0.0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.0])
        re_label = 100
    
    return {
        'y_ghia': y_ghia,
        'u_ghia': u_ghia,
        'x_ghia': x_ghia,
        'v_ghia': v_ghia,
        're': re_label
    }
    
# FUNCTION TO STORE EVERYTHING
def init_solver(length=0.4, nx=80, ny=80, rho=1.0, nu=0.004, u_lid=1.0, scheme="CDS", alpha_u=0.7, alpha_v=0.7, alpha_p=0.3):
    # Create a dictionary to store all solver data so it becomes easier for us
    data = {}
    
    # Basic parameters
    data['length'], data['nx'], data['ny'] = length, nx, ny
    data['dx'], data['dy'] = length/nx, length/ny
    data['rho'], data['nu'], data['mu'] = rho, nu, rho*nu
    data['u_lid'], data['scheme'] = u_lid, scheme
    data['Re'] = u_lid * length / nu
    data['alpha_u'], data['alpha_v'], data['alpha_p'] = alpha_u, alpha_v, alpha_p
    data['convergence_criterion'] = 1e-5
    
    # Solution fields (staggered grid)
    data['u'] = np.zeros((nx+1, ny+2))  # u-velocity at (i+1/2, j)
    data['v'] = np.zeros((nx+2, ny+1))  # v-velocity at (i, j+1/2)
    data['p'] = np.zeros((nx+2, ny+2))  # pressure at cell centers
    data['p_corr'] = np.zeros((nx+2, ny+2))
    
    # Coefficients for momentum equations
    data['ae_u'], data['aw_u'], data['an_u'], data['as_u'] = [np.zeros((nx+1, ny)) for _ in range(4)] # u momentum eq
    data['ap_u'], data['b_u'] = np.zeros((nx+1, ny)), np.zeros((nx+1, ny))
    
    data['ae_v'], data['aw_v'], data['an_v'], data['as_v'] = [np.zeros((nx, ny+1)) for _ in range(4)] # v momentum eq
    data['ap_v'], data['b_v'] = np.zeros((nx, ny+1)), np.zeros((nx, ny+1))
    
    # Coefficients for pressure correction
    data['ae_p'], data['aw_p'], data['an_p'], data['as_p'] = [np.zeros((nx, ny)) for _ in range(4)]
    data['ap_p'], data['b_p'] = np.zeros((nx, ny)), np.zeros((nx, ny))
    
    # Grid for plotting
    data['x_cell'] = np.linspace(data['dx']/2, length-data['dx']/2, nx)
    data['y_cell'] = np.linspace(data['dy']/2, length-data['dy']/2, ny)
    
    # Colors for plotting
    colors = {"CDS": "blue", "Upwind": "red", "Hybrid": "green"}
    data['color'] = colors.get(scheme, "blue")
    
    print(f"Reynolds number: {data['Re']}, Using {scheme} scheme (α_u={alpha_u}, α_v={alpha_v}, α_p={alpha_p})")
    
    return data

#  BOUNDARY CONDITIONS FUNCTION
def set_boundary_conditions(data):
    # Extract velocity variables
    u, v = data['u'], data['v']
    u_lid = data['u_lid']
    
    # Lid ar the top
    u[:, -1] = 2*u_lid - u[:, -2] # I am using a ghost cell approach
    
    # Walls - no slip & no penetration
    u[0, :] = u[-1, :] = u[:, 0] = 0
    v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = 0

# THIS FUNCTION CALCULATES COEFFICIENTS FOR ALL SCHEMES
def calculate_coefficients(data):
    # Extract variables
    nx, ny = data['nx'], data['ny']
    dx, dy = data['dx'], data['dy']
    rho, mu = data['rho'], data['mu']
    scheme = data['scheme']
    
    u, v, p = data['u'], data['v'], data['p']
    ae_u, aw_u, an_u, as_u = data['ae_u'], data['aw_u'], data['an_u'], data['as_u']
    ap_u, b_u = data['ap_u'], data['b_u']
    ae_v, aw_v, an_v, as_v = data['ae_v'], data['aw_v'], data['an_v'], data['as_v']
    ap_v, b_v = data['ap_v'], data['b_v']
    
    small = 1e-20  # Small number to avoid division by zero
    
    # U - MOMENTUM COEFFICIENTS
    for i in range(1, nx):
        for j in range(1, ny+1):
            if j-1 >= ny: continue

            # Diffusion terms will be there in every coeff so initialize them first then append more terms
            ae_u[i, j-1] = aw_u[i, j-1] = mu * dy / dx
            an_u[i, j-1] = as_u[i, j-1] = mu * dx / dy
            
            # Average velocities for convection
            u_e = 0.5 * (u[i+1, j] + u[i, j])
            u_w = 0.5 * (u[i, j] + u[i-1, j])
            v_n = 0.5 * (v[i, j] + v[i+1, j])
            v_s = 0.5 * (v[i, j-1] + v[i+1, j-1])
            
            # COEFFICIENTS CORRESPONDING TO CDS SCHEME
            if scheme.lower() == "cds":
                ae_u[i, j-1] -= 0.5 * rho * u_e * dy
                aw_u[i, j-1] += 0.5 * rho * u_w * dy
                an_u[i, j-1] -= 0.5 * rho * v_n * dx
                as_u[i, j-1] += 0.5 * rho * v_s * dx
                
            # COEFFICIENTS CORRESPONDING TO UPWIND SCHEME    
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
            
            # HYBRID SCHEME COEFFICIENTS            
            elif scheme.lower() == "hybrid":
                # Peclet numbers
                Pe_e = rho * u_e * dx / mu
                Pe_w = rho * u_w * dx / mu
                Pe_n = rho * v_n * dy / mu
                Pe_s = rho * v_s * dy / mu
                
                # East face
                if abs(Pe_e) < 2:
                    ae_u[i, j-1] -= 0.5 * rho * u_e * dy
                else:
                    if Pe_e > 2:
                        continue
                    else:
                        ae_u[i, j-1] -= rho * u_e * dy
                
                # West face
                if abs(Pe_w) < 2:
                    aw_u[i, j-1] += 0.5 * rho * u_w * dy
                else:
                    if Pe_w > 2:
                        aw_u[i, j-1] += rho * u_w * dy
                    else:
                        continue
                
                # North face
                if abs(Pe_n) < 2:
                    an_u[i, j-1] -= 0.5 * rho * v_n * dx
                else:
                    if Pe_n > 2:
                        continue
                    else:
                        an_u[i, j-1] -= rho * v_n * dx
                
                # South face
                if abs(Pe_s) < 2:
                    as_u[i, j-1] += 0.5 * rho * v_s * dx
                else:
                    if Pe_s > 2:
                        as_u[i, j-1] += rho * v_s * dx
                    else:
                        continue
            
            A_e = dy * 1
            
            # Assemble
            ap_u[i, j-1] = ae_u[i, j-1] + aw_u[i, j-1] + an_u[i, j-1] + as_u[i, j-1] + rho*u_e*dy/2 - rho*u_w*dy/2 + rho*v_n*dx/2 - rho*v_s*dx/2  
            ap_u[i, j-1] = max(ap_u[i, j-1], small)
            b_u[i, j-1] = (p[i, j] - p[i+1, j]) * A_e
    
    # v MOMENTUM COEFFICIENTS
    for i in range(1, nx+1):
        for j in range(1, ny):
            if i-1 >= nx: 
                continue
            
            # Diffusion terms
            ae_v[i-1, j] = aw_v[i-1, j] = mu * dy / dx
            an_v[i-1, j] = as_v[i-1, j] = mu * dx / dy
            
            # Average velocities
            u_e = 0.5 * (u[i, j] + u[i, j+1])
            u_w = 0.5 * (u[i-1, j] + u[i-1, j+1])
            v_n = 0.5 * (v[i, j+1] + v[i, j])
            v_s = 0.5 * (v[i, j] + v[i, j-1])
            
            if scheme.lower() == "cds":
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
                    
            elif scheme.lower() == "hybrid":
                Pe_e = rho * u_e * dx / mu
                Pe_w = rho * u_w * dx / mu
                Pe_n = rho * v_n * dy / mu
                Pe_s = rho * v_s * dy / mu
                
                if abs(Pe_e) < 2:
                    ae_v[i-1, j] -= 0.5 * rho * u_e * dy
                else:
                    if Pe_e > 2:
                        continue
                    else:
                        ae_v[i-1, j] -= rho * u_e * dy
                
                if abs(Pe_w) < 2:
                    aw_v[i-1, j] += 0.5 * rho * u_w * dy
                else:
                    if Pe_w > 2:
                        aw_v[i-1, j] += rho * u_w * dy
                    else:
                        continue
                
                if abs(Pe_n) < 2:
                    an_v[i-1, j] -= 0.5 * rho * v_n * dx
                else:
                    if Pe_n > 2:
                        continue
                    else:
                        an_v[i-1, j] -= rho * v_n * dx
                
                if abs(Pe_s) < 2:
                    as_v[i-1, j] += 0.5 * rho * v_s * dx
                else:
                    if Pe_s > 2:
                        as_v[i-1, j] += rho * v_s * dx
                    else:
                        continue

            A_n = dx * 1
            
            # Assemble 
            ap_v[i-1, j] = ae_v[i-1, j] + aw_v[i-1, j] + an_v[i-1, j] + as_v[i-1, j] + rho*u_e*dx/2 - rho*u_w*dx/2 + rho*v_n*dy/2 - rho*v_s*dy/2  
            ap_v[i-1, j] = max(ap_v[i-1, j], small)
            b_v[i-1, j] = (p[i, j] - p[i, j+1]) * A_n

# FUNCTION THAT SOLVES MOMENTUM EQUATION
def solve_momentum_equations(data):
    # Extract needed variables
    nx, ny = data['nx'], data['ny']
    alpha_u, alpha_v = data['alpha_u'], data['alpha_v']
    
    u, v = data['u'], data['v']
    ae_u, aw_u, an_u, as_u = data['ae_u'], data['aw_u'], data['an_u'], data['as_u']
    ap_u, b_u = data['ap_u'], data['b_u']
    ae_v, aw_v, an_v, as_v = data['ae_v'], data['aw_v'], data['an_v'], data['as_v']
    ap_v, b_v = data['ap_v'], data['b_v']
    
    # SOLVE U MOMENTUM EQ
    for _ in range(5): # 5 iterations per simple step
        for i in range(1, nx):
            for j in range(1, ny+1):
                if j-1 >= ny: continue
                u_old = u[i, j]
                u_new = (ae_u[i, j-1] * u[i+1, j] +  aw_u[i, j-1] * u[i-1, j] + an_u[i, j-1] * u[i, j+1] + as_u[i, j-1] * u[i, j-1] + b_u[i, j-1]) / ap_u[i, j-1]
                u[i, j] = u_old + alpha_u * (u_new - u_old)
    
    # SOLVE V MOMENTUM EQ
    for _ in range(5): 
        for i in range(1, nx+1):
            for j in range(1, ny):
                if i-1 >= nx: continue
                v_old = v[i, j]
                v_new = (ae_v[i-1, j] * v[i+1, j] + aw_v[i-1, j] * v[i-1, j] + an_v[i-1, j] * v[i, j+1] + as_v[i-1, j] * v[i, j-1] +b_v[i-1, j]) / ap_v[i-1, j]
                v[i, j] = v_old + alpha_v * (v_new - v_old)

# FUNCTION TO CALCULATE PRESSURE COEFFICIENTS
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
    
    small = 1e-20 # This is used so that there is no division by zero
    
    # WE NOW NEED TO CALCULATE PRESSURE CORRECTION COEFF
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            if i-1 >= nx or j-1 >= ny: continue
            
            ae_p[i-1, j-1] = rho * dy**2 / max(ap_u[i, j-1], small) if i < nx else 0
            aw_p[i-1, j-1] = rho * dy**2 / max(ap_u[i-1, j-1], small) if i > 1 else 0
            an_p[i-1, j-1] = rho * dx**2 / max(ap_v[i-1, j], small) if j < ny else 0
            as_p[i-1, j-1] = rho * dx**2 / max(ap_v[i-1, j-1], small) if j > 1 else 0
            
            ap_p[i-1, j-1] = ae_p[i-1, j-1] + aw_p[i-1, j-1] + an_p[i-1, j-1] + as_p[i-1, j-1]
            
            # Source term 
            b_p[i-1, j-1] = rho * ((u[i-1, j] - u[i, j]) * dy + (v[i, j-1] - v[i, j]) * dx)
    
    # SOLVE PRESSURE CORRECTION
    p_corr.fill(0.0)
    for _ in range(50): # Performs 50 iterations to get a good approximation of the pressure correction
        for i in range(1, nx+1):
            for j in range(1, ny+1):
                if i-1 >= nx or j-1 >= ny: continue
                p_corr[i, j] = (
                    ae_p[i-1, j-1] * p_corr[i+1, j] + aw_p[i-1, j-1] * p_corr[i-1, j] + an_p[i-1, j-1] * p_corr[i, j+1] + as_p[i-1, j-1] * p_corr[i, j-1] + b_p[i-1, j-1]) / (ap_p[i-1, j-1] + small)

# FUNCTION TO CORRECT PRESSURE AND VELOCITY
def correct_pressure_and_velocity(data):
    # Extract needed variables
    nx, ny = data['nx'], data['ny']
    dx, dy = data['dx'], data['dy']
    alpha_p = data['alpha_p']
    
    u, v, p, p_corr = data['u'], data['v'], data['p'], data['p_corr']
    ae_p, an_p = data['ae_p'], data['an_p']
    
    # CORRECT PRESSURE
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            p[i, j] += alpha_p * p_corr[i, j]
    
    # CORRECT U VEL
    for i in range(1, nx):
        for j in range(1, ny+1):
            if j-1 >= ny: continue
            dp_corr = p_corr[i, j] - p_corr[i+1, j]
            u[i, j] += ae_p[i, j-1] * dp_corr / dy
    
    # CORRECT V VEL
    for i in range(1, nx+1):
        for j in range(1, ny):
            if i-1 >= nx: continue
            dp_corr = p_corr[i, j] - p_corr[i, j+1]
            v[i, j] += an_p[i-1, j-1] * dp_corr / dx

# FUNCTION TO CALCULATE MASS RESIDUAL, This is particularly important for convergence check
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

# FUNCTION THAT CALLS OTHER FUNCTIONS AND SOLVES THEM USING SIMPLE ALGORITHM
def solve(data, max_iterations=5000):
    # Extract needed variables
    scheme = data['scheme']
    convergence_criterion = data['convergence_criterion']
    
    print(f"Starting SIMPLE algorithm solution with {scheme} scheme...")
    start_time = time.time()
    
    # Set BC
    set_boundary_conditions(data)
    iteration, residual, residuals = 0, float('inf'), []
    
    # Implement SIMPLE Algo
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
    print(f"Solution completed in {iteration} iterations, {elapsed:.2f} seconds")
    print(f"Final residual: {residual:.8e}")
    
    # Store the solution time to display in plots
    solution_times[scheme] = elapsed
    return residuals

# Function that stores the velocities so we can post process properly
# Also extracts velolcity profiles at x = 20 cm and y = 20 cm
def get_velocity_data(data):
    # Extract variables
    nx, ny = data['nx'], data['ny']
    dx, dy = data['dx'], data['dy']
    length = data['length']
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
    
    u_profile_y = u[x_index, :].copy()
    v_profile_x = v[:, y_index].copy()
    
    x_profile = np.linspace(0, length, nx+2)
    y_profile = np.linspace(0, length, ny+2)
    
    return {
        'u_center': u_center,
        'v_center': v_center,
        'velocity_magnitude': velocity_magnitude,
        'x_profile': x_profile,
        'v_profile_x': v_profile_x,
        'y_profile': y_profile,
        'u_profile_y': u_profile_y
    }

##### POST PROCESSING #####

# Plots velocity vectors colored by vel magnitude 
def plot_velocity_vectors(data, scheme):
    # Get velocity data
    vel_data = get_velocity_data(data)
    X, Y = np.meshgrid(data['x_cell'], data['y_cell'])
    
    # Create figure
    plt.figure(figsize=(8, 8))
    
    # Plot velocity vectors
    norm = Normalize(vmin=0, vmax=np.max(vel_data['velocity_magnitude']))
    quiver = plt.quiver(X, Y, vel_data['u_center'].T, vel_data['v_center'].T, vel_data['velocity_magnitude'].T, cmap='viridis', norm=norm, scale=5)
    plt.colorbar(quiver).set_label('Velocity Magnitude (m/s)')
    
    # Set labels and title
    plt.title(f'Velocity Field - {scheme} Scheme')
    plt.xlabel('x (m)') 
    plt.ylabel('y (m)')
    plt.xlim(0, data['length'])
    plt.ylim(0, data['length'])
    plt.grid(True)
    plt.tight_layout()
    
    return plt.gcf()

# Plots Streamlines, visualizing the primary and secondary vortices
def plot_streamlines(data, scheme):
    
    vel_data = get_velocity_data(data)
    X, Y = np.meshgrid(data['x_cell'], data['y_cell'])
    
    # Create figure
    plt.figure(figsize=(8, 8))
    
    # Plot streamlines
    strm = plt.streamplot(X, Y, vel_data['u_center'].T, vel_data['v_center'].T, color='k', cmap='viridis', density=2)
    plt.colorbar(strm.lines).set_label('Velocity Magnitude (m/s)')
    
    # Set labels and title
    plt.title(f'Streamline Pattern - {scheme} Scheme')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.xlim(0, data['length'])
    plt.ylim(0, data['length'])
    plt.grid(True)
    plt.tight_layout()
    
    return plt.gcf()

# Prints plots of velocity profiles for all schemes, this is important to measure accuracy of each scheme
def plot_velocity_profiles(data_dict):
    # Get Reynolds number from first scheme data
    re = next(iter(data_dict.values()))['Re']

    # Get Ghia et al. benchmark data for the re = 100
    
    ghia_data = get_ghia_data(100, length=data_dict[next(iter(data_dict))]['length'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for each scheme
    for scheme, data in data_dict.items():
        vel_data = get_velocity_data(data)
        color = data['color']

        # y-velocity profile
        ax1.plot(vel_data['x_profile'], vel_data['v_profile_x'], color=color, linewidth=2, label=scheme)

        # x-velocity profile
        ax2.plot(vel_data['u_profile_y'], vel_data['y_profile'], color=color, linewidth=2, label=scheme)

    # Plot Ghia et al. benchmark data - using a line of best fit
    if ghia_data: # Make sure ghia_data is not empty
        # Fit a curve to the Ghia data
        v_ghia_interp = interp1d(ghia_data['x_ghia'], ghia_data['v_ghia'], kind='cubic')
        u_ghia_interp = interp1d(ghia_data['y_ghia'], ghia_data['u_ghia'], kind='cubic')

        # Generate more points for the smooth curve
        x_ghia_smooth = np.linspace(min(ghia_data['x_ghia']), max(ghia_data['x_ghia']), 100)
        y_ghia_smooth = np.linspace(min(ghia_data['y_ghia']), max(ghia_data['y_ghia']), 100)

        ax1.plot(x_ghia_smooth, v_ghia_interp(x_ghia_smooth), 'k-', linewidth=2, label='Ghia et al. Fit')
        ax2.plot(u_ghia_interp(y_ghia_smooth), y_ghia_smooth, 'k-', linewidth=2, label='Ghia et al. Fit')

        # Add the Ghia data points as dots
        ax1.plot(ghia_data['x_ghia'], ghia_data['v_ghia'], 'ko', markersize=4, label='Ghia et al. Points')
        ax2.plot(ghia_data['u_ghia'], ghia_data['y_ghia'], 'ko', markersize=4, label='Ghia et al. Points')
    else:
        print(f"Warning: Ghia data not available for Re = {re}. Skipping Ghia plot.")

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

# FUnction creates convergence graph for all schemes and uses logarithmic y - scale
def plot_convergence(residuals_dict):
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot for each scheme
    for scheme, residuals in residuals_dict.items():
        color = solvers_data[scheme]['color']
        time_taken = solution_times[scheme]
        plt.semilogy(residuals, color=color, linewidth=2, label=f"{scheme} ({time_taken:.2f} sec)")
    # Add convergence criterion line
    plt.axhline(y=1e-5, color='r', linestyle='--')
    
    plt.title('Convergence History')
    plt.xlabel('Iteration')
    plt.ylabel('Mass Source Residual')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()


# This function runs the simulation for each scheme and does post processing
def run_lid_driven_cavity(length=0.4, nx=80, ny=80, rho=1.0, nu=0.004, u_lid=1.0, max_iterations=2000):
    # Schemes 
    schemes = ["CDS", "Upwind", "Hybrid"]
    residuals_dict = {}
    
    # Solve for each scheme
    for scheme in schemes:
        print(f"\nSolving with {scheme} scheme...")
        data = init_solver(length, nx, ny, rho, nu, u_lid, scheme=scheme)
        residuals = solve(data, max_iterations)
        residuals_dict[scheme] = residuals
        solvers_data[scheme] = data
    
    # Plot velocity vectors for each scheme separately
    for scheme in schemes:
        plot_velocity_vectors(solvers_data[scheme], scheme)
        plt.show()
    
    # Plot streamlines for each scheme separately
    for scheme in schemes:
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

# Main execution
# First setups up the problem parameters and then calls the main function to run the simulation
if __name__ == "__main__":
    # Problem parameters
    length = 1  # 40 cm = 0.4 m
    nx = ny = 200  # Number of grid cells
    rho = 1.0  # Density (kg/m^3)
    nu = 0.004  # Kinematic viscosity (m^2/s)
    u_lid = 0.4  # Lid velocity (m/s)
    
    # Run the simulation
    run_lid_driven_cavity(length, nx, ny, rho, nu, u_lid, max_iterations=2000)