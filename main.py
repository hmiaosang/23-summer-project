#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import the required modules
import numpy as np                # scientific library
import matplotlib.pyplot as plt   # for creating plots
from scipy.integrate import solve_ivp
get_ipython().run_line_magic('matplotlib', 'inline')
from functools import partial
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# ## chapter 2 
# 
# The Kronauer's model :
# \begin{equation}
# k^2 \frac{d^2x}{dt^2} +k\mu_x (x^2 -1)\frac{dx}{dt} + \frac{24}{\tau_x} x + k F_{yx} \frac{dy}{dt} =0.\\
# k^2 \frac{d^2y}{dt^2} +k\mu_y (y^2 -1)\frac{dy}{dt} + \frac{24}{\tau_y} y + k F_{xy} \frac{dx}{dt} =0.
# \end{equation}
# 
# using lienard transformation then we have :
# \begin{eqnarray*}
# \frac{dx}{dt} &=& \epsilon\left(x-\frac{x^3}{3}-p_x\right)\\
# \frac{dp_x}{dt} &=&\frac{1}{\epsilon} \left(x+\frac{F_{yx}}{\hat{\omega}_x}y\right)\\
# \frac{dy}{dt} &=& \epsilon\left(y-\frac{y^3}{3}-p_y\right)\\
# \frac{dp_y}{dt} &=&\frac{1}{\epsilon} \left((1+\epsilon\nu)y+\frac{F_{xy}}{\hat{\omega}_x}x\right).
# \end{eqnarray*}

# In[5]:


# Define the range of x values
x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)
x, y = np.meshgrid(x_vals, y_vals) 
# Calculate the corresponding p_x values
p_x_vals = x - x**3 / 3
p_y_vals = y - y**3 / 3
# Create a 3D plot
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(131, projection='3d')
# Plot the surface of M1
ax1.plot_surface(x, y, p_x_vals, cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('p_x')
ax1.set_title('Plot of p_x')

def coupled_vdp_ode2(t, state, d, F_yx, F_xy,n,  wx):
    x, px, y, py = state

    dx_dt = x-x**3/(3) - px
    dpx_dt = d*(x+F_yx*y/wx)
    dy_dt = y-y**3/(3) - py
    dpy_dt = d*((1+d*n)*y+F_xy*x/wx)

    return [dx_dt, dpx_dt, dy_dt, dpy_dt]

# define parameters delta = 0.01
d = 0.01
F_yx = 0.01
F_xy = 0.04
n=0.1
wx=0.1
# define initial conditions
x0 = -2
px0 = 1.0
y0 = -2
py0 = 1.0
initial_state = [x0, px0, y0, py0]
# define time
t_start= 0.0
t_end = 800.0
dt = 0.5
t = np.arange(t_start, t_end, dt)
sol = solve_ivp(lambda t, state: coupled_vdp_ode2(t, state, d, F_yx, F_xy,n,  wx), [t_start, t_end], initial_state, t_eval=t)
x_simulated = sol.y[0]
px_simulated = sol.y[1]
y_simulated = sol.y[2]
py_simulated = sol.y[3]

# Plot the simulated trajectory as arrows
#arrow_scale = 20  # Adjust this value to change the arrow size
ax1.quiver(x_simulated[:-1], y_simulated[:-1], px_simulated[:-1], 
           x_simulated[1:] - x_simulated[:-1], y_simulated[1:] - y_simulated[:-1], 
           px_simulated[1:] - px_simulated[:-1], color='black', 
           length=0.2, arrow_length_ratio=0.1)
# Set labels and title
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('p_x')
ax1.set_title('Simulated Trajectory (Arrows)')
# Add legend
ax1.invert_zaxis()
ax1.view_init(elev=-150, azim=-120)

# Create the second subplot: x vs t
ax2 = fig.add_subplot(132)
ax2.plot(t, x_simulated, 'b-', label='x(t)')
ax2.plot(sol.t, y_simulated, 'g-', label='y(t)')
ax2.set_xlabel('t')
ax2.set_title("x&y vs. Time")

# Create the third subplot: x vs y
ax3 = fig.add_subplot(133)
# Plot the flow vectors (arrows)
ax3.quiver(x_simulated[:-1], y_simulated[:-1], x_simulated[1:] - x_simulated[:-1], y_simulated[1:] - y_simulated[:-1], angles='xy', scale_units='xy', scale=3)
# Add lines at x=-1, x=1, y=-1, y=1
ax3.axvline(x=-1, color='b', linestyle='--', label='x=-1')
ax3.axvline(x=1, color='b', linestyle='--', label='x=1')
ax3.axhline(y=-1, color='r', linestyle='--', label='y=-1')
ax3.axhline(y=1, color='r', linestyle='--', label='y=1')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title("x vs. y")

plt.tight_layout()
plt.show()


# # chapter 3
# ## section2
# 
# Full order JFK model:
# \begin{eqnarray*}
# \frac{dn}{dt} &=& 60(\alpha_0 (\frac{I}{I_0})^p (1 - \beta) (1 - n) - \gamma  n)\\
# u&=&G \alpha_0 (\frac{I}{I_0})^p (1 - \beta) (1 - n)\\
# \frac{dx}{dt} &=&  \frac{\pi}{12}[(x_c + \mu  (\frac{x}{3} + \frac{4}{3}x^3 - \frac{256}{105} x^7 ) + (1 - 0.4 x) (1 - k_c * x_c) u) ]\\
# \frac{dx_c}{dt} &=&  \frac{\pi}{12}[(q x_c (1 - 0.4 x) (1 - k_c x_c)  u - (\frac{24}{ 0.99729 \tau_x})^2  x - k_x (1 - 0.4 x) (1 - k_c x_c) u)] .
# \end{eqnarray*}

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define 3-dimensional Van der Pol equation
def CR(state, t):
    n, x, xc = state
    a0 = 0.05
    g = 0.0075
    I0 = 9500
    p = 0.5
    mu = 0.13
    q = 1 / 3
    tx = 24.2
    k = 0.55
    kc = 0.4
    G = 33.75

    # the piecewise function for light I(t)
    if t % 24 < 16:
        I = 1000
    else:
        I = 0
    # the piecewise function for the sleep state B(t)
    if t % 24 < 8:
        b = 1
    else:
        b = 0

    dn_dt = 60 * (a0 * (I / I0)**p * (1 - b) * (1 - n) - g * n)
    u = G * a0 * (I / I0)**p * (1 - b) * (1 - n)
    dx_dt = np.pi / 12 * (xc + mu * (x / 3 + 4 * x**3 / 3 - 256 * x**7 / 105) + (1 - 0.4 * x) * (1 - kc * xc) * u)
    dxc_dt = np.pi / 12 * (q * xc * (1 - 0.4 * x) * (1 - kc * xc) * u - (24 / (0.99729 * tx))**2 * x - k * x * (1 - 0.4 * x) * (1 - kc * xc) * u)

    return [dn_dt, dx_dt, dxc_dt]

# Define parameters
t_start = 0  # initial time
t_end = 50  # end time
dt = 0.01  # stepsize

# Initial condtions
n0 = 0.1
x0 = -0.49958105
xc0 = 0.90829892
initial_state = [n0, x0, xc0]

# Simulate Van der Pol equation
t = np.arange(t_start, t_end, dt)
states = odeint(CR, initial_state, t)
n_simulated = states[:, 0]
x_simulated = states[:, 1]
xc_simulated = states[:, 2]

# Initialize an array for I(t)
I_values = []
b_values = []
# Iterate through each time value in t
for current_t in t:
    # the piecewise function for light I(t)
    if current_t % 24 < 16:
        I_values.append(1000)
    else:
        I_values.append(0)
        
for current_t in t:        
    # the piecewise function for the sleep state B(t)
    if current_t % 24 < 8:
        b_values.append(1)
    else:
        b_values.append(0)
# Convert the I_values list to a numpy array
I_values = np.array(I_values)
b_values = np.array(b_values)
G = 33.75
a0 = 0.05
I0 = 9500
p = 0.5
# Calculate u_simulated using the I_values array
u_simulated = G * a0 * (I_values / I0)**p * (1 - b_values) * (1 - n_simulated)

# Plot t vs x & xc
plt.plot(t, x_simulated, 'b--', label='X(t)')
plt.plot(t, xc_simulated, 'b', label='Xc(t)')
plt.plot(t, u_simulated, 'r', label='U(t)')
plt.xlabel(r'$t$', fontsize='14')
plt.title("x  u & xc vs. Time")
plt.legend()
plt.show()


# Second order JFK model:
# \begin{eqnarray*}
# \frac{dx}{dt} &=&  \frac{\pi}{12}[(x_c + \mu  (\frac{x}{3} + \frac{4}{3}x^3 - \frac{256}{105} x^7 ) + (1 - 0.4 x) (1 - k_c * x_c) u) ]\\
# \frac{dx_c}{dt} &=&  \frac{\pi}{12}[(q x_c (1 - 0.4 x) (1 - k_c x_c)  u - (\frac{24}{ 0.99729 \tau_x})^2  x - k_x (1 - 0.4 x) (1 - k_c x_c) u)] .\\
# n &=& \frac{\alpha_0 (\frac{I}{I_0})^p (1 - \beta)}{\alpha_0 (\frac{I}{I_0})^p (1 - \beta)+\gamma} \\
# u&=&G \alpha_0 (\frac{I}{I_0})^p (1 - \beta) (1 - n)\\
# \end{eqnarray*}

# In[7]:


def CR2(state, t):
    x, xc = state
    a0 = 0.05
    g = 0.0075
    I0 = 9500
    p = 0.5
    mu = 0.13
    q = 1/3
    tx = 24.2
    k = 0.55
    kc = 0.4
    G=33.75
    
    # the piecewise function for light I(t)
    if t % 24 < 16:
        I = 1000
    else:
        I = 0
    #the piecewise function for the sleep state B(t)
    if t % 24 <8:
        b=1   
    else:
        b=0
    
    n = a0 * (I / I0)**p * (1 - b) /(a0 * (I / I0)**p * (1 - b)+g)
    u = G * a0 * (I / I0)**p * (1 - b) * (1 - n)
    dx_dt = np.pi / 12 * (xc + mu * (x / 3 + 4 * x**3 / 3 - 256 * x**7 / 105) + (1 - 0.4 * x) * (1 - kc * xc) * u)
    dxc_dt = np.pi / 12 * (q * xc * (1 - 0.4 * x) * (1 - kc * xc) * u - (24 / (0.99729 * tx))**2 * x - k * x * (1 - 0.4 * x) * (1 - kc * xc) * u)
    
    return [ dx_dt, dxc_dt]

# Define parameters
t_start = 0  # initial time
t_end = 50  # end time
dt = 0.01  # stepsize

# Initial condtions
x0 = -0.49958105
xc0 = 0.90829892
initial_state2 = [ x0, xc0]

# Simulate Van der Pol equation
t = np.arange(t_start, t_end, dt)
states2 = odeint(CR2, initial_state2, t)
x_simulated2 = states2[:, 0]
xc_simulated2 = states2[:, 1]

# Plot t vs x & xc
plt.plot(t, x_simulated2, 'y--', label='X(t)')
plt.plot(t, xc_simulated2, 'y', label='Xc(t)')
plt.plot(t, u_simulated, 'r', label='U(t)')
plt.xlabel(r'$t$', fontsize='14')
plt.title("x u & xc vs. Time")
plt.legend()
plt.show()


# compare two model

# In[8]:


# Plot the full limit cycle in blue and the 2nd order limit cycle in yellow
plt.plot(x_simulated, xc_simulated, 'b--', label='Limit cycle in full order model')
plt.plot(x_simulated2, xc_simulated2, 'y', label='Limit cycle in 2nd order model')
start_index = 50
start_index2 = 100
# Plot the same segment with an offset in y direction (green, offset added to y coordinates)
offset = 0.2  # Adjust the offset value as needed
plt.plot(x_simulated[start_index:start_index+200], xc_simulated[start_index:start_index+200] + offset, 'b')
plt.plot(x_simulated2[start_index2:start_index2+200], xc_simulated2[start_index2:start_index2+200] - offset, 'y')
plt.legend()
plt.show()


# ## section 3
# different sleep schemes

# In[ ]:





# different regions

# In[ ]:





# # chapter 4
# the model of Wake/NREM/REM Cycling:
# $$
# \frac{d f_W}{d t}=\frac{W_{\infty}\left(g_{A C h, W} C_{A C h}\left(f_R\right)+g_{S, W} C_S\left(f_{S C N}\right)-g_{G, W} C_G\left(f_N\right)\right)-f_W}{\tau_W}
# $$
# $$
# \frac{d f_N}{d t}=\frac{N_{\infty}\left(-g_{N E, N} C_{N E}\left(f_W\right)-g_{S, N} C_S\left(f_{S C N}\right)-g_{G, N} C_G\left(f_N\right)\right)-f_N}{\tau_N}
# $$
# $$
# \frac{d f_R}{d t}=\frac{R_{\infty}\left(g_{A C h, R} C_{A C h}\left(f_R\right)-g_{N E, R} C_{N E}\left(f_W\right)-g_{S, R} C_S\left(f_{S C N}\right)-g_{G, R} C_G\left(f_N\right)\right)-f_R}{\tau_R}
# $$
# $$
# \frac{d f_{S C N}}{d t}=\frac{S C N_{\infty}(c(t))-f_{S C N}}{\tau_{S C N}}
# $$
# $$
# \frac{d h}{d t}=\frac{\mathcal{H}\left(f_W-\theta_W\right) \cdot\left(h_{\max }-h\right)}{\chi \tau_{h w}}+\frac{\mathcal{H}\left(\theta_W-f_W\right) \cdot\left(h_{\min }-h\right)}{\chi \tau_{h s}}
# $$

# In[12]:


# Define the sleep-wake model
def swmodel(y, t):
    
    W_inf = 6.0  
    N_inf = 5.0  
    R_inf = 5.0  
    SCN_inf = 7.0  
    h_max = 323.88  
    h_min = 0.0  
    theta_W = 0.5  
    tau_W = 23.0  
    tau_N = 10.0  
    tau_R = 1.0  
    tau_SCN = 0.5  
    chi = 1.0  
    tau_hw = 946.8  
    tau_hs = 202.2  
    g_Ach_W = 0.8  
    g_S_W = 0.1911
    g_G_W = 1.4928  
    g_NE_N = 1.5  
    g_S_N = 0.2141  
    g_G_N = 0.0  
    g_Ach_R = 2.2  
    g_NE_R = 10.7473 
    g_S_R = 0.8 
    g_G_R = 1.07  
    gam_Ach = 2.5
    gam_NE = 5
    gam_S = 4
    gam_G = 2.5
    xc_simulated = states[:, 2]
    c_t = xc_simulated[int(t)]
    
    f_W, f_N, f_R, f_SCN, h = y

    dfW_dt = (W_inf * (g_Ach_W * np.tanh(f_R/gam_Ach) + g_S_W * np.tanh(f_SCN/gam_S) - g_G_W * np.tanh(f_N/gam_G)) - f_W) / tau_W
    dfN_dt = (N_inf * (-g_NE_N * np.tanh(f_W/gam_NE) - g_S_N * np.tanh(f_SCN/gam_S) - g_G_N * np.tanh(f_N/gam_G)) - f_N) / tau_N
    dfR_dt = (R_inf * (g_Ach_R * np.tanh(f_R/gam_Ach) - g_NE_R * np.tanh(f_W/gam_NE) - g_S_R * np.tanh(f_SCN/gam_S) - g_G_R * np.tanh(f_N/gam_G)) - f_R) / tau_R
    dfSCN_dt = (SCN_inf*c_t - f_SCN) / tau_SCN
    dh_dt = np.heaviside(f_W - theta_W,1) * (h_max - h) / (chi * tau_hw) + np.heaviside(theta_W - f_W, 1) * (h_min - h) / (chi * tau_hs)

    return [dfW_dt, dfN_dt, dfR_dt, dfSCN_dt, dh_dt]



xc_simulated = states[:, 2]
# Define the time points at which to evaluate the solution
t_start = 0
t_end = 100
dt = 0.01
t_points = np.arange(t_start, t_end, dt)  # Replace the time range as needed

# Define the initial conditions
initial_conditions = [4.0, 0.0, 0.0, 0.4 , 180.0]  # Replace with the appropriate initial values

# Solve the ODEs numerically
solution = odeint(swmodel, initial_conditions, t_points)

# Extract the solutions for each variable
f_W_solution = solution[:, 0]
f_N_solution = solution[:, 1]
f_R_solution = solution[:, 2]
f_SCN_solution = solution[:, 3]
h_solution = solution[:, 4]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_points, f_W_solution, label='f_W')
plt.plot(t_points, f_N_solution, label='f_N')
plt.plot(t_points, f_R_solution, label='f_R')
plt.plot(t_points, f_SCN_solution, label='f_SCN')
plt.xlabel('Time')
plt.ylabel('Variables')
plt.legend()
plt.grid(True)
plt.show()


# ## the sleep wake model without SCN state
# 
# \begin{aligned}
# F_{W}^{\prime} & =\frac{F_{W \infty}\left(g_{G, W} C_{G \infty}\left(F_{N}\right)+g_{A, W} C_{A \infty}\left(F_{R}\right)\right)-F_{W}}{\tau_{W}} \\
# & =\Phi_{W}\left(F_{W}, F_{N}, F_{R}\right),\\
# F_{N}^{\prime} & =\frac{F_{N \infty}\left(g_{E, N} C_{E \infty}\left(F_{W}\right)\right)-F_{N}}{\tau_{N}} \\
# & =\Phi_{N}\left(F_{W}, F_{N}, h\right),\\
# F_{R}^{\prime} & =\frac{F_{R \infty}\left(g_{E, R} C_{E \infty}\left(F_{W}\right)+g_{G, R} C_{G \infty}\left(F_{N}\right)+g_{A, R} C_{A \infty}\left(F_{R}\right)\right)-F_{R}}{\tau_{R}} \\
# & =\Phi_{R}\left(F_{W}, F_{N}, F_{R}\right),
# \end{aligned}
# where $C_{i \infty}(f)=\tanh \left(f / \gamma_{i}\right)$ and $F_{X \infty}(c)=X_{\max }\left(0.5\left(1+\tanh \left(\left(c-\beta_{X}\right) / \alpha_{X}\right)\right)\right)$
# 
# Especially, $F_{N \infty}(c)=N_{\max }\left(0.5\left(1+\tanh \left(\left(c-\beta_{N}(h)\right) / \alpha_{N}\right)\right)\right)$
# where  $\beta_{N}(h)=-k h$  for constant $ k$ .

# In[11]:


# define sleep-wake model 2
def swmodel2(y, t):
    
    W_max = 6.5 
    N_max = 5.0  
    R_max = 5.0
    g_G_W= -1.68
    g_G_R=-1.3
    gam_E = 5
    gam_G = 4
    gam_A = 2
    tau_hw = 580.5
    a_W = 0.5
    a_N = 0.175
    a_R = 0.13
    g_A_W = 1
    g_A_R = 1.6
    tau_W = 25
    tau_N = 10
    tau_R = 1
    tau_hs = 510
    b_W = -0.4
    k = 1.5
    b_R = -0.9
    g_E_N = -2
    g_E_R = -4
    h_max = 1
    theta_W = 2
    
    f_W, f_N, f_R, h = y

    dfW_dt = (W_max * (0.5*(1+np.tanh(((g_G_W * np.tanh(f_N/gam_G)+g_A_W * np.tanh(f_R/gam_A))- b_W)*a_W)))- f_W) / tau_W
    dfN_dt = (N_max * (0.5*(1+np.tanh(((g_E_N * np.tanh(f_W/gam_E))+k*h)/a_N))) - f_N) / tau_N
    dfR_dt = (R_max * (0.5*(1+np.tanh(((g_E_R * np.tanh(f_W/gam_E) + g_G_R * np.tanh(f_N/gam_G) + g_A_R * np.tanh(f_R/gam_A))- b_R)*a_R)))-f_R) / tau_R
    dh_dt = np.heaviside(f_W - theta_W,1) * (h_max - h) /  tau_hw - np.heaviside(theta_W - f_W, 1) *  h /  tau_hs

    return [dfW_dt, dfN_dt, dfR_dt, dh_dt]

# Define the time points at which to evaluate the solution
t_start = 0
t_end = 1440
dt = 1
t_points = np.arange(t_start, t_end, dt)  # Replace the time range as needed

# Define the initial conditions
initial_conditions = [4.0, 0.0, 0.0, 0.4]  # Replace with the appropriate initial values

# Solve the ODEs numerically
solution = odeint(swmodel2, initial_conditions, t_points)

# Extract the solutions for each variable
f_W_solution = solution[:, 0]
f_N_solution = solution[:, 1]
f_R_solution = solution[:, 2]
h_solution = solution[:, 3]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_points, f_W_solution, label='f_W')
plt.plot(t_points, f_N_solution, label='f_N')
plt.plot(t_points, f_R_solution, label='f_R')
plt.plot(t_points, h_solution, label='h')
plt.xlabel('Time')
plt.ylabel('Variables')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




