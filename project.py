# %%
# %matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import place_poles
from mpl_toolkits.mplot3d import Axes3D
import control

# %% [markdown]
## Lorenz system equations
# \begin{align}
# \frac{\delta x}{\delta t} & = \sigma(y-x) \\
# \frac{\delta y}{\delta t} & = x(\rho-z)-y \\
# \frac{\delta z}{\delta t} & = xy - \beta z \\
# \end{align}
### Common parameter values
# $\rho = 28$, $\sigma = 10$, $\beta = 8/3$ produces chaotic motions (though apparently not all solutions are chaotic, does this mean unstable limit cycles?)

# %% [markdown]
# First we need to define a surface of section
# Using code obtained from various places on the internet (wikipedia, stack overflow), we can see that using $z=\rho$ makes a good surface that intersects all trajectories
#
# Also we might want to try multiple periods for orbit length


# %%
# To get the surface of section, just take points from the integration that are equal to (or near I suppose the surface)

rho = 28.0
# sigma = 10.0
# beta = 8.0 / 3.0

def f(state, t, rho = 28.0, sigma = 10.0, beta = 8/3):
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

def run_lorenz(state_init, t_end, rho=28.0, sigma=10.0, beta=8/3):
    state0 = state_init
    t = np.arange(0.0, t_end, 0.001)

    states = odeint(f, state0, t, args=(rho, sigma, beta))
    return states

time_span= 500.0
states = run_lorenz([1.0, 1.0, 1.0], time_span, rho=rho)

# how to plot surface from stack overflow, thanks
xx, yy = np.meshgrid(range(-30,30), range(-30,30))

# calculate corresponding z
z = np.ones((60,60)) * rho

section_error = 0.1
def get_surface_of_section(states, rho=28.0, section_error=0.1):

    surface_indicies = np.where(np.abs(states[:,2] - rho) < section_error)[0]

    temp = []
    i = 0
    while i < len(surface_indicies):
        if len(temp) != 0 and surface_indicies[i] == temp[-1] + 1: 
            q = 1
            while i < len(surface_indicies) and surface_indicies[i] == temp[-1] + q:
                i += 1
                q += 1
        if i < len(surface_indicies):
            temp.append(surface_indicies[i])
        i += 1
    for group in temp:

    surface_indicies = np.array(temp)
    upward_indicies = surface_indicies[::2]

    surface_of_section = states[upward_indicies, :]
    return surface_of_section

surface_of_section = get_surface_of_section(states, rho, 0.1)

# %%
# TODO so it looks like the numerical calculation of the surface can 
# get like 4 points for the same piercing. I'm not sure if reducing 
# the delta will help because then some of the outer orbits' piercings get missed
# Maybe making the times calculated more fine while reducing the delta could do 
# it, but that might just cause it to take longer to calculate and 
# could also just have the same problem. Maybe the best option is to choose the 
# first index of sequential indexes (because we are fairly certain 
# there will not be two piercings within the small time window we provide)


# %% [markdown]
# Important to note how much your fidelity in finding the surface of section determines how effective this method is
# TODO missing one piercing of the surface of section can throw off your whole algorithm, so that's a problem...

# %%

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(states[:, 0], states[:, 1], states[:, 2])
ax.plot_surface(xx, yy, z, alpha=0.4)
ax.scatter(surface_of_section[:, 0], surface_of_section[:, 1], surface_of_section[:, 2], color='r', s=1)

# %%
# So now that we have the surface of section, and the piercings, we need to find several piercings that are close together

# Lets just look through the first 20 for close matches (any period)
def get_proximate_piercings(max_size=20, delta=1.0):
    close = []
    # size = min(max_size,surface_of_section.shape[0])
    size = surface_of_section.shape[0]
    for i in range(size):
        for j in range(i,size):
            if i == j:
                continue
            if np.linalg.norm(surface_of_section[i] - surface_of_section[j]) < delta:
                close.append((i,j))

    return close

close = get_proximate_piercings(20, 1.0)
# print(len(close))
# print(close)
# %%

# So basically we need to make sure these "close" piercings are also within a time distance of each other... Although time is not continuous but the sequence of discrete upward piercings
# So being close to each other numerically in "close" should work

# %%
# Okay so now lets pick an arbitrary pair, say piercing 5 and 6
# print(close[18])

def get_piercing_pairs_close_to_point(point, radius = 1.0, period = 1):
    # now we need more piercings in this area, lets just pick one of the piercings as the location and accept more pairs of piercings in that area
    radius = 1.0
    close_piercings = []
    for i in range(len(close)):
        # print(np.linalg.norm(surface_of_section[close[i][0]] - point) < radius)
        # print(np.linalg.norm(surface_of_section[close[i][1]] - point)< radius)
        if np.linalg.norm(surface_of_section[close[i][0]] - point) < radius \
        and np.linalg.norm(surface_of_section[close[i][1]] - point) < radius:
            close_piercings.append(close[i])
    # next we only want time gaps that match the period we are looking for
    temp = []
    for pair in close_piercings:
        # print(pair[1] - pair[0])
        if pair[0] + period == pair[1]:
            temp.append(pair)
    close_piercings = np.array(temp)
    return close_piercings
# point_of_interest = surface_of_section[close[18][0]]
num = np.random.randint(len(close))
# num = 18
point_of_interest = (surface_of_section[close[num][0]] + surface_of_section[close[num][1]])/2
print(point_of_interest)
close_piercings = get_piercing_pairs_close_to_point(point_of_interest, 1.0, 1)
print(close_piercings)

# %% [markdown]
# So now we need to figure out
# $$z_{n+1} = \hat{A}z_n + \hat{C}$$

# %%
def find_cycle_params(Z_n, Z_n_1):
    # first build matrix of xs
    #   z_n will be 3 by x. 
    num_points = Z_n.shape[0]
    A1 = np.vstack((Z_n[:,0], np.zeros(num_points), np.zeros(num_points), Z_n[:,1], np.zeros(num_points), np.zeros(num_points), Z_n[:,2], np.zeros(num_points), np.zeros(num_points), np.ones(num_points), np.zeros(num_points), np.zeros(num_points))).T
    A2 = np.vstack((np.zeros(num_points), Z_n[:,0], np.zeros(num_points), np.zeros(num_points), Z_n[:,1], np.zeros(num_points), np.zeros(num_points), Z_n[:,2], np.zeros(num_points), np.zeros(num_points), np.ones(num_points), np.zeros(num_points))).T
    A3 = np.vstack((np.zeros(num_points), np.zeros(num_points), Z_n[:,0], np.zeros(num_points), np.zeros(num_points), Z_n[:,1], np.zeros(num_points), np.zeros(num_points), Z_n[:,2], np.zeros(num_points), np.zeros(num_points), np.ones(num_points))).T
    A = np.vstack((A1, A2, A3))
    # print(A)

    # then add on the identity for the cs
    # A = np.hstack((A, np.identity(num_points * 3)))
    # print(A)

    # then turn bs into a vector
    b = np.hstack((Z_n_1[:,0], Z_n_1[:,1], Z_n_1[:,2]))
    # print(b)
    # Solve using least squares
    x = np.linalg.lstsq(A, b)[0]
    # print(x)
    # Split the resulting answer up into A and C and rearrange them as vectors
    #   First 9 entries are entries of A, all the rest are entries of C
    A = np.vstack((x[:3], x[3:6], x[6:9])).T
    C = x[9:]
    C_temp = np.vstack((C,)*num_points)
    # Sanity check: Z_n_1 = A Z_n + C
    check = Z_n_1.T - (A@Z_n.T + C_temp.T)
    # print(np.linalg.norm(check))
    #   How much does the error affect the control?
    # Calculate Z_star using these vectors
    Z_star = np.linalg.inv(np.identity(A.shape[1]) - A) @ C.T

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2])
    ax.plot_surface(xx, yy, z, alpha=0.4)
    ax.scatter(surface_of_section[close, 0], surface_of_section[close, 1], surface_of_section[close, 2], color='r', s=1)
    ax.scatter(Z_star[0], Z_star[1], Z_star[2], color="blue")

    # Return the correct things
    return A,C,Z_star

z_n_indexes = close_piercings[:,0]
z_n_plus_indexes = close_piercings[:,1]

# if all the points are 3-vectors, Zs will be (3 by ?) not vectors.... how are we gonna handle that?
# Naiive approach is to just stretch it out to a longer vector, or maybe solve for different parts of A and stack them together
#   See notes from discussion with charles

A, C, Z_star = find_cycle_params(surface_of_section[z_n_indexes], surface_of_section[z_n_plus_indexes])

# %%
# Now that we have A, we need to get B (check list of things to calculate)
#   This means running it again with a purturbation in our parameter that we will be able to change (which one is that?)
delta_rho = 0.01

states = run_lorenz([1.0, 1.0, 1.0], time_span, rho=rho + delta_rho)
surface_of_section = get_surface_of_section(states, rho + delta_rho, 0.1)
close = get_proximate_piercings(20, 1.0)
close_piercings = get_piercing_pairs_close_to_point(point_of_interest, 1.0, 1)

z_n_indexes = close_piercings[:,0]
z_n_plus_indexes = close_piercings[:,1]

_, _, Z_star_new = find_cycle_params(surface_of_section[z_n_indexes], surface_of_section[z_n_plus_indexes])
# delta_p = np.array([delta_rho, 0, 0])
print(Z_star_new)
print(Z_star)
# print(np.linalg.norm(delta_p))
# I thought p might be a vector, but it makes more sense to just do one parameter at a time
B = (Z_star_new - Z_star) / delta_rho
print(B)

# %%
# Now we need to get the k vector
#   I think for discrete time having all 0 eigenvalues is actually stable
# print(B.shape)
# if B.shape != (3,3):
    # B = np.diag(B)
# print(A.shape)
# print(B.shape)
# K = control.acker(A,B,[0.5,0.5,0.5])
# K = control.acker(A,B,[0,0,0])
K = place_poles(A,B.reshape((B.shape[0], 1)),[0, 0.1, 0.2]).gain_matrix
print(K)
# def run_lorenz(state_init, t_end, rho=28.0, sigma=10.0, beta=8/3):

# %%
    
def f_control(state, t, poi, K=np.identity(3), rho = 28.0, sigma = 10.0, beta = 8/3, delta=0.1):
    x, y, z = state  # Unpack the state vector
    if z - rho < section_error and np.linalg.norm(state - poi) < delta and False:
        # TODO I also need to determine if it is an upward or downward piercing
        rho_new = -K@(state-poi) + rho
        # print(rho_new)
        return sigma * (y - x), x * (rho_new - z) - y, x * y - beta * z  # Derivatives
    else:
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

def controlled_lorenz(state_init, t_end, poi, K, rho_init=28, sigma=10.0, beta=8/3, delta=0.1):
    state0 = state_init
    t = np.arange(0.0, t_end, 0.001)

    states = odeint(f_control, state0, t, args=(poi, K, rho_init, sigma, beta, delta))
    return states

# states = controlled_lorenz([1.0, 1.0, 1.0], time_span, K=K, poi=Z_star, rho_init=rho, delta=1.0)
states = controlled_lorenz([1.0, 1.0, 1.0], 40.0, K=K, poi=Z_star, rho_init=rho, delta=1.0)
print(states.shape)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
partition = states.shape[0] // 3
ax.plot(states[:partition, 0], states[:partition, 1], states[:partition, 2], color="green")
ax.plot(states[partition:-partition, 0], states[partition:-partition, 1], states[partition:-partition, 2])
ax.plot(states[-partition:, 0], states[-partition:, 1], states[-partition:, 2], color="yellow")
# ax.plot(states[:, 0], states[:, 1], states[:, 2])
ax.plot_surface(xx, yy, z, alpha=0.4)
# ax.scatter(surface_of_section[close, 0], surface_of_section[close, 1], surface_of_section[close, 2], color='r', s=1)
ax.scatter(Z_star[0], Z_star[1], Z_star[2], color="blue")
ax.scatter(states[-1, 0], states[-1, 1], states[-1, 2], color="red")
ax.scatter(states[0, 0], states[0, 1], states[0, 2], color="green")

# %%
# For when I run this not in a notebook
print("num: {}".format(num))
plt.draw()
plt.show()