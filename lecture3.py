
from cr3bp import EarthMoon as EM, EOMConstructor, initial_velocity
from scipy.integrate import solve_ivp, RK45
import numpy as np
import matplotlib.pyplot as plt

IC1 = np.array([EM.L1 + 100/EM.l, 100/EM.l, 100/EM.l, 0, 0, 0])
IC1[3:5] = initial_velocity(IC1[:2], EM.L1, EM.mu)

IC2 = np.array([EM.L1 + 1000/EM.l, 1000/EM.l, 1000/EM.l, 0, 0, 0])
IC2[3:5] = initial_velocity(IC2[:2], EM.L1, EM.mu)

IC3 = np.array([EM.L1 + 10000/EM.l, 10000/EM.l, 10000/EM.l, 0, 0, 0])
IC3[3:5] = initial_velocity(IC3[:2], EM.L1, EM.mu)

EOMs = EOMConstructor(EM.mu)

atol = 0.001/EM.l
traj1 = solve_ivp(EOMs, [0, 28*24*3600/EM.s], IC1, atol=atol, rtol=0)
traj2 = solve_ivp(EOMs, [0, 28*24*3600/EM.s], IC2, atol=atol, rtol=0)
traj3 = solve_ivp(EOMs, [0, 28*24*3600/EM.s], IC3, atol=atol, rtol=0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(traj1.y[0, :], traj1.y[1, :], traj1.y[2, :], 'b')
ax.plot(traj1.y[0, 0], traj1.y[1, 0], traj1.y[2, 0], 'b+')
ax.plot(traj2.y[0, :], traj2.y[1, :], traj2.y[2, :], 'g')
ax.plot(traj2.y[0, 0], traj2.y[1, 0], traj2.y[2, 0], 'g+')
ax.plot(traj3.y[0, :], traj3.y[1, :], traj3.y[2, :], 'r')
ax.plot(traj3.y[0, 0], traj3.y[1, 0], traj3.y[2, 0], 'r+')

# set the axes to be equal
bound = 12000/EM.l
ax.axes.set_xlim3d(left=EM.L1 - bound, right=EM.L1 + bound)
ax.axes.set_ylim3d(bottom=-bound, top=bound)
ax.axes.set_zlim3d(bottom=-bound, top=bound)

# plot L1
ax.plot(EM.L1, 0, 0, 'k+')

# plot the moon
lunar_radius = 1740000 / EM.l
ax.plot(1 - EM.mu - lunar_radius, 0, 0, 'b+')
ax.plot(1 - EM.mu + lunar_radius, 0, 0, 'b+')

## Stationkeeping
IC4 = np.array([EM.L1 + 10000/EM.l, 10000/EM.l, 10000/EM.l, 0, 0, 0])
IC4[3:5] = initial_velocity(IC4[:2], EM.L1, EM.mu)

solver = RK45(EOMs, 0, IC4, 365*24*3600/EM.s, max_step=24*3600/EM.s, atol=atol, rtol=0)

y = [IC4]
all_dvs = []
while solver.status == "running":
    solver.step()
    y.append(solver.y)
    new_velocity = initial_velocity(solver.y[:2], EM.L1, EM.mu)
    dv = new_velocity - solver.y[3:5]
    dvnorm = np.linalg.norm(dv)
    if dvnorm > 0.01/EM.l*EM.s:
        print("burning")
        all_dvs.append(dvnorm)
        solver.y[3:5] = new_velocity
        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

y = np.array(y).T

ax.plot(y[0, :], y[1, :], y[2, :], 'b')
ax.plot(y[0, 0], y[1, 0], y[2, 0], 'b+')

# set the axes to be equal
bound = 12000/EM.l
ax.axes.set_xlim3d(left=EM.L1 - bound, right=EM.L1 + bound)
ax.axes.set_ylim3d(bottom=-bound, top=bound)
ax.axes.set_zlim3d(bottom=-bound, top=bound)

# plot L1
ax.plot(EM.L1, 0, 0, 'k+')
