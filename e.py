from math import ceil, sqrt, exp
import numpy as np
import matplotlib.pyplot as plt

rho = 1.293  # Density of air [kg m^-3]
A = 0.45  # Runner cross-section [m^2]
C_D = 1.2  # Drag coefficient
w = -1.0  # Wind [m s^-1]
m = 80.0  # Mass of runner [kg]
F = 400.0  # Driving force [kg m^-2]
fv = 25.8  # Physiological limit [s N m^-1]
fc = 488.0  # Characteristic force [kg m s^-2]
tc = 0.67  # Characteristic time [s]
time = 10.0  # Time to simulate [s]
dt = 0.1  # Time-step [s]
N = ceil(time / dt)  # Number of steps


# Empty arrays
x = np.zeros(N)
v = np.zeros(N)
a = np.zeros(N)
t = np.zeros(N)

# Force lists
F_list = []
F_C_list = []
F_V_list = []
D_list = []

# Initial conditions
x[0] = 0.0
v[0] = 0.0
t[0] = 0.0

end_index = 0
for i in range(N - 1):
    F = F
    F_C = fc * exp(-(t[i] / tc) ** 2)
    F_V = fv * v[i]
    D = 0.5 * A * (1 - 0.25 * exp(-(t[i] / tc) ** 2)) * rho * C_D * (v[i] - w) ** 2
    F_list.append(F)
    F_C_list.append(F_C)
    F_V_list.append(F_V)
    D_list.append(D)
    driving_force = F + F_C - F_V
    resistance = D
    a[i] = (driving_force - resistance) / m
    v[i + 1] = v[i] + a[i] * dt
    x[i + 1] = x[i] + v[i + 1] * dt
    t[i + 1] = t[i] + dt
    if x[i] > 100:
        print('The 100m race ended after %.2f seconds' % t[i])
        end_index = i + 1
        break

plt.close('all')
f, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(t[:end_index], x[:end_index])
axarr[0].set_title('Position')
axarr[0].set_ylabel(r'[m]')
axarr[1].plot(t[:end_index], v[:end_index])
axarr[1].set_title('Velocity')
axarr[1].set_ylabel('[m s^-1]')
axarr[2].plot(t[:end_index], a[:end_index])
axarr[2].set_title('Acceleration')
axarr[2].set_ylabel('[m s^-2]')
axarr[2].set_xlabel('Time [s]')

# plt.plot(t[:end_index], F_list, label='F')
# plt.plot(t[:end_index], F_C_list, label='FC')
# plt.plot(t[:end_index], F_V_list, label='FV')
# plt.plot(t[:end_index], D_list, label='D')
# plt.title('Comparison of forces')
# plt.xlabel('Time [s]')
# plt.ylabel('Force [N]')
# plt.legend()
#plt.show()

# Terminal velocity
v_terminal = sqrt((2 * F) / (rho * C_D * A))
print(v_terminal)
