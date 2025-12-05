import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

################ Tweede mogelijkheid

# We must solve the time evolution of the system with the Hamiltonian which desribes the interaction of the two particles

# Base operators for 1 qubit
si = qt.qeye(2) # Identity
sx = qt.sigmax() # Pauli X
sz = qt.sigmaz() # Pauli Z

'''
Edit 1: made |++> state the initial state in stead of |00>, to achieve entanglement

# # Initial state: Productstate (not entangled)
# # |00> tstate
# psi0 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) # to create Hilbert space
# density_matrix = qt.ket2dm(psi0)
'''

# Initial state: Productstate (not entangled)
# |++> tstate - een eigenstaat van de z-vrije Hamiltonian die goed werkt voor entanglement
psi_plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit() # .unit() is for normalization
psi0 = qt.tensor(psi_plus, psi_plus) 
density_matrix = qt.ket2dm(psi0) 

# ... rest van de code blijft hetzelfde, met omega = 1

'''
We use Ising-coupling (sig_x X sig_x) where coupling strength g represtents the distance
H = H_0 + H_int(R(t)) , where:
    H_0 = individual energy of qubits (in magn. field) = omega/2 (sig_z X I + I X sig_z)
    H_int(R) = coupling part between qubits as distance R reduces (we increase g(t) in time to simulate them getting closer)
             = g(t)(sig_x X sig_x)
    

'''
omega = 1

# H0
H0 = 0.5 * omega * (qt.tensor(sz, si) + qt.tensor(si, sz))

# H_int: Interaction term 
H_int_op = qt.tensor(sx, sx)

# Define coupling strength g(t) as function of time (simulates getting closer)
# We increase distance linearly from no coupling at all to max coupling (g_max)
g_max = 1 # Max coupling

def g_t(t, args):
    T = args['T'] # Total simulation time
    return g_max * (t / T) # Lineairly increasing of coupling

# Total Hamiltonian for time evolution
# [Operator, Function/String]
H_coupled = [H0, [H_int_op, g_t]]

# Solving time evolution Schrödinger equation |psi(t)>
T = 10.0 # Total simulation time
t_list = np.linspace(0, T, 100)
args = {'T': T}

result = qt.mesolve(H_coupled, density_matrix, t_list, [], args=args)

'''
We use concurrence to view how much the qubits are entangled:
0 is no entanglement, 1 is fully entangled
'''
# Calculation concurrence for every time step
concurrence_list = []
for state in result.states:
    # State passed on as a density matrix
    rho = state.dag() * state # converts ket to density matrix
    concurrence_list.append(qt.concurrence(rho))

'''
Now plotting concurrence against time
'''
print('Plot with magnetic field')
plt.figure()
plt.plot(t_list, concurrence_list, label='Concurrence $C(t)$')
plt.xlabel('Time (t) [Simulates decreasing distance]')
plt.ylabel('Concurrence')
plt.title('Concurrence $C(t)$ for increasing entanglement')
plt.grid(True)
plt.legend()
plt.ylim(0, 1.05)
plt.show()
