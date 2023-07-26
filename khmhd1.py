import numpy as np
from spectralDNS import config, get_solver, solve
import matplotlib.pyplot as plt
from mpi4py import MPI
import h5py
from math import ceil, sqrt
from scipy import signal
pi = np.pi

def initialize(UB_hat, UB, U, B, X, **context):
    params = config.params
    N = params.N
    x = X[0]; y = X[1]; z = X[2]
    U[0] = -1 + np.tanh((z-pi/2)/params.kh_width) - np.tanh((z-3*pi/2)/params.kh_width)
    if init_mode == "noise":
        U += np.random.normal(scale=params.deltaU, size=U.shape)
        B += np.random.normal(scale=params.deltaB, size=B.shape)

    UB_hat = UB.forward(UB_hat)


def update(context):
    params = config.params
    solver = config.solver
    dx, L, N = params.dx, params.L, params.N
    UB = context.UB_hat.backward(context.UB)
    U, B = UB[:3], UB[3:]
    u2 = solver.comm.allreduce(np.mean(U[0]**2 + U[1]**2 + U[2]**2))
    b2 = solver.comm.allreduce(np.mean(B[0]**2 + B[1]**2 + B[2]**2))
    update_outfile(f, params.t, ("u2", "b2"), (u2, b2))
    print(params.t, u2, b2)


def init_outfile(path, dnames):
    f = h5py.File(path, mode="w", driver="mpio", comm=MPI.COMM_WORLD)
    f.create_dataset("sim_time", dtype=np.float64, shape=(0), maxshape=(10000))
    for dname in dnames:
        f.create_dataset(dname, dtype=np.float64, shape=(0), maxshape=(10000))
    return f

def update_outfile(f, sim_time, dnames, data):
    st = f["sim_time"]
    st.resize(st.shape[0]+1, axis=0)
    st[-1] = sim_time
    for i, dname in enumerate(dnames):
        ds = f[dname]
        ds.resize(ds.shape[0]+1, axis=0)
        ds[-1] = data[i]


if __name__ == '__main__':
    M = 8
    Re = 100.0
    # Make sure we can resolve the Kolmogorov scale
    assert Re <= ((2/3)*2**M)**(4/3) 
    Pm = 1.0
    nu = 1.0/Re
    eta = Pm*nu
    config.update(
        {'nu': nu,             # Viscosity
         'eta': eta,
         'dt': 0.01,                 # Time step
         'T': 40.0,                   # End time
         'M': [M, M, M],
         'L': [2*np.pi, 2*np.pi, 2*np.pi],
         'write_result': 100,
         'solver': "MHD",
         'amplitude_name': f"out_M{M}_Re{Re}.h5",
         'optimization': 'cython',
         'kh_width': 0.01,
         'deltaU': 0.01,
         'deltaB': 0.01,
         'init_mode': 'noise',
         'convection': 'Divergence'})

    solver = get_solver(update=update)
    context = solver.get_context()
    context.hdf5file.filename = f"img_M{M}_Re{Re}"
    initialize(**context)
    f = init_outfile(config.params.amplitude_name, ["u2", "b2"])
    with f:
        solve(solver, context)
