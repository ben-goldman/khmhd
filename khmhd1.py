import numpy as np
from spectralDNS import config, get_solver, solve
import matplotlib.pyplot as plt
from mpi4py import MPI
import h5py
from math import ceil, sqrt
from scipy import signal
from findiff import Curl
pi = np.pi

def initialize(UB_hat, UB, U, B, X, **context):
    params = config.params
    x = X[0]; y = X[1]; z = X[2]
    dx = params.L/params.N
    curl = Curl(h=dx)
    U[0] = -1 + np.tanh((z-pi/2)/params.kh_width) - np.tanh((z-3*pi/2)/params.kh_width)
    if params.init_mode == "noise":
        U += curl(np.random.normal(scale=params.deltaU, size=U.shape))
        B += curl(np.random.normal(scale=params.deltaB, size=B.shape))
        UB_hat = UB.forward(UB_hat)
    elif params.init_mode == "noise_v2":
        UB_hat = UB.forward(UB_hat)
        UB_hat += np.random.normal(scale=params.deltaU, size=UB_hat.shape)

def spectrum(solver, context):
    c = context
    uiui = np.zeros(c.U_hat[0].shape)
    uiui[..., 1:-1] = 2*np.sum((c.U_hat[..., 1:-1]*np.conj(c.U_hat[..., 1:-1])).real, axis=0)
    uiui[..., 0] = np.sum((c.U_hat[..., 0]*np.conj(c.U_hat[..., 0])).real, axis=0)
    uiui[..., -1] = np.sum((c.U_hat[..., -1]*np.conj(c.U_hat[..., -1])).real, axis=0)
    uiui *= (4./3.*np.pi)

    # Create bins for Ek
    Nb = int(np.sqrt(sum((config.params.N/2)**2)/6))
    bins = np.array(range(0, Nb))+0.5
    z = np.digitize(np.sqrt(context.K2), bins, right=True)

    # Sample
    UEk = np.zeros(Nb)
    ll = np.zeros(Nb)
    for i, k in enumerate(bins[1:]):
        k0 = bins[i] # lower limit, k is upper
        ii = np.where((z > k0) & (z <= k))
        ll[i] = len(ii[0])
        UEk[i] = (k**3 - k0**3)*np.sum(uiui[ii])

    UEk = solver.comm.allreduce(UEk)
    ll = solver.comm.allreduce(ll)
    for i in range(Nb):
        if not ll[i] == 0:
            UEk[i] = UEk[i] / ll[i]

    bibi = np.zeros(c.B_hat[0].shape)
    bibi[..., 1:-1] = 2*np.sum((c.B_hat[..., 1:-1]*np.conj(c.B_hat[..., 1:-1])).real, axis=0)
    bibi[..., 0] = np.sum((c.B_hat[..., 0]*np.conj(c.B_hat[..., 0])).real, axis=0)
    bibi[..., -1] = np.sum((c.B_hat[..., -1]*np.conj(c.B_hat[..., -1])).real, axis=0)
    bibi *= (4./3.*np.pi)

    # Create bins for Ek
    Nb = int(np.sqrt(sum((config.params.N/2)**2)/6))
    bins = np.array(range(0, Nb))+0.5
    z = np.digitize(np.sqrt(context.K2), bins, right=True)

    # Sample
    BEk = np.zeros(Nb)
    ll = np.zeros(Nb)
    for i, k in enumerate(bins[1:]):
        k0 = bins[i] # lower limit, k is upper
        ii = np.where((z > k0) & (z <= k))
        ll[i] = len(ii[0])
        BEk[i] = (k**3 - k0**3)*np.sum(bibi[ii])

    BEk = solver.comm.allreduce(BEk)
    ll = solver.comm.allreduce(ll)
    for i in range(Nb):
        if not ll[i] == 0:
            BEk[i] = BEk[i] / ll[i]

    return UEk, BEk, bins

def update(context):
    params = config.params
    solver = config.solver
    dx, L, N = params.dx, params.L, params.N
    # UB = context.UB_hat.backward(context.UB)
    # U, B = UB[:3], UB[3:]
    # u2 = solver.comm.allreduce(np.mean(U[0]**2 + U[1]**2 + U[2]**2))
    # b2 = solver.comm.allreduce(np.mean(B[0]**2 + B[1]**2 + B[2]**2))
    UEk, BEk, _ = spectrum(solver, context)
    update_outfile(f, params.t, ("UEk", "BEk"), (UEk, BEk))
    print(params.t, UEk, BEk)


def init_outfile(path, dnames, shape):
    f = h5py.File(path, mode="w", driver="mpio", comm=MPI.COMM_WORLD)
    f.create_dataset("sim_time", dtype=np.float64, shape=(0), maxshape=(10000))
    for dname in dnames:
        f.create_dataset(dname, dtype=np.float64, shape=(0)+shape, maxshape=(10000)+shape)
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
    Re = 900.0
    # Make sure we can resolve the Kolmogorov scale
    assert Re <= ((2/3)*2**M)**(4/3) 
    Pm = 1.0
    nu = 1.0/Re
    eta = Pm*nu
    config.update(
        {'nu': nu,             # Viscosity
         'eta': eta,
         'dt': 0.01,                 # Time step
         'T': 20.0,                   # End time
         'M': [M, M, M],
         'L': [2*np.pi, 2*np.pi, 2*np.pi],
         'write_result': 500,
         'solver': "MHD",
         'amplitude_name': f"out_M{M}_Re{Re}.h5",
         'optimization': 'cython',
         'kh_width': 1e-2,
         'deltaU': 1e-3,
         'deltaB': 1e-3,
         'init_mode': 'noise_v2',
         'convection': 'Divergence'})

    solver = get_solver(update=update)
    context = solver.get_context()
    context.hdf5file.filename = f"img_M{M}_Re{Re}"
    initialize(**context)
    f = init_outfile(config.params.amplitude_name, ["UEk", "BEk"])
    with f:
        solve(solver, context)
