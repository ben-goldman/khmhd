import numpy as np
from spectralDNS import config, get_solver, solve
import matplotlib.pyplot as plt
from mpi4py import MPI
import h5py
# from math import ceil, sqrt
# from scipy import signal
from findiff import Curl
pi = np.pi

def initialize(UB_hat, UB, U, B, X, K, K_over_K2, **context):
    params = config.params
    x = X[0]; y = X[1]; z = X[2]
    dx = params.L/params.N
    curl = Curl(h=dx)
    UB[0] = -1 + np.tanh((z-pi/2)/params.kh_width) - np.tanh((z-3*pi/2)/params.kh_width)
    if params.init_mode == "noise":
        theta = np.random.sample(UB_hat.shape)*2j*np.pi
        phi = np.random.sample(UB_hat.shape)*2j*np.pi
        UB_hat = UB.forward(UB_hat)
        UB_hat += params.deltaU + 1j*params.deltaU + np.random.sample(UB_hat.shape)*params.deltaU #theta + 1j*phi
        UB_hat[:] -= (K[0]*UB_hat[0]+K[1]*UB_hat[1]+K[2]*UB_hat[2]+K[0]*UB_hat[3]+K[1]*UB_hat[4]+K[2]*UB_hat[5])# * K_over_K2
    else:
        k = 2*np.pi*np.array([0, np.cos(params.theta_p), np.sin(params.theta_p)])
        vx = params.deltaU*np.cos(np.tensordot(k, X, axes=1) + np.random.rand())
        vy = params.deltaU*np.cos(np.tensordot(k, X, axes=1) + np.random.rand())
        UB[1] = vx
        UB[2] = vy
        UB[3] = params.deltaB
        UB[4] = 0
        UB[5] = params.deltaB
        UB_hat = UB.forward(UB_hat)

def spectrum(solver, U_hat):
    # SUBTRACTS KH BACKGROUND!
    print(U_hat.shape)
    uiui = np.zeros(U_hat[0].shape)
    uiui[..., 1:-1] = 2*np.sum((U_hat[..., 1:-1]*np.conj(U_hat[..., 1:-1])).real, axis=0)
    uiui[..., 0] = np.sum((U_hat[..., 0]*np.conj(U_hat[..., 0])).real, axis=0)
    uiui[..., -1] = np.sum((U_hat[..., -1]*np.conj(U_hat[..., -1])).real, axis=0)
    uiui *= (4./3.*np.pi)

    # Create bins for Ek
    Nb = int(np.sqrt(sum((config.params.N/2)**2)/6))
    bins = np.array(range(0, Nb))+0.5
    z = np.digitize(np.sqrt(context.K2), bins, right=True)

    # Sample
    Ek = np.zeros(Nb)
    ll = np.zeros(Nb)
    for i, k in enumerate(bins[1:]):
        k0 = bins[i] # lower limit, k is upper
        ii = np.where((z > k0) & (z <= k))
        ll[i] = len(ii[0])
        Ek[i] = (k**3 - k0**3)*np.sum(uiui[ii])

    Ek = solver.comm.allreduce(Ek)
    ll = solver.comm.allreduce(ll)
    for i in range(Nb):
        if not ll[i] == 0:
            Ek[i] = Ek[i] / ll[i]

    return Ek, bins

def update(context):
    params = config.params
    solver = config.solver
    # dx, L, N = params.dx, params.L, params.N
    UEk, bins = spectrum(solver, context.UB_hat[1:3])
    BEk, _ = spectrum(solver, context.UB_hat[4:6])
    update_outfile(f, params.t, ("UEk", "BEk"), (UEk, BEk))
    with np.errstate(divide='ignore'):
        if params.tstep % params.plot_spectrum == 0:
            plt.plot(np.log10(bins), np.log10(UEk))
            plt.suptitle(f"$U^2(k), t={params.t/(2*np.pi)}$")
            plt.savefig(f"UEk{params.tstep:05}.jpg")
            plt.close()
            plt.plot(np.log10(bins), np.log10(BEk))
            plt.suptitle(f"$B^2(k), t={params.t/(2*np.pi)}$")
            plt.savefig(f"BEk{params.tstep:05}.jpg")
            plt.close()
    print(params.t)


def init_outfile(path, dnames, length):
    f = h5py.File(path, mode="w", driver="mpio", comm=MPI.COMM_WORLD)
    f.create_dataset("sim_time", dtype=np.float64, shape=(0), maxshape=(10000))
    for dname in dnames:
        f.create_dataset(dname, dtype=np.float64, shape=(0, length), maxshape=(10000, length))
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
         'T': 40.0,                   # End time
         'M': [M, M, M],
         'L': [2*np.pi, 2*np.pi, 2*np.pi],
         'write_result': 500,
         'solver': "MHD",
         'amplitude_name': f"out_M{M}_Re{Re}.h5",
         'optimization': 'cython',
         'kh_width': 1e-2,
         'deltaU': 1e-4,
         'deltaB': 1e-4,
         'init_mode': 'NOT_noise',
         'theta_p': 0.005,
         'plot_spectrum': 10,
         'convection': 'Divergence'})

    solver = get_solver(update=update)
    context = solver.get_context()
    context.hdf5file.filename = f"img_M{M}_Re{Re}"
    initialize(**context)
    UEk, bins = spectrum(solver, context.U_hat[1:3])
    print(UEk)
        with np.errstate(divide='ignore'):
        plt.plot(np.log10(bins), np.log10(UEk))
        plt.suptitle(f"$U^2(k), t=0$ (initial conditions)")
        plt.savefig(f"Ek_0.jpg")
        plt.close()
    f = init_outfile(config.params.amplitude_name, ["UEk", "BEk"], bins.shape[0])
    # with f:
        # solve(solver, context)
