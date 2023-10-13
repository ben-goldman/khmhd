import numpy as np
from spectralDNS import config, get_solver, solve
import matplotlib.pyplot as plt
from matplotlib import colors
from mpi4py import MPI
import h5py
import logging

import argparse

pi = np.pi
logging.basicConfig(level=logging.INFO, format="%(asctime)s~>%(message)s")
log = logging.getLogger(__name__)

def initialize(UB, UB_hat, X, K, K2, K_over_K2, **context):
    params = config.params
    assert 1./params.nu <= params.Re_max
    assert 1./params.eta <= params.Re_max
    kf = 3
    np.random.seed(42)
    x = X[0]; y = X[1]; z = X[2]
    k = np.sqrt(K2)
    k = np.where(k == 0, 1, k)
    kk = K2.copy()
    kk = np.where(kk == 0, 1, kk)
    k1, k2, k3 = K[0], K[1], K[2]
    ksq = np.sqrt(k1**2+k2**2)
    ksq = np.where(ksq == 0, 1, ksq)

    UB[0] = -1 + np.tanh((z-pi/2)/params.kh_width) - np.tanh((z-3*pi/2)/params.kh_width)
    UB[3] = params.B0
    UB[4] = params.B0
    UB_hat = UB.forward(UB_hat)
    theta1, theta2, phi = np.random.sample(UB_hat[0:3].shape)*2j*np.pi
    alpha = np.sqrt(params.delta/4./np.pi/kk)*np.exp(1j*theta1)*np.cos(phi)
    beta = np.sqrt(params.delta/4./np.pi/kk)*np.exp(1j*theta2)*np.sin(phi)
    UB_hat[0] += (alpha*k*k2 + beta*k1*k3)/(k*ksq)
    UB_hat[1] = (beta*k2*k3 - alpha*k*k1)/(k*ksq)
    UB_hat[2] = beta*ksq/k
    UB_hat[0:3] -= (K[0]*UB_hat[0] + K[1]*UB_hat[1] + K[2]*UB_hat[2])*K_over_K2

def spectrum(solver, U_hat):
    uiui = np.zeros(U_hat.shape)
    uiui[..., 1:-1] = 2*np.sum((U_hat[..., 1:-1]*np.conj(U_hat[..., 1:-1])).real, axis=0)
    uiui[..., 0] = np.sum((U_hat[..., 0]*np.conj(U_hat[..., 0])).real, axis=0)
    uiui[..., -1] = np.sum((U_hat[..., -1]*np.conj(U_hat[..., -1])).real, axis=0)
    uiui *= (4./3.*np.pi)

    # Create bins for Ek
    Nb = int(np.sqrt(sum((config.params.N/2)**2)/3))
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

    # E0 = uiui.mean(axis=(1, 2))
    # E1 = uiui.mean(axis=(0, 2))
    # E2 = uiui.mean(axis=(0, 1))
    # print(E0.shape)
    # print(E1.shape)
    # print(E2.shape)

    # return Ek, bins, E0, E1, E2
    return Ek, bins


def update(context):
    params = config.params
    solver = config.solver
    if params.tstep % params.compute == 0:
        UBk = []
        bins = []
        means = []
        logstr = f"tstep={params.tstep}/{params.T/params.dt}, t_sim={params.t:2.3f}, log(means)=["
        fig, ax = plt.subplots()
        for i in range(6):
            Ek, bs = spectrum(solver, context.UB_hat[i])
            UBk.append(Ek)
            bins = bs
            mean = solver.comm.allreduce(np.mean(np.abs(context.UB_hat[i])))
            means.append(mean)
            ax.plot(bs, Ek, label = f"U[{i}]")
            logstr += f"{np.log(mean):2.3f}, "
        ax.set_xscale("log")
        ax.set_yscale("log")
        fig.legend()
        fig.suptitle(f"t={params.tstep}")
        plt.savefig(f"frames/Ek{params.tstep:04d}")
        plt.close()

        update_outfile(f, params.t, ("bins", "spectra", "means"), (bins, UBk, means))
        logstr+="]"
        if solver.rank == 0:
            log.info(logstr)


def init_outfile(path, dnames, shapes):
    log.info(f"Creating output file at '{path}' with names {dnames} and shapes {shapes}")
    f = h5py.File(path, mode="w", driver="mpio", comm=MPI.COMM_WORLD)
    f.create_dataset("sim_time", dtype=np.float64, shape=(0,), maxshape=(10000,))
    for i, dname in enumerate(dnames):
        f.create_dataset(dname, dtype=np.float64, shape=(0,) + shapes[i], maxshape=(10000,) + shapes[i])
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
    config.update(
        {'dt': 0.01,
         'T': 60,
         'L': [2*np.pi, 2*np.pi, 2*np.pi],
         'write_result': 25,
         'solver': "MHD",
         'optimization': 'cython',
         'kh_width': 1e-2,
         'delta': 1e-10,
         'init_mode': 'noise',
         'dealias': '3/2-rule',
         'compute': 1,
         'B0': 1e-2,
         'convection': 'Divergence'}, "triplyperiodic")
    config.triplyperiodic.add_argument('--Pm', type=float, default=1.0)
    config.triplyperiodic.add_argument('--N_Re', type=int)
    log.info("Building solver.")
    solver = get_solver(update=update, mesh="triplyperiodic")
    Re_max = ((2/3)*2**config.params.M[0])**(4/3) 
    Re = ((2/3)*2**config.params.N_Re)**(4/3)
    Rm = config.params.Pm*Re
    config.params.nu = 1.0/Re
    config.params.Re_max = Re_max
    config.params.eta = 1.0/Rm
    config.params.out_file = f"out_N{config.params.M[0]}_Re{config.params.N_Re}.h5"
    log.info("Starting simulation.")
    log.info(f"N={config.params.M[0]}, Re={Re}, Rm={Rm} Pm={config.params.Pm}, Re_max={Re_max}")

    log.info("Solver built.")
    context = solver.get_context()
    context.hdf5file.filename = f"img_N{config.params.M[0]}_Re{config.params.N_Re}"
    log.info("Initializing simulation.")
    initialize(**context)
    log.info("Simulation initialized.")
    log.info("Calculating initial energy spectrum.")
    Ek, bins = spectrum(solver, context.U_hat[0])
    for i in range(6):
        Ek, bins = spectrum(solver, context.UB_hat[i])
        plt.plot(Ek, label=f"UB[{i}]")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig("Ek0.pdf")
    nbins = len(bins)
    log.info("Initializing custom HDF5 file.")
    f = init_outfile(config.params.out_file, ("bins", "spectra", "means"), ((nbins,), (6, nbins), (6,)))
    log.info("Ready to start simulation.")
    with f:
        log.info("About to start solver.")
        solve(solver, context)
