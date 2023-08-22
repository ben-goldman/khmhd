import numpy as np
from spectralDNS import config, get_solver, solve
import matplotlib.pyplot as plt
from matplotlib import colors
from mpi4py import MPI
import h5py
import logging
pi = np.pi
logging.basicConfig(level=logging.INFO, format="%(asctime)s~>%(message)s")
log = logging.getLogger(__name__)

def initialize(UB, UB_hat, X, K, K2, K_over_K2, **context):
    params = config.params
    x = X[0]; y = X[1]; z = X[2]
    UB[0] = -1 + np.tanh((z-pi/2)/params.kh_width) - np.tanh((z-3*pi/2)/params.kh_width)
    UB[3:6] = params.deltaB

    UB_hat = UB.forward(UB_hat)
    np.random.seed(42)
    k = np.sqrt(K2)
    k = np.where(k == 0, 1, k)
    kk = K2.copy()
    kk = np.where(kk == 0, 1, kk)
    k1, k2, k3 = K[0], K[1], K[2]
    ksq = np.sqrt(k1**2+k2**2)
    ksq = np.where(ksq == 0, 1, ksq)

    theta1, theta2, phi = np.random.sample(UB_hat[0:3].shape)*2j*np.pi
    alpha = np.sqrt(params.deltaU/4./np.pi/kk)*np.exp(1j*theta1)*np.cos(phi)
    beta = np.sqrt(params.deltaU/4./np.pi/kk)*np.exp(1j*theta2)*np.sin(phi)
    UB_hat[0] += (alpha*k*k2 + beta*k1*k3)/(k*ksq)
    UB_hat[1] += (beta*k2*k3 - alpha*k*k1)/(k*ksq)
    UB_hat[2] += beta*ksq/k
    UB_hat[0:3] -= (k1*UB_hat[0] + k2*UB_hat[1] + k3*UB_hat[2])*K_over_K2

    # theta1, theta2, phi = np.random.sample(UB_hat[3:6].shape)*2j*np.pi
    # alpha = np.sqrt(params.deltaB/4./np.pi/kk)*np.exp(1j*theta1)*np.cos(phi)
    # beta = np.sqrt(params.deltaB/4./np.pi/kk)*np.exp(1j*theta2)*np.sin(phi)
    # UB_hat[3] += (alpha*k*k2 + beta*k1*k3)/(k*ksq)
    # UB_hat[4] += (beta*k2*k3 - alpha*k*k1)/(k*ksq)
    # UB_hat[5] += beta*ksq/k
    # UB_hat[3:6] -= (k1*UB_hat[3] + k2*UB_hat[4] + k3*UB_hat[5])*K_over_K2


def spectrum(solver, U_hat):
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
    # U = context.UB_hat[0:3].copy()
    # U_bar = np.mean(U[0], axis=0)
    # B = context.UB_hat[3:6].copy()
    # B_bar = np.mean(B[0], axis=0)
    # for i in range(len(U[0, 0, 0])):
        # U[0, i] =- U_bar
        # B[0, i] =- B_bar
    global U_mean
    global B_mean
    U_mean_previous = U_mean
    B_mean_previous = B_mean
    U_mean = solver.comm.allreduce(np.mean(np.abs(context.U_hat)))*3
    B_mean = solver.comm.allreduce(np.mean(np.abs(context.B_hat)))*3
    g_u = (np.log(U_mean) - np.log(U_mean_previous))/params.dt
    g_b = (np.log(B_mean) - np.log(B_mean_previous))/params.dt
    if solver.rank == 0:
        log.info(f"tstep={params.tstep}, t_sim={params.t:2.3f}, U_mean={U_mean:2.3e}, B_mean={B_mean:2.3e}, g_u={g_u:2.3f}, g_b={g_b:2.3f}")
    if params.tstep % params.plot_spectrum == 0:
        Uk, bins = spectrum(solver, context.U_hat[1:3])
        Bk, _ = spectrum(solver, context.B_hat[1:3])
        update_outfile(f, params.t, ("Uk", "Bk", "U_mean", "B_mean"), (Uk, Bk, U_mean, B_mean))
        with np.errstate(divide='ignore'):
            fname = f"frames/Ek{params.tstep:05}.jpg"
            # log.info(f"Plotting energy spectrum in {fname}")
            fig, ax = plt.subplots()
            ax.plot(bins, Uk, label="$U^2(k)$")
            ax.plot(bins, Bk, label="$B^2(k)$")
            ax.set_yscale("log")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("$k$")
            ax.set_ylabel("$E(k)$")
            fig.legend()
            fig.suptitle(f"$Energy spectrum, t={params.t:2.3f}, M={M}, Re={Re}, Rm={Rm}, dU = {params.deltaU}, dB={params.deltaB}$")
            plt.savefig(fname)
            plt.close()


def init_outfile(path, dnames, length):
    log.info(f"Creating output file at '{path}' with names {dnames} and length {length}")
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
    log.info("Starting simulation.")
    M = 8
    Pm = 1.0
    Re = 900.0
    Rm = Pm*Re
    nu = 1.0/Re
    eta = 1.0/Rm
    # Make sure we can resolve the Kolmogorov scale
    Re_max = ((2/3)*2**M)**(4/3) 
    log.info(f"M={M}, Re={Re}, Rm={Rm} Pm={Pm}, nu={nu}, eta={eta}, Re_max={Re_max}")
    assert Re <= Re_max
    assert Rm <= Re_max
    config.update(
        {'nu': nu,             # Viscosity
         'eta': eta,
         'dt': 0.01,                 # Time step
         'T': 50.0,                   # End time
         'M': [M, M, M],
         'L': [2*np.pi, 2*np.pi, 2*np.pi],
         'write_result': 500,
         'solver': "MHD",
         'out_file': f"out_M{M}_Re{Re}.h5",
         'optimization': 'cython',
         'kh_width': 1e-2,
         'deltaU': 1e-8,
         'deltaB': 1e-8,
         'init_mode': 'noise',
         'plot_spectrum': 20,
         'convection': 'Divergence'})

    log.info("Building solver.")
    solver = get_solver(update=update)
    log.info("Solver built.")
    context = solver.get_context()
    context.hdf5file.filename = f"img_M{M}_Re{Re}"
    log.info("Initializing simulation.")
    initialize(**context)
    log.info("Simulation initialized.")
    log.info("Calculating initial energy spectrum.")
    Uk, bins = spectrum(solver, context.UB_hat[1:3])
    Bk, _ = spectrum(solver, context.UB_hat[4:6])
    with np.errstate(divide='ignore'):
        fig, ax = plt.subplots()
        ax.plot(bins, Uk, label="$U^2(k)$")
        ax.plot(bins, Bk, label="$B^2(k)$")
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("$k$")
        ax.set_ylabel("$E(k)$")
        fig.legend()
        fig.suptitle(f"Energy spectrum, t=0")
        plt.savefig(f"Ek0.jpg")
        plt.close()
    log.info("Initializing custom HDF5 file.")
    f = init_outfile(config.params.out_file, ["Uk", "Bk", "U_mean", "B_mean"], bins.shape[0])
    log.info("Ready to start simulation.")
    with f:
        log.info("About to start solver.")
        U_mean = config.params.deltaU
        B_mean = config.params.deltaB
        solve(solver, context)