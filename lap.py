"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.

Usage:
    rayleigh_benard_2d.py [options]

Options:
    --Nx=<Nx>              Horizontal modes; default is aspect x Nz
    --Nz=<Nz>              Vertical modes [default: 64]
    --aspect=<aspect>      Aspect ratio of domain [default: 4]

    --tau_drag=<tau_drag>       1/Newtonian drag timescale; default is zero drag

    --Rayleigh=<Rayleigh>       Rayleigh number [default: 1e6]

    --run_time_iter=<iter>      How many iterations to run for
    --run_time_simtime=<run>    How long (simtime) to run for

    --label=<label>             Additional label for run output directory
"""
from mpi4py import MPI
import numpy as np
import time
import sys
import os

from docopt import docopt
args = docopt(__doc__)

aspect = float(args['--aspect'])
# Parameters
Lx, Lz = aspect, 1
Nz = int(args['--Nz'])
if args['--Nx']:
    Nx = int(args['--Nx'])
else:
    Nx = int(aspect*Nz)

data_dir = './'+sys.argv[0].split('.py')[0]
data_dir += '_Ra{}'.format(args['--Rayleigh'])
if args['--tau_drag']:
    τ_drag = float(args['--tau_drag'])
    data_dir += '_tau{}'.format(args['--tau_drag'])
else:
    τ_drag = 0
data_dir += '_Nz{}_Nx{}'.format(Nz, Nx)
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])

import logging
logger = logging.getLogger(__name__)
dlog = logging.getLogger('evaluator')
dlog.setLevel(logging.WARNING)

from dedalus.tools.config import config
config['logging']['filename'] = os.path.join(data_dir,'logs/dedalus_log')
config['logging']['file_level'] = 'DEBUG'

from dedalus.tools.parallel import Sync
with Sync() as sync:
    if sync.comm.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))
        logdir = os.path.join(data_dir,'logs')
        if not os.path.exists(logdir):
            os.mkdir(logdir)

import dedalus.public as d3

# TODO: maybe fix plotting to directly handle vectors
# TODO: optimize and match d2 resolution
# TODO: get unit vectors from coords?

comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size

Rayleigh = float(args['--Rayleigh'])
Prandtl = 1
dealias = 3/2
if args['--run_time_simtime']:
    stop_sim_time = float(args['--run_time_simtime'])
else:
    stop_sim_time = np.inf
if args['--run_time_iter']:
    stop_iter = int(float(args['--run_time_iter']))
else:
    stop_iter = np.inf
timestepper = d3.SBDF2
max_timestep = 0.1
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
x = xbasis.local_grid(1)
z = zbasis.local_grid(1)

# Fields
b = dist.Field(name='b', bases=(xbasis,zbasis))
tau1b = dist.Field(name='tau1b', bases=xbasis)
tau2b = dist.Field(name='tau2b', bases=xbasis)

dot = lambda A, B: d3.DotProduct(A, B)
grad = lambda A: d3.Gradient(A, coords)
lap = lambda A: d3.Laplacian(A)

lift_basis = zbasis.clone_with(a=zbasis.a+2, b=zbasis.b+2)
lift = lambda A, n: d3.LiftTau(A, lift_basis, n)
lift_basis1 = zbasis.clone_with(a=zbasis.a+1, b=zbasis.b+1)
lift1 = lambda A, n: d3.LiftTau(A, lift_basis, n)

b0 = dist.Field(name='b0', bases=zbasis)
b0['g'] = Lz - z

problem = d3.LBVP([b,tau1b,tau2b])
problem.add_equation((lap(b)+lift(tau1b,-2)+lift(tau2b,-1),0))
problem.add_equation((b(z=0), 0))
problem.add_equation((b(z=Lz), 0))
solver = problem.build_solver()

# Plot matrices
import matplotlib
import matplotlib.pyplot as plt

# Plot options
fig = plt.figure(figsize=(9,3))
cmap = matplotlib.cm.get_cmap("winter_r")
clim = (-16, 0)
lim_margin = 0.05

def plot_sparse(A):
    I, J = A.shape
    A_mag = np.log10(np.abs(A.A))
    ax.pcolor(A_mag[::-1], cmap=cmap, vmin=clim[0], vmax=clim[1])
    ax.set_xlim(-lim_margin, I+lim_margin)
    ax.set_ylim(-lim_margin, J+lim_margin)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', 'box')
    ax.text(0.95, 0.95, 'nnz: %i' %A.nnz, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    ax.text(0.95, 0.95, '\ncon: %.1e' %np.linalg.cond(A.A), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

for sp in solver.subproblems:
    m = sp.group[0]
    # Plot LHS
    ax = fig.add_subplot(1, 3, 1)
    LHS = (sp.L_min) @ sp.pre_right
    plot_sparse(LHS)
    ax.set_title('LHS (m = %i)' %m)
    # Plot L
    ax = fig.add_subplot(1, 3, 2)
    L = sp.LHS_solvers[-1].LU.L
    plot_sparse(L)
    ax.set_title('L (m = %i)' %m)
    # Plot U
    ax = fig.add_subplot(1, 3, 3)
    U = sp.LHS_solvers[-1].LU.U
    plot_sparse(U)
    ax.set_title('U (m = %i)' %m)
    plt.tight_layout()
    plt.savefig("m_%i.pdf" %m)
    fig.clear()
