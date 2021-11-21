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
p = dist.Field(name='p', bases=(xbasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau1b = dist.Field(name='tau1b', bases=xbasis)
tau2b = dist.Field(name='tau2b', bases=xbasis)
tau1u = dist.VectorField(coords, name='tau1u', bases=xbasis)
tau2u = dist.VectorField(coords, name='tau2u', bases=xbasis)

grid = lambda A: d3.Grid(A)
div = lambda A: d3.Divergence(A, index=0)
from dedalus.core.operators import Skew
skew = lambda A: Skew(A)
#avg = lambda A: d3.Integrate(A, coords)/(Lx*Lz)
avg = lambda A: d3.Integrate(d3.Integrate(A, coords['x']), coords['z'])/(Lx*Lz)
x_avg = lambda A: d3.Integrate(A, coords['x'])/(Lx)
dot = lambda A, B: d3.DotProduct(A, B)
grad = lambda A: d3.Gradient(A, coords)

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)

ex = dist.VectorField(coords, name='ex')
ez = dist.VectorField(coords, name='ez')
ex['g'][0] = 1
ez['g'][1] = 1

exg = grid(ex).evaluate()
ezg = grid(ez).evaluate()

lift_basis = zbasis.clone_with(a=zbasis.a+2, b=zbasis.b+2)
lift = lambda A, n: d3.LiftTau(A, lift_basis, n)

b0 = dist.Field(name='b0', bases=zbasis)
b0['g'] = Lz - z

# Problem
problem = d3.IVP([p, b, u, tau1b, tau2b, tau1u, tau2u], namespace=locals())
problem.add_equation("div(u) + dot(lift(tau2u,-1),ez) = 0")
problem.add_equation("dt(b) + dot(u, grad(b0)) - kappa*lap(b) + lift(tau2b,-2) + lift(tau1b,-1) = - dot(u,grad(b))")
problem.add_equation("dt(u) + τ_drag*u - nu*lap(u) + grad(p) + lift(tau2u,-2) + lift(tau1u,-1) - b*ez = -skew(grid(u))*div(skew(u))")
problem.add_equation("dot(ez, grad(b)(z=0)) = 0")
problem.add_equation("u(z=0) = 0")
problem.add_equation("b(z=Lz) = 0")
problem.add_equation("u(z=Lz) = 0", condition="nx != 0")
problem.add_equation("dot(ex,u)(z=Lz) = 0", condition="nx == 0")
problem.add_equation("p(z=Lz) = 0", condition="nx == 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
solver.stop_iteration = 20 #stop_iter
startup_iter = 10
max_timestep = 0.1

# Initial conditions
zb, zt = zbasis.bounds
b.fill_random('g', seed=42, distribution='normal', scale=1e-5) # Random noise
b['g'] *= z * (Lz - z) # Damp noise at walls
b.low_pass_filter(scales=0.25)

cadence = 1
# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

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
    LHS = (sp.M_min + sp.L_min) @ sp.pre_right
#    LHS = (sp.L_min) @ sp.pre_right
    plot_sparse(LHS)
    ax.set_title('LHS (m = %i)' %m)
    # # Plot L
    # help(sp.LHS_solvers)
    # ax = fig.add_subplot(1, 3, 2)
    # L = sp.LHS_solvers[-1].LU.L
    # plot_sparse(L)
    # ax.set_title('L (m = %i)' %m)
    # # Plot U
    # ax = fig.add_subplot(1, 3, 3)
    # U = sp.LHS_solvers[-1].LU.U
    # plot_sparse(U)
    # ax.set_title('U (m = %i)' %m)
    plt.tight_layout()
    plt.savefig("m_%i.pdf" %m)
    fig.clear()

# Main loop
good_solution = True
logger.info('Starting loop')
start_time = time.time()
while solver.proceed and good_solution:
    if solver.iteration == startup_iter:
        main_start = time.time()
    timestep = CFL.compute_timestep()
    solver.step(timestep)
logger.info("done timesteps")
