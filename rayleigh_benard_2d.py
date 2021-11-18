"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.

Usage:
    rayleigh_benard_2d.py [options]

Options:
    --Nx=<Nx>              Horizontal modes; default is 2x Nz
    --Nz=<Nz>              Vertical modes   [default: 64]

    --Rayleigh=<Rayleigh>       Rayleigh number [default: 1e6]

    --run_time_iter=<iter>      How many iterations to run for [default: 20]
    --run_time_simtime=<run>    How long (simtime) to run for
"""
from mpi4py import MPI
import numpy as np
import time
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

from docopt import docopt
args = docopt(__doc__)

# TODO: maybe fix plotting to directly handle vectors
# TODO: optimize and match d2 resolution
# TODO: get unit vectors from coords?

comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size

# Parameters
Lx, Lz = 4, 1
Nz = int(args['--Nz'])
if args['--Nx']:
    Nx = int(args['--Nx'])
else:
    Nx = 2*Nz

Rayleigh = float(args['--Rayleigh'])
Prandtl = 1
dealias = 3/2
if args['--run_time_simtime']:
    stop_sim_time = float(args['--run_time_simtime'])
else:
    stop_sim_time = np.inf
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
skew = lambda A: d3.Skew(A)
# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)

ex = dist.VectorField(coords, name='ex')
ez = dist.VectorField(coords, name='ez')
ex['g'][0] = 1
ez['g'][1] = 1

exg = grid(ex).evaluate()
ezg = grid(ez).evaluate()

lift_basis = zbasis.clone_with(a=1/2, b=1/2) # First derivative basis
lift = lambda A, n: d3.LiftTau(A, lift_basis, n)
grad_u = d3.grad(u) + ez*lift(tau1u,-1) # First-order reduction
grad_b = d3.grad(b) + ez*lift(tau1b,-1) # First-order reduction

b0 = dist.Field(name='tau2b', bases=zbasis)
b0['g'] = Lz - z

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, b, u, tau1b, tau2b, tau1u, tau2u], namespace=locals())
#problem.add_equation("div(u) + lift(dot(tau2u,ez),-1) = 0")
problem.add_equation("div(u) + dot(lift(tau2u,-1),ez) = 0")
problem.add_equation("dt(b) + dot(u, grad(b0)) - kappa*lap(b) + lift(tau2b,-2) + lift(tau1b,-1) = - dot(u,grad(b))")
problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau2u,-2) + lift(tau1u,-1) - b*ez = -skew(grid(u))*div(skew(u))")
problem.add_equation("b(z=0) = 0")
problem.add_equation("u(z=0) = 0")
problem.add_equation("b(z=Lz) = 0")
problem.add_equation("u(z=Lz) = 0", condition="nx != 0")
problem.add_equation("dot(ex,u)(z=Lz) = 0", condition="nx == 0")
problem.add_equation("p(z=Lz) = 0", condition="nx == 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
solver.stop_iteration = int(float(args['--run_time_iter']))

# Initial conditions
zb, zt = zbasis.bounds
b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
b['g'] *= z * (Lz - z) # Damp noise at walls
#b['g'] += Lz - z # Add linear background

# Analysis
# snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=50)
# snapshots.add_task(p)
# snapshots.add_task(b)
# snapshots.add_task(d3.dot(u,ex), name='ux')
# snapshots.add_task(d3.dot(u,ez), name='uz')

cadence = 100
# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=cadence, safety=0.5, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=cadence)
flow.add_property(np.sqrt(d3.dot(u,u))/nu, name='Re')

# Main loop
try:
    logger.info('Starting loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % cadence == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
