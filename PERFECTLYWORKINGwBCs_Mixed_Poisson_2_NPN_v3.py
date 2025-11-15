import os

from mpi4py import MPI
from petsc4py import PETSc
import meshio
import numpy as np

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot, default_scalar_type
from dolfinx.fem import Constant, Function, functionspace, locate_dofs_geometrical, locate_dofs_topological, DirichletBC, dirichletbc, Expression
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_square, locate_entities_boundary, exterior_facet_indices, Mesh
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, dS, ds, grad, inner, nabla_grad, div, SpatialCoordinate, Measure, exp, dot, FacetNormal
from dolfin.fem.solving import LinearVariationalProblem, LinearVariationalSolver
from mpi4py.MPI import COMM_WORLD as comm
from boundary_condition_class import *
from functions_NPN import *

try:
    import pyvista as pv
    import pyvistaqt as pvqt

    have_pyvista = True
    if pv.OFF_SCREEN:
        pv.start_xvfb(wait=0.5)
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False

# Save all logging to file
log.set_output_file("log.txt")
import os
from os import path		# for checking if data file already exists or not
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--length', nargs = 1) #  help='length of pore in micrometers'
parser.add_argument('--width', nargs = 1) # width of pore in nanometers
parser.add_argument('--Vl', nargs = 1)  # left hand side voltage in Volts
parser.add_argument('--dt', nargs = 1)  # left hand side voltage in Volts
parser.add_argument('--numsteps', nargs = 1)  # left hand side voltage in Volts
parser.add_argument('--outputFolder', nargs = 1)  # left hand side voltage in Volts
args = parser.parse_args()
L = int(args.length[0])
w_ = int(args.width[0])        
Vl = int(args.Vl[0])
dt = float(args.dt[0])
NSteps = int(args.numsteps[0])
outFolder = str(args.outputFolder[0])

filename = 'FeNICsInputFiles/'+'L'+str(L)+'_w'+str(w_)
output_dir = outFolder +'/'
'''
tet_data = msh.cell_data_dict["gmsh:physical"]["tetra10"]
#tet_data = msh.cell_data_dict["gmsh:physical"]["line"]
meshio.write(mesh_name+".xdmf",
    meshio.Mesh(points=msh.points,
        cells={"tetra": msh.cells_dict["tetra10"]},
        #cells={"line": msh.cells_dict["line"]},
        cell_data={"dom_marker": [tet_data]}
    )
)
'''
'''
import meshio
msh = meshio.read(filename+".msh")
#meshio.write(filename+".xdmf", meshio.Mesh(points = msh.points, cells = {'tetra': msh.cells_dict['tetra']}))
meshio.write(filename+".xdmf", meshio.Mesh(points = msh.points, cells = {'line3': msh.cells_dict['line3']}))
meshio.write(filename+".xdmf", meshio.Mesh(points = msh.points, cells = {'triangle6': msh.cells_dict['triangle6']}))
'''

convert_msh2xdmf_nitish(filename)
##unit_box_surf.xdmf
##electrode2_surf.xdmf
with XDMFFile(comm, filename+"_surf"+".xdmf", "r") as file:
		msh = file.read_mesh(name="Grid")

#msh = create_unit_square(MPI.COMM_WORLD, 100, 100, CellType.quadrilateral)

P1 = element("Lagrange", msh.basix_cell(), 1)
ME = functionspace(msh, mixed_element([P1, P1]))
'''
k = 1
Q_el = element("BDMCF", msh.basix_cell(), k)
P_el = element("DG", msh.basix_cell(), k - 1)
V_el = mixed_element([Q_el, P_el])
ME = fem.functionspace(msh, V_el)
'''
#(q, v) = TestFunctions(ME)
q, v = ufl.TestFunctions(ME)


u = Function(ME)  # current solution
u0 = Function(ME)  # solution from previous converged step

# Split mixed functions
c, phi = ufl.split(u)
c0, phi0 = ufl.split(u0)

#dt = 1.0e-6  # time step
theta = 0.5  # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicholson
e=1.6e-19
Z=1
D=1e-8
F=96500
T=300
R=8.314





# Zero u
u.x.array[:] = 1e-14


fdim = msh.topology.dim - 1
facets_left = locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0, rtol=1e-05, atol=1e-14))
facets_right = locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], L*1e-9, rtol=1e-05, atol=1e-14))  #electrode2: 8E-05  unit_box: 1E-05
print(facets_right)
V0 = ME.sub(0)
Q0, _ = V0.collapse()
conc_dofs_left = locate_dofs_topological((V0, Q0), fdim, facets_left)
conc_dofs_right = locate_dofs_topological((V0, Q0), fdim, facets_right)
conc_bc_left = Function(Q0)
conc_bc_right = Function(Q0)
conc_bc_left.interpolate(lambda x: 1.0E-14*( 1 + 1.0E-16*x[0]))
conc_bc_right.interpolate(lambda x: 1.0E-0*( 1 + 1.0E-16*x[0]) )
bc_c_left = dirichletbc(conc_bc_left, conc_dofs_left, V0)
bc_c_right = dirichletbc(conc_bc_right, conc_dofs_right, V0)

V1 = ME.sub(1)
Q1, _ = V1.collapse()
phi_dofs_left = locate_dofs_topological((V1, Q1), fdim, facets_left)
phi_dofs_right = locate_dofs_topological((V1, Q1), fdim, facets_right)
phi_bc_left = Function(Q1)
phi_bc_right = Function(Q1)
phi_bc_left.interpolate(lambda x: Vl*(1 + 1.0E-16*x[0]))
phi_bc_right.interpolate(lambda x: 1.0E-14*(1 + 1.0E-16*x[0]))
bc_phi_left = dirichletbc(phi_bc_left, phi_dofs_left, V1)
bc_phi_right = dirichletbc(phi_bc_right, phi_dofs_right, V1)

bc = [bc_c_right, \
        bc_c_left, \
        bc_phi_right, \
        bc_phi_left
        ]


ds = Measure('ds', domain=msh)
n  = FacetNormal(msh)
# Compute the chemical potential df/dc
c = ufl.variable(c)
#f = 100 * c**2 * (1 - c) ** 2
#dfdc = ufl.diff(f, c)
def J(c0, phi0):
	return -D*(grad(c0) - (Z*F/(R*T)*c0*grad(phi0)))


# mu_(n+theta)
c_mid = (1.0 - theta) * c0 + theta * c0

dx = Measure("dx", msh)
F0 = inner(c, q) * dx - inner(c0, q) * dx \
    - dt * inner(J(c,phi0), grad(q)) * dx \
    + dt * inner(dot(J(c0, phi0), n), q)*ds
    #+ ( dot( (J(c0, phi0)), n ) )*dS \

F1 = -inner(grad(phi), grad(v))*dx + e*Z*inner(c_mid, v) * dx + (dot(grad(phi),n)*v) * ds 
F = F0 + F1



# Create nonlinear problem and Newton solver
problem = NonlinearProblem(F, u,
	bcs = bc
	)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
#solver.convergence_criterion = "residual"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-6
#solver.atol = np.sqrt(np.finfo(default_real_type).eps) * 1e-3
#solver.max_it = 2000  # https://fenicsproject.discourse.group/t/krylov-solvers-option-max-it-and-a-few-questions/12859/3

# We can customize the linear solver used inside the NewtonSolver by
# modifying the PETSc options
ksp = solver.krylov_solver
opts = PETSc.Options()  # type: ignore
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}ksp_type"] = "gmres"
#opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
opts[f"{option_prefix}pc_type"] = "hypre"
opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 2000
opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
sys = PETSc.Sys()  # type: ignore
# For factorisation prefer MUMPS, then superlu_dist, then default.
if sys.hasExternalPackage("mumps"):
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
elif sys.hasExternalPackage("superlu_dist"):
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
ksp.setFromOptions()



# Output file
file = XDMFFile(MPI.COMM_WORLD, output_dir+"/output.xdmf", "w")
file.write_mesh(msh)

# Step in time
t = 0.0

#  Reduce run time if on test (CI) server
if "CI" in os.environ.keys() or "GITHUB_ACTIONS" in os.environ.keys():
    T = 10 * dt			# original: T = 3 * dt
else:
    T = NSteps * dt		# original: T = 50 * dt

# Get the sub-space for c and the corresponding dofs in the mixed space
# vector
V0, dofs = ME.sub(0).collapse()

# Prepare viewer for plotting the solution during the computation
if have_pyvista:
    # Create a VTK 'mesh' with 'nodes' at the function dofs
    topology, cell_types, x = plot.vtk_mesh(V0)
    grid = pv.UnstructuredGrid(topology, cell_types, x)

    # Set output data
    grid.point_data["c"] = u.x.array[dofs].real
    grid.set_active_scalars("c")

    p = pvqt.BackgroundPlotter(title="concentration", auto_update=True)
    p.add_mesh(grid, clim=[0, 1])
    p.view_xy(True)
    p.add_text(f"time: {t}", font_size=12, name="timelabel")

c = u.sub(0)
phi = u.sub(1)
u0.x.array[:] = u.x.array
while t < T:
    t += dt
    r = solver.solve(u)
    #print(f"Step {int(t / dt)}: num iterations: {r[0]}", end='\n')
    u0.x.array[:] = u.x.array
    #u.sub(0).interpolate(c)               #######         ############        #######
    file.write_function(c, t)
    print("Progress (%): ", int(100*t/T), end='\r')
    # Update the plot window
    if have_pyvista:
        p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
        grid.point_data["c"] = u.x.array[dofs].real
        p.app.processEvents()

file.close()

# Update ghost entries and plot
if have_pyvista:
    u.x.scatter_forward()
    grid.point_data["c"] = u.x.array[dofs].real
    screenshot = None
    if pv.OFF_SCREEN:
        screenshot = "c.png"
    pv.plot(grid, show_edges=True, screenshot=screenshot)







'''
#Defining boundaroes for a square box:
def boundary_L(x, on_boundary):
	tol=1E-14
	return on_boundary and near(x[0], 0, tol)

def boundary_R(x):
	return np.isclose(x[0], 1)

def boundary_T(x, on_boundary):
	tol=1E-14
	return on_boundary and near(x[1], 1, tol)

def boundary_D(x, on_boundary):
	tol=1E-14
	return on_boundary and near(x[1], 0, tol)
'''

'''
# Interpolate initial condition
rng = np.random.default_rng(42)
u.sub(0).interpolate(lambda x: 1e-1 + 1e-14*x[0])
#u.sub(0).interpolate(lambda x: 0.63 + 0.02 * (0.5 - rng.random(x.shape[0])))
u.sub(1).interpolate(lambda x: 1e1 + 1e-14*x[0])
u.x.scatter_forward()
'''


'''
#HINTS:
#https://fenicsproject.discourse.group/t/dirichletbcs-assignment-for-coupled-vector-field-problem/9050/3
#https://fenicsproject.discourse.group/t/how-to-interpolate-special-functions/14470

fdim = msh.topology.dim - 1
facets_left = locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
facets_right = locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], float(L/1e6)))  #electrode2: 8E-05  unit_box: 1E-05
V0 = ME.sub(0)
Q0, _ = V0.collapse()
conc_dofs_left = locate_dofs_topological((V0, Q0), fdim, facets_left)
conc_dofs_right = locate_dofs_topological((V0, Q0), fdim, facets_right)
conc_bc_left = Function(Q0)
conc_bc_right = Function(Q0)
conc_bc_left.interpolate(lambda x: 1.0E-14*( 1 + 1.0E-14*x[0]))
conc_bc_right.interpolate(lambda x: 1.0E+00*( 1 + 1.0E-14*x[0]) )
bc_c_left = dirichletbc(conc_bc_left, conc_dofs_left, V0)
bc_c_right = dirichletbc(conc_bc_right, conc_dofs_right, V0)
V1 = ME.sub(1)
Q1, _ = V1.collapse()
phi_dofs_left = locate_dofs_topological((V1, Q1), fdim, facets_left)
phi_dofs_right = locate_dofs_topological((V1, Q1), fdim, facets_right)
phi_bc_left = Function(Q1)
phi_bc_right = Function(Q1)
phi_bc_left.interpolate(lambda x: Vl*1e0*(1 + 1.0E-14*x[0]))
phi_bc_right.interpolate(lambda x: 1.0E-14*(1 + 1.0E-14*x[0]))
bc_phi_left = dirichletbc(phi_bc_left, phi_dofs_left, V1)
bc_phi_right = dirichletbc(phi_bc_right, phi_dofs_right, V1)

bc = [bc_c_right, bc_phi_left, bc_phi_right]
#bc = [bc_c_left, bc_c_right, bc_phi_left, bc_phi_right]
'''

'''
# Define the Dirichlet condition
boundary_conditions = [BoundaryCondition("Dirichlet", 1, u_ex),
                       BoundaryCondition("Dirichlet", 2, u_ex),
                       BoundaryCondition("Robin", 3, (r, s)),
                       BoundaryCondition("Neumann", 4, g)]
'''

'''
#https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html
#loop through the boundary condition and append them to L(v) or the list of Dirichlet boundary conditions
bc = []
for condition in boundary_conditions:
    if condition.type == "Dirichlet":
        bcs.append(condition.bc)
    else:
        F += condition.bc
'''
