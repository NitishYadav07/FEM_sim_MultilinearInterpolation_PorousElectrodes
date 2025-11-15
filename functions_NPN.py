import meshio
from mpi4py import MPI
import numpy as np
from dolfinx import mesh

#https://bleyerj.github.io/comet-fenicsx/tips/mark_facets/mark_facets.html
#a utility function to tag some facets of a mesh by providing geometrical marker functions
def mark_facets(domain, surfaces_dict):
    """Mark facets of the domain according to a geometrical marker

    Parameters
    ----------
    domain : Mesh
        `dolfinx` mesh object
    surfaces_dict : dict
        A dictionary mapping integer tags with a geometrical marker function {tag: marker(x)}

    Returns
    -------
    facet_tag array
    """
    fdim = domain.topology.dim - 1
    marked_values = []
    marked_facets = []
    # Concatenate and sort the arrays based on facet indices
    for tag, location in surfaces_dict.items():
        facets = mesh.locate_entities_boundary(domain, fdim, location)
        marked_facets.append(facets)
        marked_values.append(np.full_like(facets, tag))
    marked_facets = np.hstack(marked_facets)
    marked_values = np.hstack(marked_values)
    sorted_facets = np.argsort(marked_facets)
    facet_tag = mesh.meshtags(
        domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
    )
    return facet_tag

#For instance, tagging the bottom, right, top and left boundary of a square mesh will look like this:
'''
N = 4
domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)

def left(x):
    return np.isclose(x[0], 0.0)

def bottom(x):
    return np.isclose(x[1], 0.0)

def right(x):
    return np.isclose(x[0], 1.0)

def top(x):
    return np.isclose(x[1], 1.0)

facets = mark_facets(domain, {1: bottom, 2: right, 3: top, 4: left})
print(facets.values)
#output >> [1 1 2 2 1 2 1 2 4 3 4 3 4 3 4 3]
'''


'''
#Note that we can also adapt the function to mark entities of specified dimension i.e. subdomains if dim=tdim, facets if dim=tdim-1, etc. where tdim is the domain topological dimension.
def mark_entities(domain, dim, entities_dict):
    """Mark entities of specified dimension according to a geometrical marker function

    Parameters
    ----------
    domain : Mesh
        `dolfinx` mesh object
    dim : int
        Dimension of the entities to mark
    entities_dict : dict
        A dictionary mapping integer tags with a geometrical marker function {tag: marker(x)}

    Returns
    -------
    entities_tag array
    """
    marked_values = []
    marked_entities = []
    # number of non-ghosted entities
    num_entities_local = domain.topology.index_map(dim).size_local
    # Concatenate and sort the arrays based on indices
    for tag, location in entities_dict.items():
        entities = mesh.locate_entities(domain, dim, location)
        entities = entities[entities < num_entities_local]  # remove ghost entities
        marked_entities.append(entities)
        marked_values.append(np.full_like(entities, tag))
    marked_entities = np.hstack(marked_entities)
    marked_values = np.hstack(marked_values)
    sorted_entities = np.argsort(marked_entities)
    entities_tags = mesh.meshtags(
        domain, dim, marked_entities[sorted_entities], marked_values[sorted_entities]
    )
    return entities_tags


def half_left(x):
    return x[0] <= 0.5


def half_right(x):
    return x[0] >= 0.5


tdim = domain.topology.dim
cell_markers = mark_entities(domain, tdim, {1: half_left, 2: half_right})
print(cell_markers.values)
#output >> [2 2 2 2 2 2 1 2 2 1 2 2 1 1 2 2 1 1 2 2 1 1 2 1 1 2 1 1 1 1 1 1]

#Warning: When calling mesh.locate_entities for a cell or a facet, the geometrical marker function gets evaluated for all vertices of the cell/facet. The marker must therefore evaluate to True for all vertices to properly identify the entity.
'''


def convert_to_XDMF(filename):
    #reads a .msh file and writes mesh, subdomains, and boundary subdomains to XDMF files
    #check if input file i in msh
    if filename.split(sep = '.')[-1] != 'msh':
        raise TypeError('.msh file required')
    msh = meshio.read(filename)
    print("msh.cell_data_dict = ", msh.cell_data_dict)
    print("msh.cell_data_dict[gmsh:physical].keys() = ", msh.cell_data_dict["gmsh:physical"].keys())
    print('msh.cell_data_dict[gmsh:physical][line3]', msh.cell_data_dict["gmsh:physical"]['line3'])
    #print("msh.points = ", msh.points)
    print("msh.cells = ", msh.cells)
    
    #write mesh xdmf file
    #meshio.write(''.join(filename.split(sep ='.')[:-1]) + '_mesh.xdmf', meshio.Mesh(points = msh.points, cells = {'tetra': msh.cells['tetra']}))
    #meshio.write(''.join(filename.split(sep ='.')[:-1]) + '_mesh.xdmf', meshio.Mesh(points = msh.points, cells = {'line3': msh.cells['line3']}))
    meshio.write(''.join(filename.split(sep ='.')[:-1]) + '_mesh.xdmf', meshio.Mesh(points = msh.points, cells = {'line3': msh.cells_dict['line3']}))
    #write boundary subdomains xdmf file
    #meshio.write(''.join(filename.split(sep ='.')[:-1]) + '_boundary_subdomains.xdmf', meshio.Mesh(points = msh.points, cells = {'triangle6': msh.cells['triangle6']}, cell_data = {'triangle6': {'name_to_read' : msh.cell_data['triangle6']['gmsh:physical']}}))
    #meshio.write(''.join(filename.split(sep ='.')[:-1]) + '_boundary_subdomains.xdmf', meshio.Mesh(points = msh.points, cells = {'triangle6': msh.cells_dict['triangle6']}, cell_data = {'triangle6': {'name_to_read' : msh.cell_data['triangle6']['gmsh:physical']}}))
    #write subdomains xdmf file
    #meshio.write(''.join(filename.split(sep ='.')[:-1]) + '_subdomains.xdmf', meshio.Mesh(points = msh.points, cells = {'tetra' : msh.cells['tetra']}, cell_data = {'tetra' : {'name_to_read' : msh.cell_data['tetra']['gmsh:physical']}}))

def load_XDMF_files(filename):
    #create mesh from msh file
    mesh = dol.Mesh()
    with dol.XDMFFile(filename) as infile:
        infile.read(mesh)
        #create subdomain cell function
        mvc = dol.MeshValueCollection('size_t', mesh, 3)
        with dol.XDMFFile(''.join(filename.split(sep ='_')[:-1]) + '_subdomains.xdmf') as infile:
            infile.read(mvc, 'name_to_read')
        subdomain_marker = dol.cpp.mesh.MeshFunctionSizet(mesh, mvc)
        #create boundary subdomain facet function
        mvc = dol.MeshValueCollection('size_t', mesh, 2)
        with dol.XDMFFile(''.join(filename.split(sep ='_')[:-1]) + '_boundary_subdomains.xdmf') as infile:
            infile.read(mvc, 'name_to_read')
        boundary_marker = dol.cpp.mesh.MeshFunctionSizet(mesh, mvc)


def convert_msh2xdmf_nitish(mesh_name):
    msh = meshio.read(mesh_name+".msh")
    print(msh.cell_data_dict)
    
    #line_data = msh.cell_data_dict["gmsh:physical"]["line3"]
    #meshio.write(mesh_name+".xdmf", meshio.Mesh(points=msh.points, cells={"line3": msh.cells_dict["line3"]},
    line_data = msh.cell_data_dict["gmsh:physical"]["line"]
    meshio.write(mesh_name+".xdmf", meshio.Mesh(points=msh.points, cells={"line": msh.cells_dict["line"]}, cell_data={"dom_marker": [line_data]}) )
    
    #tri_data = msh.cell_data_dict["gmsh:physical"]["triangle6"]
    #meshio.write(mesh_name+"_surf.xdmf", meshio.Mesh(points=msh.points, cells={"triangle6": msh.cells_dict["triangle6"]}, cell_data={"bnd_marker": [tri_data]}
    tri_data = msh.cell_data_dict["gmsh:physical"]["triangle"]
    meshio.write(mesh_name+"_surf.xdmf", meshio.Mesh(points=msh.points, cells={"triangle": msh.cells_dict["triangle"]}, cell_data={"bnd_marker": [tri_data]} ) )

