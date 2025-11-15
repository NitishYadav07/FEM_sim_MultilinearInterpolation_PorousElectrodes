from dolfinx.fem import Constant, Function, functionspace, locate_dofs_geometrical, locate_dofs_topological, DirichletBC, dirichletbc, Expression
from functions_NPN import mark_facets
from dolfinx.mesh import locate_entities_boundary

#'''
#https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html
#create a general boundary condition class
# NOTE: index (0 or 1) KEYWORD TELLS WITH PART OF FUNCTION SPACE SPLIT - ME.sub(0) or ME.sub(1)
class BoundaryCondition():
    def __init__(self, type, marker, facet_tag, values, ME_, mesh, index):
        self._type = type
        fdim = mesh.topology.dim - 1
        V = ME_.sub(index)
        Q, _ = V.collapse()
        if type == "Dirichlet":
            u_D = Function(Q)
            u_D.interpolate(values)
            facets = locate_entities_boundary(mesh, fdim, values)
            #facets = facet_tag.find(marker)
            dofs = locate_dofs_topological((V, Q), fdim, facets)
            self._bc = dirichletbc(u_D, dofs, V)
        elif type == "Neumann":
                self._bc = inner(values, v) * ds(marker)
        elif type == "Robin":
            self._bc = values[0] * inner(u-values[1], v)* ds(marker)
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type



