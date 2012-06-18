from dolfin import *

class PlotLine(object):
    """Line plot along x=[0,1] in 2D. The mapping argument maps from the
    interval [0,1] to a 2D coordinate (a two-component list)."""
    def __init__(self, mesh, mapping):
        hmin = MPI.min(mesh.hmin())
        self.mesh = UnitInterval(int(1.0/hmin))
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.F = {}
        self.mapping = mapping

    def __call__(self, expr, title):
        if not expr in self.F:
            self.F[expr] = Function(self.V)
        P1expr = project(expr, FunctionSpace(expr.function_space().mesh(), "CG", 1))
        v = self.F[expr].vector()
        for i,x in enumerate(self.mesh.coordinates()):
            v[i] = P1expr(self.mapping(x))
        plot(self.F[expr], title=title)

class DeltaFunction(object):
    """Unit area delta function in discrete space V"""
    def __init__(self, mesh):
        self.V = FunctionSpace(mesh, "CG", 1)

    def __call__(self, pt):
        q = Function(self.V)
        v = q.vector()
        PointSource(self.V, pt).apply(v)
        v[:] = v.array()/assemble(q*dx)
        return q
