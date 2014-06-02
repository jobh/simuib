from __future__ import division
"""Utility functions"""

from dolfin import *

class PlotLine(object):
    """Line plot along x=[0,1] in a domain of any dimension. The mapping
    argument maps from the interval [0,1] to a xD coordinate (a list when
    dim>1)."""
    def __init__(self, mesh, mapping):
        hmin = MPI.min(mesh.hmin())
        self.mesh = UnitIntervalMesh(int(1.0/hmin))
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.F = {}
        self.mapping = mapping

    def __call__(self, expr, title):
        if not expr in self.F:
            self.F[expr] = Function(self.V)
        v = self.F[expr].vector()
        index_map = dof_to_vertex_map(self.V)
        for i,x in enumerate(self.mesh.coordinates()[index_map]):
            v[i] = expr(self.mapping(x))
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
