from __future__ import division
"""Common parameters for the Buckley-Leverett solvers"""

from dolfin import *

set_log_level(WARNING)
#parameters["form_compiler"]["optimize"] = True
#parameters["form_compiler"]["cpp_optimize"] = True
#parameters["form_compiler"]["representation"] = 'quadrature'
#parameters["form_compiler"]["quadrature_degree"] = 1

# Which plots to draw; any of 'upslSLdm' (m:mesh, d:diff, l:line, uppercase:analytical)
do_plot = 'sSlLpd'
#do_plot = ''

# Mesh
mesh = UnitInterval(128)  	# Buckley-Leverett 1D (on line interval)
#mesh = UnitCircle(16)   	# Buckley-Leverett 2D (on circle, divided by 4)

# End time
T = 0.15
