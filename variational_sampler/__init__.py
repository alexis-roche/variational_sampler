from .variational_sampler import (VariationalSampler,
                                  VariationalFit,
                                  DirectSampler,
                                  DirectFit)
from .gaussian import Gaussian
from .sampling import Sample

from numpy.testing import Tester
test = Tester().test
