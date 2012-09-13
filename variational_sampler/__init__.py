from .variational_sampler import (VariationalSampler,
                                  VariationalFit,
                                  ImportanceSampler,
                                  ImportanceFit)
from .gaussian import Gaussian
from .sampling import Sample

from numpy.testing import Tester
test = Tester().test
