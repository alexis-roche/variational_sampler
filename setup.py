#!/usr/bin/env python
version = '0.1dev'


def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True,
    )
    config.add_subpackage('variational_sampler')
    return config


def setup_package():

    from numpy.distutils.core import setup

    setup(
        configuration=configuration,
        name='scikits.variational_sampler',
        version=version,
        maintainer='Alexis Roche',
        maintainer_email='alexis.roche@gmail.com',
        description='Gaussian approximation via variational sampling',
        url='http://www.scipy.org',
        license='BSD',
        #install_requires=['numpy >= 1.0.2',],
    )

    return

if __name__ == '__main__':
    setup_package()
