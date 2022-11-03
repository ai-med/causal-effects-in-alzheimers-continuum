# This file is part of Estimation of Causal Effects in the Alzheimer's Continuum (Causal-AD).
#
# Causal-AD is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Causal-AD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Causal-AD. If not, see <https://www.gnu.org/licenses/>.
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os.path
from pathlib import Path

from setuptools import find_packages
from setuptools import Extension, setup

from Cython.Build import cythonize


requirements = (
    'joblib',
    'matplotlib',
    'neuroCombat',
    'numpy',
    'pandas',
    'patsy',
    'pystan',
    'scikit-learn',
    'scipy',
    'seaborn',
    'tqdm',
)


# see https://github.com/stan-dev/pystan2/blob/1dd043db3c2618a9360a0f2ccbb57221634e5b08/pystan/model.py#L223
def get_stan_extension(stan_file: Path) -> Extension:
    import pystan.api
    import string
    import numpy as np

    stan_timestamp = os.path.getmtime(stan_file)

    pyx_file = stan_file.parent / f"_{stan_file.stem}.pyx"
    hpp_file = stan_file.with_suffix(".hpp")

    if pyx_file.exists() and hpp_file.exists():
        pyx_timestamp = min(os.path.getmtime(pyx_file), os.path.getmtime(hpp_file))
    else:
        pyx_timestamp = -1

    pystan_dir = Path(pystan.api.__file__).parent

    if pyx_timestamp < stan_timestamp:
        stanc_ret = pystan.api.stanc(
            file=str(stan_file),
            charset="utf-8",
            model_name=stan_file.stem,
            verbose=True,
            obfuscate_model_name=False,
        )
        if stanc_ret['status'] != 0:  # success == 0
            raise ValueError("stanc_ret is not a successfully returned "
                            "dictionary from stanc.")

        pyx_template_file = pystan_dir / 'stanfit4model.pyx'
        with open(pyx_template_file) as infile:
            s = infile.read()
            template = string.Template(s)
        with open(pyx_file, 'w') as outfile:
            s = template.safe_substitute(model_cppname=stan_file.stem)
            outfile.write(s)

        with open(hpp_file, 'w') as outfile:
            outfile.write(stanc_ret['cppcode'])

    stan_macros = [
        ("EIGEN_USE_MKL", None),
        ("EIGEN_USE_BLAS", None),
        ("BOOST_RESULT_OF_USE_TR1", None),
        ("BOOST_NO_DECLTYPE", None),
        ("BOOST_DISABLE_ASSERTS", None),
        ("BOOST_PHOENIX_NO_VARIADIC_EXPRESS", "ION"),
        ("NDEBUG", None),
    ]

    extra_compile_args = [
        "-O3",
        "-fvisibility-inlines-hidden",
        "-fmessage-length=0",
        "-ftree-vectorize",
        "-fstack-protector-strong",
        "-fno-plt",
        "-ffunction-sections",
        "-Wno-unused-function",
        "-Wno-uninitialized",
        "-Wno-sign-compare",
        "-Wno-ignored-attributes",
        "-std=c++1y",
        "-mtune=native",
    ]

    from numpy.distutils.__config__ import get_info
    blas_info = get_info('blas_mkl')

    include_dirs = [
        stan_file.parent,
        pystan_dir,
        pystan_dir / "stan" / "src",
        pystan_dir / "stan" / "lib" / "stan_math",
        pystan_dir / "stan" / "lib" / "stan_math" / "lib" / "eigen_3.3.3",
        pystan_dir / "stan" / "lib" / "stan_math" / "lib" / "boost_1.69.0",
        pystan_dir / "stan" / "lib" / "stan_math" / "lib" / "sundials_4.1.0" / "include",
        np.get_include(),
    ] + blas_info["include_dirs"]
    include_dirs = [str(d) for d in include_dirs]

    libraries = [
        "mkl_intel_lp64",
        "mkl_tbb_thread",
        "mkl_core",
        "tbb",
        "pthread",
        "m",
        "dl",
    ]

    extension = Extension(
        name=f"causalad.pystan_models._{stan_file.stem}",
        language="c++",
        sources=[str(pyx_file)],
        define_macros=stan_macros,
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=blas_info["library_dirs"],
        extra_compile_args=extra_compile_args,
    )
    return extension


def get_extensions():
    import pystan

    stan_dir = Path("causalad/pystan_models/stan_files")

    extensions = [
        "bpmf_reparameterized.stan",
        "bpmf_reparameterized_adni.stan",
        "logreg.stan",
        "ppca4.stan",
        "ppca4_adni.stan",
    ]

    pystan_dir = Path(pystan.__file__).parent
    cython_include_dirs = [str(pystan_dir)]

    build_extension = cythonize(
        [get_stan_extension(stan_dir / e) for e in extensions],
        include_path=cython_include_dirs,
    )

    return build_extension


setup(
    name='causalad',
    license='GPLv3+',
    description='Estimation of Causal Effects in the Presence of Unobserved Confounding in the Alzheimer\'s Continuum',
    author='Sebastian Pölsterl',
    author_email='sebastian.poelsterl@med.uni-muenchen.de',
    packages=find_packages(),
    ext_modules=get_extensions(),
    package_data={"causalad.adni": ["notebooks/*.ipynb"], "causalad.ukb": ["notebooks/*.ipynb"]},
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Utilities',
    ],
    version='0.2.0',
    python_requires='>=3.7',
    install_requires=requirements,
)
