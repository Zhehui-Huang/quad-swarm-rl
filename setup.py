# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='swarm_rl',  # Required

    version='1.0.0',  # Required

    description='Quadrotor Gym Envs',  # Optional

    long_description=long_description,  # Optional

    long_description_content_type='text/markdown',  # Optional

    # This should be a valid link to your project's main homepage.
    #
    # This field corresponds to the "Home-Page" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url='https://github.com/Zhehui-Huang',  # Optional

    # This should be your name or the name of the organization which owns the
    # project.
    author='Zhehui Huang',  # Optional

    # This should be a valid email address corresponding to the author listed
    # above.
    author_email='zhehuihu@usc.edu',  # Optional

    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a string of words separated by whitespace, not a list.
    keywords='Reinforcement Learning for Quadrotors',  # Optional

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=find_packages(where='.'),  # Required

    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. If you
    # do not support Python 2, you can simplify this to '>=3.5' or similar, see
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires='>=3.8.6',

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'pytest', 'numpy<2.0', 'pyglet<=1.5.23', 'matplotlib>3', 'gym>=0.17', 'transforms3d', 'noise', 'tqdm',
        'bezier', 'numba', 'scipy', 'sample-factory', 'plotly<=5.13.0', 'attrdict==2.0.1',
        'pandas==2.0.1', 'gymnasium==0.28.1'
    ],
)
