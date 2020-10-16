""" :noindex:
Setup.py file with generic info
"""
import os
from setuptools import setup
from setuptools import find_packages
from setuptools.command.test import test as testcommand


def read(fname) -> str:
    """From Wenjie Lei 2019:
    Utility function to read the README.md file.
    Used for the long_description.  It's nice, because now 1) we have a top levelx
    README.md file and 2) it's easier to type in the README.md file than to put a raw
    string in below ...
    """

    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except Exception as e:
        return "Can't open %s" % fname


class PyTest(testcommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.tests")]

    def initialize_options(self):
        testcommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest
        import sys
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name="lwsspy",
    description="LWSS Collection of Python Function",
    long_description="%s" % read("README.md"),
    version="0.0.1",
    author="Lucas Sawade",
    author_email="lsawade@princeton.edu",
    license='GNU Lesser General Public License, Version 3',
    keywords="collection, functions",
    url='https://github.com/lsawade/GCMT3D',
    packages=find_packages(exclude=['*.Notebooks', '*.notebooks.*',
                                    'notebooks.*', 'notebooks']),
    package_dir={"": "."},
    include_package_data=True,
    exclude_package_data={'': ['notebooks']},
    # package_data={'gcmt3d': ['data/download/resources/Fetchdata',
    #                          'data/download/resources/stations.txt',
    #                          'data/management/STATIONS'],
    # },
    install_requires=['numpy', 'matplotlib',  'obspy',
                      'PyYAML', 'h5py', 'mpi4py', 'matplotlib',
                      'pyasdf', 'autopep8'
                      ],
    tests_require=['pytest'],
    cmdclass={'tests': PyTest},
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        ("License :: OSI Approved "
         ":: GNU General Public License v3 or later (GPLv3+)"),
    ],
    # install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "docs": ["sphinx", "sphinx_rtd_theme"],
        "tests": ["pytest", "py"]
    },
    # entry_points={
    #     'console_scripts': [
    #         'catalog = gcmt3d.bins.catalog:main',
    #         'convert2asdf = gcmt3d.bins.convert_to_asdf:main',
    #         'convert2sac = gcmt3d.bins.convert_to_sac:main',
    #         'count-windows = gcmt3d.bins.count_overall_windows:main',
    #         'compute-stats = gcmt3d.bins.compute_stats:main',
    #         'copy-data-to-database = gcmt3d.bins.copy_data_to_database:main',
    #         'create-entry = gcmt3d.bins.create_entry:main',
    #         'create-entry-wj = gcmt3d.bins.create_entry_wenjie:main',
    #         'create-inversion-dicts = gcmt3d.bins.write_inversion_dicts:main',
    #         'create-gridsearch-dicts = gcmt3d.bins.write_grid_search_dicts:main',
    #         'create-path-files = gcmt3d.bins.create_path_files:main',
    #         'extract-station-info = gcmt3d.bins.extract_station_info:main',
    #         'filter-windows = gcmt3d.bins.filter_windows:main',
    #         'generate-stations-asdf = gcmt3d.bins.generate_stations_asdf:main',
    #         'get-cmt-id = gcmt3d.bins.get_cmt_id:main',
    #         'get-cmt-in-db = gcmt3d.bins.get_cmt_in_db:main',
    #         'inversion = gcmt3d.bins.inversion:main',
    #         'gridsearch = gcmt3d.bins.gridsearch:main',
    #         'process-asdf = gcmt3d.bins.process_asdf:main',
    #         'plot-section = gcmt3d.bins.plot_section:main',
    #         'plot-catalog-stats = gcmt3d.bins.plot_catalog:main',
    #         'plot-event = gcmt3d.bins.plot_event:main',
    #         'plot-event-summary = gcmt3d.bins.plot_event_summary:main',
    #         'plot-stats = gcmt3d.bins.plot_stats:main',
    #         'request-data = gcmt3d.bins.request_data:main',
    #         'select-windows = gcmt3d.bins.window_selection_asdf:main',
    #         'write-sources = gcmt3d.bins.write_sources:main',
    # ]}
)
