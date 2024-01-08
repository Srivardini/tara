from setuptools import setup, find_packages

setup(
        name='tara',
        version='0.0.1',
        description="Time series Analysis and Reduction Algorithm",
        url='https://github.com/Srivardini/tara',
        author='Srivardini Ayyappan',
        author_email='srivardini.ayyappan@students.mq.edu.au',
        license='BSD 2-clause',
        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        install_requires=['matplotlib', 'astropy', 'photutils',
                          'numpy', 'astroalign','aafitrans'],
        include_package_data=True,
        package_data={'': ['tara/data/*']},
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.9',
        ],
)
