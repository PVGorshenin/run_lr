import re
import setuptools

(__version__,) = re.findall("__version__.*\s*=\s*[']([^']+)[']",
                            open('run_lr/__init__.py').read())

setuptools.setup(
    name="run_lr",
    version=__version__,
    packages=setuptools.find_packages(),
    python_requires="<3.9.0",
    install_requires=[
        "numpy==1.18.0",
        "pandas==0.25.3",
        "scikit-learn==0.21.2",
        "tqdm==4.41.1",
    ],
)
