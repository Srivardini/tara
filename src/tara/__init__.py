__version__ = "0.0.1"
__author__ = 'Srivardini Ayyappan'
__credits__ = 'Australian Astronomical Optics, Macquarie University'

from .core import tara
from . import utils
from pathlib import Path

data_dir = Path(__file__).parent.joinpath('data')