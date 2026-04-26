from .base import MPPT
from .pno  import PnO
from .pso  import PSO
from .foa  import FOA
from .ahfoa import AHFOA

ALGORITHMS = {
    "P&O":   PnO,
    "PSO":   PSO,
    "FOA":   FOA,
    "AH-FOA": AHFOA,
}
