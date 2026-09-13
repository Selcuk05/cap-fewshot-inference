"""
Match a query image to gallery identities with a Siamese encoder.
Each identity score is the best similarity against that identity's gallery embeddings.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.helper.executor import Executor
from capsules.Fewshot.src.classes.runtime import FewShotRuntime


class Siamese(FewShotRuntime):
    """Nearest-neighbor identity matching with a trained Siamese encoder."""

    matching = "siamese"


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
