"""
Match a query image to gallery identities with a Prototypical encoder.
Each identity is represented by the mean embedding of its support images.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.helper.executor import Executor
from capsules.Fewshot.src.classes.runtime import FewShotRuntime


class Prototypical(FewShotRuntime):
    """Prototype matching with a trained Prototypical encoder."""

    matching = "prototypical"


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
