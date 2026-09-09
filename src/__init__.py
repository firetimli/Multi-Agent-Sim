"""Multi-agent AutoML for user behaviour prediction -- Version 0.

Trust boundary:
  * TRUSTED (this package): dataset loading, splitting, feature implementations,
    model implementations, metrics, budget, experiment execution.
  * AGENT-CONTROLLED: the JSON proposals and critiques produced by src.agents,
    written only under generated/ and runs/.
"""
__version__ = "0.1.0"
