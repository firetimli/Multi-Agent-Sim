"""Agent implementations (trusted, human-written plumbing).

The agents themselves have no filesystem or shell access to the repository: they
receive a rendered prompt string and return JSON, which the orchestrator writes
under runs/ and generated/. This is what makes "agents cannot modify evaluation
code" a structural property rather than an instruction.
"""
from .backend import Backend, BackendError, CodexCLIBackend, MockBackend, make_backend
from .critic_agent import CriticAgent
from .modeling_agent import AgentResult, ModelingAgent

__all__ = [
    "Backend", "BackendError", "CodexCLIBackend", "MockBackend", "make_backend",
    "CriticAgent", "ModelingAgent", "AgentResult",
]
