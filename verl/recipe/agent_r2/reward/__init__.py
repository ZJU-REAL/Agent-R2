"""reward/ -- LLM-as-a-judge reward for sandbox output files."""
from .mllm_reward import compute_reward

__all__ = ["compute_reward"]
