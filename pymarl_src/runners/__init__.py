REGISTRY = {}

from .episode_runner_maca import EpisodeRunner

REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner

REGISTRY["parallel"] = ParallelRunner
