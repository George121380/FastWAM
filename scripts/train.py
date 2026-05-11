# Capture process start time as the very first line so "startup_seconds" reflects
# the full cost of imports + hydra + run_training before the trainer is ready.
import fastwam.utils.process_start  # noqa: F401  (side-effect: records start time)

import hydra
from omegaconf import DictConfig

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
