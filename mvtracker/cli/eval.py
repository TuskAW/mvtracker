import hydra
from omegaconf import DictConfig

from mvtracker.cli.train import main as train_main


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval")
def main(cfg: DictConfig):
    train_main(cfg)


if __name__ == "__main__":
    main()
