import wandb
from omegaconf import OmegaConf

class WandbLogger:
    def __init__(self, cfg):
        self.cfg = cfg
        wandb.init(project=cfg.logger.project_name)
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    def log_metric(self, metric_dict):
        wandb.log(metric_dict)
