import hydra
from omegaconf import DictConfig
from pytorch_tabular import TabularModel

from utils import covtype_data_loader


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    data_config = hydra.utils.instantiate(cfg.data.config)
    model_config = hydra.utils.instantiate(cfg.model.estimator)
    optimizer_config = hydra.utils.instantiate(cfg.optimizer)
    trainer_config = hydra.utils.instantiate(cfg.trainer)

    train, valid, test = covtype_data_loader()

    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )

    model.fit(train, valid)
    model.evaluate(valid)
    model.evaluate(test)


if __name__ == "__main__":
    main()
