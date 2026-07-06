from logs.logging import init_wandb, finish_wandb
from configs.default import get_config
from models.train import train
from data.dataset import TeethDataset, StagesDataset
from data.image_gen import generate_processed_images

import os
import torch
import torch.multiprocessing as mp
import numpy as np


def main(config=None, run=None):
    gpu_ids = config.get("gpu_ids", "all")

    seed = int(config.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if config["model_task"] == "stages":
        img_dir = "data/stages_images/"
        idx_dataset = StagesDataset(
            ann_file="data/" + config["ann_file"],
            img_dir=img_dir,
            transform=None,
            config=config,
        )

    else:
        img_dir = os.path.join("data/preprocessed", config["image_processing"])

        if not os.path.exists(img_dir) or config.get("recreate_dataset", False):
            print(f"Generating processed image folders in {img_dir}...")
            generate_processed_images(config)

        idx_dataset = TeethDataset(
            ann_file="data/" + config["ann_file"],
            img_dir=img_dir,
            transform=None,
            config=config,
        )

    if torch.cuda.is_available():
        if gpu_ids == "all":
            world_size = torch.cuda.device_count()
        else:
            if isinstance(gpu_ids, str):
                gpu_list = [x.strip() for x in gpu_ids.split(",") if x.strip() != ""]
            else:
                gpu_list = list(gpu_ids)

            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))
            world_size = len(gpu_list)

        print(f"CUDA available. Using {world_size} GPU(s).")

    else:
        world_size = 1
        print("CUDA not available. Using CPU.")

    print("Starting training...")

    # Important:
    # Do not use multiprocessing when there is only CPU or one GPU.
    # This avoids Windows/W&B multiprocessing problems.
    if world_size == 1:
        train(0, 1, idx_dataset, config, run)
    else:
        mp.spawn(
            train,
            args=(world_size, idx_dataset, config, run),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    config = get_config()

    run = init_wandb(
        project_name=config["project_name"],
        config_dict=config,
        run_name=config.get("run_name", None),
    )

    config.update(run.config)

    try:
        main(config, run)
    except Exception as e:
        print(f"An error occurred: {e}")
        try:
            run.log({"error": str(e)})
        except Exception:
            pass
        raise e
    finally:
        finish_wandb(run)
