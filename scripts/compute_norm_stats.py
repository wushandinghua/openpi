"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_dataset(data_config, config.model)
    # print(f"\n{dataset[100]}")  # (4, c, h, w)
    # print(f"\n{dataset[0]['observation.images.cam_exterior'].shape=}")  # (4, c, h, w)
    # print(f"\n{dataset[0]['observation.images.cam_wrist'].shape=}")  # (4, c, h, w)
    # print(f"{dataset[0]['observation.state'].shape=}")  # (6, c)
    # print(f"{dataset[0]['action'].shape=}\n")  # (64, c)
    # print(f"\n{dataset[0]['observation.images.cam_exterior'][0]}")
    # print(f"\n{dataset[0]['observation.images.cam_wrist'][0]}")
    # print(f"{dataset[0]['observation.state'][0]}")  # (6, c)
    # print(f"{dataset[0]['action'][0]}\n")  # (64, c)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    # print(f"\n{dataset[100]}")  # (4, c, h, w)
    # print(f"\n{dataset[0]['image']['base_0_rgb'].shape=}")  # (4, c, h, w)
    # print(f"\n{dataset[0]['image']['left_wrist_0_rgb'].shape=}")  # (4, c, h, w)
    # print(f"{dataset[0]['state'].shape=}")  # (6, c)
    # print(f"{dataset[0]['actions'].shape=}\n")  # (64, c)

    # print(f"\n{dataset[0]['image']['base_0_rgb'][0]}")  # (4, c, h, w)
    # print(f"\n{dataset[0]['image']['left_wrist_0_rgb'][0]}")  # (4, c, h, w)
    # print(f"{dataset[0]['state'][0]}")  # (6, c)
    # print(f"{dataset[0]['actions'][0]}\n")  # (64, c)
    return data_config, dataset


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config, dataset = create_dataset(config)

    num_frames = len(dataset)
    shuffle = False

    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=32,
        num_workers=8,
        shuffle=shuffle,
        num_batches=num_frames,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_frames, desc="Computing stats"):
        for key in keys:
            values = np.asarray(batch[key][0])
            stats[key].update(values.reshape(-1, values.shape[-1]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
