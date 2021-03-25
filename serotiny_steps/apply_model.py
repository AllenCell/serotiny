from typing import Union, Optional, Dict, Tuple
from pathlib import Path
import fire

import torch
import pandas as pd

from serotiny.datamodules import FolderDatamodule
from serotiny.models.zoo import get_model
from serotiny.io.image import tiff_writer

def store_batch(batch, result, target_folder, mode, channels):
    target_folder = Path(target_folder)
    if mode == "csv":
        df = pd.DataFrame(result)
        df["basename"] = batch["basename"]
        fpath = (target_folder / "results.csv")
        if fpath.exists():
            df.to_csv(fpath, header=False, mode="a")
        else:
            df.to_csv(fpath, header=False, mode="a")
    else:
        for field_name, tensors in result.items():
            for basename, tensor in zip(batch["basename"], tensors):
                tiff_writer(
                    tensor,
                    target_folder / f"{basename}_{field_name}.pt",
                    channels
                )

def apply_model(
    model_path: str,
    data_path: Union[str, Path],
    target_path: Union[str, Path],
    batch_size_dl: int,
    num_workers_dl: int,
    store_mode: str,
    target_channels: Optional[str] = None,
    gpu_id: Optional[int] = None,
    model_root: Optional[Union[str, Path]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    loader_dict: Optional[Dict[str, Tuple[str, Dict[str, Dict]]]] = None,
):

    if store_mode not in ("csv", "file"):
        raise ValueError("`store_mode` must be either 'csv' or 'file'")

    data_path = Path(data_path)
    model = get_model(model_path, model_root)
    model.eval()

    dm = FolderDatamodule(
        batch_size=batch_size_dl,
        num_workers=num_workers_dl,
        path=data_path,
        train_frac=1.0,
        loader_dict=loader_dict,
        return_paths=True,
    )

    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        model = model.cuda()

    with torch.no_grad():
        for batch in dm.train_dataloader():
            if gpu_id is not None:
                batch = batch.cuda()

            result = model(batch)
            if gpu_id is not None:
                batch = batch.cpu()
                result = result.cpu()

            store_batch(batch, result, target_path, store_mode, channels)
