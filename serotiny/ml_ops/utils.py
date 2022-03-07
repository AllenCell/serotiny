import os
import sys
from pathlib import Path

def get_serotiny_project():
    try:
        if (Path(os.getcwd()) / ".serotiny").exists():
            with open(".serotiny", "r") as f:
                project_name = f.read().strip()
            return project_name
    except:
        pass

    return "serotiny"


def flatten_config(cfg):
    import pandas as pd
    conf = (
        pd.json_normalize(cfg, sep="/")
        .to_dict(orient="records")[0]
    )
    keys = list(conf.keys())

    for k in keys:
        try:
            sub_conf = flatten_config(conf[k])
            conf.update({f"{k}/{_k}" for k,v in sub_conf.items()})
            del conf[k]
            continue
        except:
            pass

        if isinstance(conf[k], list):
            for i, el in enumerate(conf[k]):
                try:
                    sub_conf = flatten_config(el)
                    conf.update({f"{k}/{_k}" for k,v in sub_conf.items()})
                except Exception as e:
                    conf[f"{k}/{i}"] = el
            del conf[k]

    return (
        pd.json_normalize(conf, sep="/")
        .to_dict(orient="records")[0]
    )
