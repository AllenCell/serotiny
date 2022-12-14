import collections
import json
import os

import yaml


def deep_merge(template, merge):
    """Recursive dict merge, combines values that are lists.

    This mutates template - the contents of merge are added to template
    (which is also returned).
    If you want to keep template you could call it like
    deep_merge(dict(template), merge)
    """
    for k, v in merge.items():
        if (
            k in template
            and isinstance(template[k], dict)
            and isinstance(merge[k], collections.Mapping)
        ):
            deep_merge(template[k], merge[k])
        # Removed option to extend the list if there are existing elements:
        # elif k in template and isinstance(template[k], list) and isinstance(v, list):
        #     template[k].extend(v)
        else:
            template[k] = merge[k]
    return template


def merge_config(
    template_path="",
    merge="",
    output_path="",
):
    template_extension = os.path.splitext(template_path)[-1]
    with open(template_path, "r") as template_file:
        if template_extension == ".yaml":
            template = yaml.load(template_file)
        elif template_extension == ".json":
            template = json.load(template_file)
        else:
            raise Exception(f"format {template_extension} not supported")

    output_extension = os.path.splitext(output_path)[-1]
    with open(output_path, "w") as output_file:
        updated = deep_merge(template, merge)

        if output_extension == ".json":
            json.dump(updated, output_file, indent=4)
        elif output_extension == ".yaml":
            yaml.dump(updated, output_file)
        else:
            raise Exception(f"format {output_extension} not supported")
