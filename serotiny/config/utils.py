import collections
import importlib
import inspect
import json
import os
import uuid
import base64
import hashlib
import yaml
from Crypto.Cipher import AES
from Crypto import Random
from getpass import getpass
from omegaconf import OmegaConf
from omegaconf._utils import split_key


def get_obj_from_path(obj_path: str):
    """Given a class/function/variable path as a string (e.g.
    some.module.ClassName), retrieve the class/function/variable."""
    obj_path = obj_path.split(".")
    module = ".".join(obj_path[:-1])
    name = obj_path[-1]
    return getattr(importlib.import_module(module), name)


def _create_config(obj, obj_path=None, add_partial=False):

    if obj_path is None:
        if hasattr(obj, "__name__"):
            obj_path = obj.__module__ + "." + obj.__name__
        else:
            obj_path = obj.__module__ + "." + obj.__class__.__name__

    if hasattr(obj, "__init__"):
        sig = inspect.getfullargspec(obj.__init__)
    else:
        sig = inspect.getfullargspec(obj)

    args_dict = dict()
    if add_partial:
        args_dict["_partial_"] = True
    args_dict["_target_"] = obj_path

    args = sig.args
    if sig.defaults is not None:
        while len(args) > len(sig.defaults):
            args_dict[args.pop(0)] = None

        assert len(args) == len(sig.defaults)
        for arg, default_value in zip(args, sig.defaults):
            args_dict[arg] = default_value

    kwargs = sig.kwonlyargs
    if sig.kwonlydefaults is not None:
        while len(kwargs) > len(sig.kwonlydefaults):
            args_dict[kwargs.pop(0)] = None

        assert len(args) == len(sig.defaults)
        for arg, default_value in zip(kwargs, sig.kwonlydefaults):
            args_dict[arg] = default_value

    for arg in sig.args + sig.kwonlyargs:
        if arg not in args_dict:
            args_dict[arg] = None

    if "self" in args_dict:
        del args_dict["self"]

    for arg, default in args_dict.items():
        if isinstance(default, tuple):
            args_dict[arg] = list(default)
        elif OmegaConf.is_config(default):
            try:
                args_dict[arg] = _create_config(
                    default, add_partial=inspect.isclass(default)
                )
            except Exception:
                args_dict[arg] = f"Unable to give default value of {type(default)}"

    return args_dict


def create_config(obj_path):
    obj = get_obj_from_path(obj_path)
    config = _create_config(obj, obj_path)
    return OmegaConf.to_yaml(OmegaConf.create(config))


def _maybe_int(value):
    return int(value) if value.isdigit() else value


def add_config(obj_path, dest):
    dest_key = None
    if ":" in dest:
        assert len([c for c in dest if c == ":"]) == 1
        dest, dest_key = dest.split(":")

    obj = get_obj_from_path(obj_path)
    config = _create_config(obj, obj_path)

    if dest_key is not None:
        dest_config = OmegaConf.load(dest)
        dest_path = split_key(dest_key)
        for step in dest_path[::-1]:
            config = {step: config}

        dest_config = OmegaConf.merge(dest_config, config)

    else:
        dest_config = config

    prompt_str_suffix = "" if dest_key is None else f":{dest_key}"
    answer = input(
        f"This is going to overwrite {dest}{prompt_str_suffix}" "\nContinue? [Y/n] "
    )
    if answer is None or answer.lower().strip() in ["", "yes", "y"]:
        with open(dest, "w") as f:
            f.write(OmegaConf.to_yaml(dest_config))
    else:
        print("Not writing the result. The resulting config would have been:\n")
        print(OmegaConf.to_yaml(dest_config))


def deep_merge(template, merge):
    """Recursive dict merge, combines values that are lists.

    This mutates template - the contents of merge are added to template
    (which is also returned). If you want to keep template you could call it
    like deep_merge(dict(template), merge)
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


BLOCK_SIZE = 16


def _pad(s):
    pad_size = BLOCK_SIZE - len(s) % BLOCK_SIZE
    pad_char = chr(pad_size)
    return s + pad_size * pad_char


def _unpad(s):
    return s[: -ord(s[len(s) - 1 :])]


def _get_priv_key():
    password = os.getenv("SEROTINY_PASSWORD")
    if password is None:
        password = getpass("Enter encryption password: ")

    return hashlib.sha256(password.encode("utf-8")).digest()


def encrypt(raw):
    private_key = _get_priv_key()
    raw = _pad(raw)
    iv = Random.new().read(AES.block_size)
    cipher = AES.new(private_key, AES.MODE_CBC, iv)
    return base64.b64encode(iv + cipher.encrypt(raw.encode())).decode()


def decrypt(enc):
    private_key = _get_priv_key()
    enc = base64.b64decode(enc)
    iv = enc[:16]
    cipher = AES.new(private_key, AES.MODE_CBC, iv)
    return _unpad(cipher.decrypt(enc[16:])).decode()


OmegaConf.register_new_resolver("uuid", lambda: str(uuid.uuid4()))
OmegaConf.register_new_resolver("decrypt", decrypt)
