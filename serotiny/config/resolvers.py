import os
import uuid
import base64
import hashlib
from omegaconf import OmegaConf
from Crypto.Cipher import AES
from Crypto import Random
from getpass import getpass


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


def _encrypt(raw):
    private_key = _get_priv_key()
    raw = _pad(raw)
    iv = Random.new().read(AES.block_size)
    cipher = AES.new(private_key, AES.MODE_CBC, iv)
    return base64.b64encode(iv + cipher.encrypt(raw.encode())).decode()


def _decrypt(enc):
    private_key = _get_priv_key()
    enc = base64.b64decode(enc)
    iv = enc[:16]
    cipher = AES.new(private_key, AES.MODE_CBC, iv)
    return _unpad(cipher.decrypt(enc[16:])).decode()


OmegaConf.register_new_resolver("uuid", lambda: str(uuid.uuid4()))
OmegaConf.register_new_resolver("decrypt", _decrypt)
