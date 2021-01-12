import json


def require(arg, message):
    if arg is None:
        raise Exception(message)


def unjson(j):
    if j is not None:
        return json.loads(j)
