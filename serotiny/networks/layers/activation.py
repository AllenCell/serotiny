from torch import nn


def activation_map(activation):
    """
    Map from strings to activation functions
    """
    if activation is None or activation.lower() == "none":
        return nn.Sequential()

    elif activation.lower() == "relu":
        return nn.ReLU(inplace=True)

    elif activation.lower() == "prelu":
        return nn.PReLU()

    elif activation.lower() == "sigmoid":
        return nn.Sigmoid()

    elif activation.lower() == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)

    elif activation.lower() == "softplus":
        return nn.Softplus()
