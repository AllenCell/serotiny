import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def calculate_blur(X):
    """
    https://github.com/tolstikhin/wae/blob/master/wae.py -- line 344
    Keep track of the min blurriness / batch for each test loop
    """
    # RGB case -- convert to greyscale
    if X.size(1) == 3:
        X = torch.mean(X, 1, keepdim=True)

    lap_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    lap_filter = lap_filter.reshape([1, 1, 3, 3])
    lap_filter = Variable(torch.from_numpy(lap_filter).float())

    lap_filter = lap_filter.cuda()

    # valid padding (i.e., no padding)
    conv = F.conv2d(X, lap_filter, padding=0, stride=1)

    # smoothness is the variance of the convolved image
    var = torch.var(conv)

    return var
