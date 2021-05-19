import re

def filter_columns(cols_to_filter, regex=None, startswith=None, endswith=None,
                   contains=None, excludes=None):
    if regex is not None:
        return [col for col in cols_to_filter if re.match(regex, col)]

    keep = [True] * len(cols_to_filter)
    for i in range(len(cols_to_filter)):
        if startswith is not None:
            keep[i] &= str(cols_to_filter[i]).startswith(startswith)
        if endswith is not None:
            keep[i] &= str(cols_to_filter[i]).endswith(endswith)
        if contains is not None:
            keep[i] &= (contains in str(cols_to_filter[i]))
        if excludes is not None:
            keep[i] &= (excludes not in str(cols_to_filter[i]))

    return [
        col for col, keep_col in zip(cols_to_filter, keep)
        if keep_col
    ]
