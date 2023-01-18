from collections.abc import MutableMapping
from typing import Union, Dict, Optional
from monai.transforms import Transform

from . import filter_columns


class GroupCols(Transform):
    """Transform to resize image by`scale_factor`"""

    def __init__(
        self,
        groups: Dict[str, Optional[Union[str, Dict]]],
    ):
        """
        Parameters
        ----------
        groups: Dict[Optional[Union[str, Dict]]]
            Dictionary where keys are column group names (which become batch keys)
            and values are either:
            - a dictionary containing the kwargs to be used in a col to `filter_cols`
            - a string, to use a single column in that group
            - `None`, to use a single column, with the same name as the key
        """
        super().__init__()
        self.groups = groups

    def _make_group(self, k, v, row):
        if v is None:
            return {k: row[k]}
        if isinstance(v, str):
            return {v: row[k]}

        assert isinstance(v, MutableMapping)

        return {k: filter_columns(row, **v)}

    def __call__(self, row):
        res = {}

        for k, v in self.groups.items():
            res.update(self._make_group(k, v, row))

        return res
