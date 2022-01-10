from deeptime.base import Dataset
from typing import Optional, List
import numpy as np


class ConcatDataset(Dataset):
    r""" Concatenates existing datasets.
    Parameters
    ----------
    datasets : list of dataset
        Datasets to concatenate.
    """

    def __init__(self, datasets: List[Dataset]):
        self._lengths = [len(ds) for ds in datasets]
        self._cumlen = np.cumsum(self._lengths)
        self._datasets = datasets

    def setflags(self, write=True):
        for ds in self._datasets:
            ds.setflags(write=write)

    @property
    def subsets(self):
        r""" Returns the list of datasets this concat dataset is composed of.
        :type: list of dataset
        """
        return self._datasets

    def _dataset_index(self, ix):
        from bisect import bisect_right

        ds_index = bisect_right(self._cumlen, ix)
        item_index = ix if ds_index == 0 else ix - self._cumlen[ds_index - 1]
        return ds_index, item_index

    def __getitem__(self, ix):
        ds_index, item_index = self._dataset_index(ix)
        return self._datasets[ds_index][item_index]

    def __len__(self):
        return self._cumlen[-1]


class TimeLaggedDataset(Dataset):
    r""" High-level container for time-lagged time-series data.
    This can be used together with pytorch data tools, i.e., data loaders and other utilities.
    Parameters
    ----------
    data : (T, n) ndarray
        The data which is wrapped into a dataset
    data_lagged : (T, m) ndarray
        Corresponding timelagged data. Must be of same length.
    See Also
    --------
    TimeLaggedConcatDataset, TrajectoryDataset, TrajectoriesDataset
    """

    def __init__(self, data, data_lagged, ids, condition, x_label, xhat_label):
        assert len(data) == len(
            data_lagged
        ), f"Length of trajectory for data and data_lagged does not match ({len(data)} != {len(data_lagged)})"
        self._data = data
        self._data_lagged = data_lagged
        self._ids = ids
        self.x_label = x_label
        self.xhat_label = xhat_label
        self._condition = condition

    def setflags(self, write=True):
        self._data.setflags(write=write)
        self._data_lagged.setflags(write=write)

    def astype(self, dtype):
        r""" Sets the datatype of contained arrays and returns a new instance of TimeLaggedDataset.
        Parameters
        ----------
        dtype
            The new dtype.
        Returns
        -------
        converted_ds : TimeLaggedDataset
            The dataset with converted dtype.
        """
        return TimeLaggedDataset(
            self._data.astype(dtype), self._data_lagged.astype(dtype)
        )

    @property
    def data(self) -> np.ndarray:
        r""" Instantaneous data. """
        return self._data

    @property
    def data_lagged(self) -> np.ndarray:
        r""" Time-lagged data. """
        return self._data_lagged

    @property
    def condition(self) -> np.ndarray:
        r""" Time-lagged data. """
        return self._condition

    @property
    def ids(self) -> np.ndarray:
        r""" Time-lagged data. """
        return self._ids

    def __getitem__(self, item):
        if self._condition is not None:
            return {
                self.x_label: self._data[item],
                self.xhat_label: self._data_lagged[item],
                "CellId": self._ids[item],
                "condition": self._condition[item],
            }
        else:
            return {
                self.x_label: self._data[item],
                self.xhat_label: self._data_lagged[item],
                "CellId": self._ids[item],
            }       

    def __len__(self):
        return len(self._data)


class TimeLaggedConcatDataset(ConcatDataset):
    r""" Specialization of the :class:`ConcatDataset` which uses that all subsets are time lagged datasets, offering
    fancy and more efficient slicing / getting items.
    Parameters
    ----------
    datasets : list of TimeLaggedDataset
        The input datasets
    """

    def __init__(self, datasets: List[TimeLaggedDataset]):
        assert all(isinstance(x, TimeLaggedDataset) for x in datasets)
        super().__init__(datasets)

    @staticmethod
    def _compute_overlap(stride, traj_len, skip):
        r""" Given two trajectories :math:`T_1` and :math:`T_2`, this function calculates for the first trajectory
        an overlap, i.e., a skip parameter for :math:`T_2` such that the trajectory fragments
        :math:`T_1` and :math:`T_2` appear as one under the given stride.
        :param stride: the (global) stride parameter
        :param traj_len: length of T_1
        :param skip: skip of T_1
        :return: skip of T_2
        Notes
        -----
        Idea for deriving the formula: It is
        .. code::
            K = ((traj_len - skip - 1) // stride + 1) = #(data points in trajectory of length (traj_len - skip)).
        Therefore, the first point's position that is not contained in :math:`T_1` anymore is given by
        .. code::
            pos = skip + s * K.
        Thus the needed skip of :math:`T_2` such that the same stride parameter makes :math:`T_1` and :math:`T_2`
        "look as one" is
        .. code::
            overlap = pos - traj_len.
        """
        return stride * ((traj_len - skip - 1) // stride + 1) - traj_len + skip

    def __getitem__(self, ix):
        if isinstance(ix, slice):
            xs, ys = [], []
            end_ds, end_ix = self._dataset_index(
                ix.stop if ix.stop is not None else len(self)
            )
            start_ds, start = self._dataset_index(
                ix.start if ix.start is not None else 0
            )
            stride = ix.step if ix.step is not None else 1
            for ds in range(start_ds, end_ds + 1):
                stop_ix = self._lengths[ds] if ds != end_ds else end_ix

                if stop_ix > start and ds < len(self._lengths):
                    local_slice = slice(start, stop_ix, stride)
                    xs.append(self._datasets[ds].data[local_slice])
                    ys.append(self._datasets[ds].data_lagged[local_slice])
                    start = self._compute_overlap(stride, self._lengths[ds], start)
            return np.concatenate(xs), np.concatenate(ys)
        else:
            return super().__getitem__(ix)


class TrajectoryDataset(TimeLaggedDataset):
    r"""Creates a trajectory dataset from a single trajectory by applying a lagtime.
    Parameters
    ----------
    lagtime : int
        Lagtime, must be positive. The effective size of the dataset reduces by the selected lagtime.
    trajectory : (T, d) ndarray
        Trajectory with T frames in d dimensions.
    Raises
    ------
    AssertionError
        If lagtime is not positive or trajectory is too short for lagtime.
    """

    def __init__(self, lagtime, trajectory, ids, condition, x_label, xhat_label):
        assert len(trajectory) > lagtime, "Not enough data to satisfy lagtime"
        if lagtime == 0:
            super().__init__(trajectory, trajectory, ids, condition, x_label, xhat_label)
        elif condition is not None:
            super().__init__(
                trajectory[:-lagtime],
                trajectory[lagtime:],
                ids[:-lagtime],
                condition[:-lagtime],
                x_label,
                xhat_label,
            )
        else:
            super().__init__(
                trajectory[:-lagtime],
                trajectory[lagtime:],
                ids[:-lagtime],
                None,
                x_label,
                xhat_label,
            )
        self._trajectory = trajectory
        self._lagtime = lagtime

    @property
    def lagtime(self):
        return self._lagtime

    @property
    def trajectory(self):
        return self._trajectory

    @staticmethod
    def from_trajectories(lagtime, ids, condition, x_label, xhat_label, data: List[np.ndarray]):
        r""" Creates a time series dataset from multiples trajectories by applying a lagtime.
        Parameters
        ----------
        lagtime : int
            Lagtime, must be positive. The effective size of the dataset reduces by the selected lagtime.
        data : list of ndarray
            List of trajectories.
        Returns
        -------
        dataset : TrajectoriesDataset
            Concatenation of timeseries datasets.
        Raises
        ------
        AssertionError
            If data is empty, lagtime is not positive,
            the shapes do not match, or lagtime is too long for any of the trajectories.
        """
        return TrajectoriesDataset.from_numpy(lagtime, ids, condition, x_label, xhat_label, data)


class TrajectoriesDataset(TimeLaggedConcatDataset):
    r""" Dataset composed of multiple trajectories.
    Parameters
    ----------
    data : list of TrajectoryDataset
        The trajectories in form of trajectory datasets.
    See Also
    --------
    TrajectoryDataset.from_trajectories
        Method to create a TrajectoriesDataset from multiple raw data trajectories.
    """

    def __init__(self, data: List[TrajectoryDataset]):
        assert len(data) > 0, "List of data should not be empty."
        assert all(x.lagtime == data[0].lagtime for x in data), "Lagtime must agree"
        super().__init__(data)

    @staticmethod
    def from_numpy(lagtime, ids, condition, x_label, xhat_label, data: List[np.ndarray]):
        r""" Creates a time series dataset from multiples trajectories by applying a lagtime.
        Parameters
        ----------
        lagtime : int
            Lagtime, must be positive. The effective size of the dataset reduces by the selected lagtime.
        data : list of ndarray
            List of trajectories.
        Returns
        -------
        dataset : TrajectoriesDataset
            Concatenation of timeseries datasets.
        Raises
        ------
        AssertionError
            If data is empty, lagtime is not positive,
            the shapes do not match, or lagtime is too long for any of the trajectories.
        """
        assert len(data) > 0 and all(
            data[0].shape[1:] == x.shape[1:] for x in data
        ), "Shape mismatch!"
        return TrajectoriesDataset(
            [
                TrajectoryDataset(lagtime, traj, ids, condition, x_label, xhat_label)
                for traj in data
            ]
        )

    @property
    def lagtime(self):
        r""" The lagtime.
        :type: int
        """
        return self.subsets[0].lagtime

    @property
    def trajectories(self):
        r""" Contained raw trajectories.
        :type: list of ndarray
        """
        return [x.trajectory for x in self.subsets]
