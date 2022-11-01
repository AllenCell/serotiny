from serotiny.io.dataframe.loaders import LoadRandomTensor


def test_random_tensor():
    loader = LoadRandomTensor(column="none", dims=(4, 3, 2, 1))

    tensor = loader(None)

    assert len(tensor.shape) == 4
    assert tuple(tensor.shape) == (4, 3, 2, 1)
