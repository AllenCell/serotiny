import warnings
import quilt3


def download_quilt_data(
    package: str,
    bucket: str,
    data_save_loc: str,
    ignore_warnings: bool = True,
):
    """
    Download a quilt dataset and supress nfs file attribe warnings

    Parameters
    ----------
    package: str
        Name of the package on s3.
        Example: "aics/hipsc_single_cell_image_dataset"

    bucket: str
        The s3 bucket storing the package
        Example: "s3://allencell"

    data_save_loc: str,
        Path to save data

    ignore_warnings: bool,
        Whether to suppress nfs file attribute warnings or not
    """
    dataset_manifest = quilt3.Package.browse(package, bucket)

    if ignore_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dataset_manifest.fetch(data_save_loc)
    else:
        dataset_manifest.fetch(data_save_loc)
