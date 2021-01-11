import logging
from datetime import datetime
from pathlib import Path
from dask_jobqueue import SLURMCluster
from distributed import LocalCluster
from prefect.engine.executors import DaskExecutor, LocalExecutor

log = logging.getLogger(__name__)


def choose_executor(
    distributed=False,
    debug=False,
    num_workers=10,
    worker_cpu=8,
    worker_mem="120GB",
    batch_size=None,
):

    if debug:
        exe = LocalExecutor()
        log.info("Debug flagged. Will use threads instead of Dask.")

    elif distributed:
        # Create or get log dir
        # Do not include ms
        log_dir_name = datetime.now().isoformat().split(".")[0]
        log_dir = Path(f".dask_logs/{log_dir_name}").expanduser()
        # Log dir settings
        log_dir.mkdir(parents=True, exist_ok=True)
        # Create cluster
        log.info("Creating SLURMCluster")
        cluster = SLURMCluster(
            cores=worker_cpu,
            memory=worker_mem,
            queue="aics_cpu_general",
            walltime="9-23:00:00",
            local_directory=str(log_dir),
            log_directory=str(log_dir),
        )

        # Spawn workers
        cluster.scale(jobs=num_workers)
        log.info("Created SLURMCluster")

        # Use the port from the created connector to set executor address
        distributed_executor_address = cluster.scheduler_address

        # Only auto batch size if it is not None
        if batch_size is None:
            # Batch size is num_workers * worker_cpu * 0.75
            # We could just do num_workers * worker_cpu but 3/4 of that is safer
            batch_size = int(num_workers * worker_cpu * 0.75)

        # Log dashboard URI
        log.info(f"Dask dashboard available at: {cluster.dashboard_link}")

    else:
        # Create local cluster
        log.info("Creating LocalCluster")
        cluster = LocalCluster()
        log.info("Created LocalCluster")

        # Set distributed_executor_address
        distributed_executor_address = cluster.scheduler_address

        # Log dashboard URI
        log.info(f"Dask dashboard available at: {cluster.dashboard_link}")

    # Use dask cluster
    exe = DaskExecutor(distributed_executor_address)

    return exe, distributed_executor_address
