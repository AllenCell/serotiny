from pathlib import Path
import logging
from makefun import wraps

logger = logging.getLogger(__name__)

def _save_as_joblib(result, output_path):
    # import here to optimize CLIs / Fire
    import joblib
    if output_path.is_dir():
        output_path = output_path / "result.joblib"
    joblib.dump(result, output_path)


class PipelineCLI:
    def __init__(
        self,
        output_path=None,
        transforms=[],
        store_methods={}
    ):
        self._output_path = output_path
        self._result = None

        for transform in transforms:
            self._add_transform(transform)

        self._store_methods = store_methods

    def _add_transform(self, func):
        @wraps(func)
        def wrapper(*iargs, **kwargs):
            args = []
            for arg in iargs:
                if (arg == ...) or (arg == "..."):
                    args.append([self._result])
                elif arg == "*...":
                    args.append(self._result)
                else:
                    args.append([arg])
            args = sum(args, [])

            for key, value in kwargs.items():
                if value in [..., "..."]:
                    kwargs[key] = self._result
            self._result = func(*args, **kwargs)
            return self

        setattr(self, func.__name__, wrapper)

    def __str__(self):
        """
        Hijack __str__ to store the final result of a pipeline,
        stored in self._result
        """

        if self._output_path is None:
            logger.warning("No output path given. Unless one of your steps explicitely "
                           "writes to disk, this command will not write any results.")
            return ""

        if self._result is None:
            raise ValueError("No result stored on pipeline")

        self._output_path = Path(self._output_path)

        if type(self._result) in self._store_methods:
            self._store_methods[type(self._result)](self._result, self._output_path)
        else:
            logger.warning("No custom storing method provided. Storing as .joblib")
            _save_as_joblib(self._result, self._output_path)

        return ""
