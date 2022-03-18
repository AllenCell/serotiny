import os
import sys

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

from serotiny.ml_ops.utils import get_serotiny_project


class SerotinySearchPath(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        if "serotiny" in sys.argv[0]:
            if "-sc" in sys.argv and any(["install" in _arg for _arg in sys.argv]):
                pass
            elif not any(
                [("config-dir" in _arg or "-cd" in _arg) for _arg in sys.argv]
            ):
                # look for a .serotiny file in the current directory. if it exists,
                # it means we're in a serotiny project. the .serotiny file will contain
                # the corresponding python package name, such that we can retrieve
                # the config

                project_name = get_serotiny_project()
                if project_name != "serotiny":
                    search_path.append(
                        provider="serotiny-project",
                        path=f"{os.getcwd()}/{project_name}/config",
                    )
                else:
                    print(
                        "Please specify --config-dir in the command line, "
                        "or run serotiny from within a serotiny project."
                    )
                    sys.exit(-1)

        search_path.append(
            provider="serotiny-searchpath-plugin", path="pkg://serotiny.config"
        )
