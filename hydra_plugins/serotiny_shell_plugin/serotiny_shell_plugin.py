import os
import re
import sys
from typing import Optional

from hydra.plugins.completion_plugin import CompletionPlugin


def strip_serotiny_call(line: str):
    regex = r"^serotiny(?:\s*|\.)(?:train|test|predict)\s*(.*)"
    match = re.match(regex, line)
    if match:
        return match.group(1)
    else:
        raise RuntimeError(f"Error parsing line '{line}'")


class SerotinyCompletion(CompletionPlugin):
    def install(self) -> None:
        script = """hydra_serotiny_bash_completion()
{
    words=($COMP_LINE)
    helper="serotiny.train"
    EXECUTABLE=($(command -v $helper))
    if [ "$HYDRA_COMP_DEBUG" == "1" ]; then
        printf "EXECUTABLE_FIRST='${EXECUTABLE[0]}'\\n"
    fi
    if ! [ -x "${EXECUTABLE[0]}" ]; then
        false
    fi
    if [ $? == 0 ]; then
        choices=$( COMP_POINT=$COMP_POINT COMP_LINE=$COMP_LINE $helper -sc query=serotiny_bash)
        word=${words[$COMP_CWORD]}
        if [ "$HYDRA_COMP_DEBUG" == "1" ]; then
            printf "\\n"
            printf "COMP_LINE='$COMP_LINE'\\n"
            printf "COMP_POINT='$COMP_POINT'\\n"
            printf "Word='$word'\\n"
            printf "Output suggestions:\\n"
            printf "\\t%s\\n" ${choices[@]}
        fi
        COMPREPLY=($( compgen -o nospace -o default -W "$choices" -- "$word" ));
    fi
}
COMP_WORDBREAKS=${COMP_WORDBREAKS//=}
COMP_WORDBREAKS=$COMP_WORDBREAKS complete -o nospace -o default -F hydra_serotiny_bash_completion serotiny train
COMP_WORDBREAKS=$COMP_WORDBREAKS complete -o nospace -o default -F hydra_serotiny_bash_completion serotiny.train
COMP_WORDBREAKS=$COMP_WORDBREAKS complete -o nospace -o default -F hydra_serotiny_bash_completion serotiny test
COMP_WORDBREAKS=$COMP_WORDBREAKS complete -o nospace -o default -F hydra_serotiny_bash_completion serotiny.test
COMP_WORDBREAKS=$COMP_WORDBREAKS complete -o nospace -o default -F hydra_serotiny_bash_completion serotiny predict
COMP_WORDBREAKS=$COMP_WORDBREAKS complete -o nospace -o default -F hydra_serotiny_bash_completion serotiny.predict
"""  # noqa
        print(script)

    def uninstall(self) -> None:
        print("unset hydra_serotiny_bash_completion")

    @staticmethod
    def provides() -> str:
        return "serotiny_bash"

    @staticmethod
    def help(command: str) -> str:
        assert command in ["install", "uninstall"]
        return f'eval "$({{}} -sc {command}=serotiny_bash)"'

    @staticmethod
    def _get_exec() -> str:
        if "train" in sys.argv[0]:
            return "serotiny.train"
        elif "test" in sys.argv[0]:
            return "serotiny.test"
        elif "predict" in sys.argv[0]:
            return "serotiny.test"
        else:
            return None

    def query(self, config_name: Optional[str]) -> None:
        line = os.environ["COMP_LINE"]
        line = strip_serotiny_call(line)

        if "train" in sys.argv[0]:
            config_name = "train"
        elif "test" in sys.argv[0]:
            config_name = "test"
        elif "predict" in sys.argv[0]:
            config_name = "predict"
        else:
            raise RuntimeError(
                "Didn't get a valid config_name. "
                "(Should be 'train', 'test' or 'predict')"
            )

        print(" ".join(self._query(config_name=config_name, line=line)))
