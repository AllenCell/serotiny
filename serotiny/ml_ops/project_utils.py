import os
import sys
from pathlib import Path

def get_serotiny_project():
    try:
        if (Path(os.getcwd()) / ".serotiny").exists():
            with open(".serotiny", "r") as f:
                project_name = f.read().strip()
            return project_name
    except:
        pass

    return "serotiny"
