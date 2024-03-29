import json
import platform
import sys

import safecheck


def get_system_info():
    info = {
        "library": safecheck.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
    }
    return info


if __name__ == "__main__":
    print(json.dumps(get_system_info(), indent=4))
