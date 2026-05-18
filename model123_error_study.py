from __future__ import print_function

import sys

if sys.version_info < (3, 10):
    sys.stderr.write(
        "model123_error_study.py requires Python 3.10+.\n"
        "You are running Python %s.\n"
        "Use `python3 model123_error_study.py ...` instead.\n" % sys.version.split()[0]
    )
    raise SystemExit(1)

from scripts.run_model123_synthetic_study import main


if __name__ == "__main__":
    main()
