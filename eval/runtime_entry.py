#!/usr/bin/env python3
"""Launch the old HERO CLI with evaluation-only fixes installed first."""
from eval_runtime import install
install()
from hero.cli import main
if __name__ == '__main__':
    main()
