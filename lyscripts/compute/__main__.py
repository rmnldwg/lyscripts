"""Run the compute module as a script."""

from lyscripts.cli import _assemble_main
from lyscripts.compute import ComputeCLI

if __name__ == "__main__":
    main = _assemble_main(settings_cls=ComputeCLI, prog_name="compute")
    main()
