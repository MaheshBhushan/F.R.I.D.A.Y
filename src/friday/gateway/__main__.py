"""`python -m friday.gateway` -- the client/smoke harness.

Exists so the harness can be invoked as the package itself. Running
`python -m friday.gateway.client` also works but makes runpy warn: importing
the package first puts the submodule in sys.modules before runpy executes it.
"""

from friday.gateway.client import main

if __name__ == "__main__":
    raise SystemExit(main())
