import sys
import pathlib

# Ensure the repository root is on sys.path so tests can import `src` as a package.
# This is robust across CI and local environments where pytest may not add the
# project root automatically.
ROOT = pathlib.Path(__file__).resolve().parent.parent
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
	sys.path.insert(0, ROOT_STR)

