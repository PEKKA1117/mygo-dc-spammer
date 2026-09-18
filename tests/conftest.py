import sys
from pathlib import Path

# Let `import bot` work without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
