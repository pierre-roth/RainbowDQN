import sys
from pathlib import Path

# Add project root to sys.path so tests can import src modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
