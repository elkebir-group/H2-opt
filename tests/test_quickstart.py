import subprocess
import sys
from pathlib import Path

QUICKSTART = Path(__file__).resolve().parent.parent / 'examples' / 'quickstart.py'


def test_quickstart_runs():
    # the README examples, with a few training steps
    result = subprocess.run([sys.executable, str(QUICKSTART), '--steps', '20'],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert 'heritability of wavelengths 1-10: [0.053  0.0322' in result.stdout
