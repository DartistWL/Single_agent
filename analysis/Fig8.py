"""Fig8."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    folder = Path(__file__).resolve().parent
    sys.path[:] = [str(folder.parent)] + [p for p in sys.path if Path(p or '.').resolve() != folder]
from analysis.plot_figures import main
if __name__ == '__main__':
    main(['--figures', '8', *sys.argv[1:]])
