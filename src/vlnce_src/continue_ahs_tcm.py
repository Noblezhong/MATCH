"""AHS + TCM ablation entry; reuse the shared MATCH evaluator."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.vlnce_src.match_eval import main

if __name__ == "__main__":
    main()
