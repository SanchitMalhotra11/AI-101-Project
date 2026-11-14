import shutil
from pathlib import Path

import kagglehub

target = Path(__file__).parent
path = kagglehub.dataset_download("alexandrepetit881234/fake-bills")

for f in Path(path).glob("*.csv"):
    shutil.copy(f, target / f.name)
    print("Copied", f.name, "to", target)
