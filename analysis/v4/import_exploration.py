"""Import completed exploration without re-running models. AI assistance: OpenAI Codex, 2026-10-26."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[2]
source=ROOT.parent/'analysis/model_optimization_20261026'
dest=ROOT/'analysis/exploration_20261026';dest.mkdir(exist_ok=True)
assert source.is_dir()
for p in source.iterdir():
    if p.is_file() and p.suffix in {'.py','.csv','.json','.npz','.md','.png'}:
        shutil.copy2(p,dest/p.name)
for p in dest.glob('*.py'):
    s=p.read_text(encoding='utf-8')
    s=s.replace("DATA=ROOT/'mathorcup_A'","DATA=ROOT if (ROOT/'1.csv').exists() else ROOT/'mathorcup_A'")
    s=s.replace("data=ROOT/'mathorcup_A'","data=ROOT if (ROOT/'1.csv').exists() else ROOT/'mathorcup_A'")
    s=s.replace("ROOT/'analysis/paper_v3_revision/evidence/src/q3_style_generation/latent_gaussian_ot.py'","ROOT/'src/q3_style_generation/latent_gaussian_ot.py'")
    p.write_text(s,encoding='utf-8')
print('Imported saved exploration and adapted model data/source paths.')
