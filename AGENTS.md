## Cursor Cloud specific instructions

### Overview

MicroGPT is a single-file, zero-dependency GPT-2 style transformer implemented in pure Python (`microgpt.py`). It uses only Python stdlib modules (`os`, `math`, `random`, `urllib.request`).

### Running

```bash
python3 microgpt.py
```

- On the first run, the script downloads `names.txt` (~228 KB) from GitHub and saves it as `input.txt`. Subsequent runs reuse the cached file.
- Training runs 1000 steps of a pure-Python autograd loop. Expect ~90 seconds wall time on cloud VMs.
- After training, the script generates 20 hallucinated names via inference.

### Caveats

- There are no external dependencies, no `requirements.txt`, no package manager files, and no tests or linters configured.
- The script uses `random.seed(42)` so output is deterministic (given the same Python version).
- `input.txt` is generated at runtime and should not be committed.
