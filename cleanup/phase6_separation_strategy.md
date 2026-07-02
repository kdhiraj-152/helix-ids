# Phase 6: Long-Term Artifact Separation Strategy
> Date: 2026-07-02

## Motivation

The repository currently mixes production code with research artifacts, making it large (~5.8GB after optimization). For long-term maintainability, standard practice is to separate concerns into discrete repositories.

## Recommended Separation

### 1. Production Repository (HELIX-IDS)
**Contents:** Everything needed to train, evaluate, and deploy the model.

```
HELIX-IDS/
├── src/helix_ids/          # Core package
├── scripts/                # Training, serving, deployment
├── tests/                  # Test suite
├── models/                 # Production checkpoints
├── config/                 # YAML configuration
├── docs/                   # Documentation + manuscript
├── data/raw/               # Canonical raw datasets (or submodule)
├── cleanup/archives/       # Compressed research archives (optional)
├── requirements-lock.txt   # Pinned dependencies
├── pyproject.toml          # Project metadata
└── AGENTS.md               # Agent context
```

**GitHub:** `github.com/user/HELIX-IDS`
**Size target:** <50MB (code, docs, config, tests only)
**Includes:** CI/CD, governance, monitoring, deployment pipelines

### 2. Research Artifact Repository (HELIX-IDS-research)
**Contents:** All intermediate and final research outputs.

```
HELIX-IDS-research/
├── results/                # All phase results (phase47–phase64+, CSVs, reports)
├── artifacts/              # Compressed archives of models, embeddings, latents
├── models/archive/         # Archived intermediate checkpoints
├── plots/                  # Generated figures (if any remain)
├── notebooks/              # Jupyter/Colab notebooks (if any)
├── reproducibility/        # Full reproducibility guide with artifact hashes
└── MANIFEST.json           # SHA256 manifest of all archived artifacts
```

**GitHub (LFS):** `github.com/user/HELIX-IDS-research`
**Size target:** ~1-2GB (LFS)
**Access model:** Lazy clone or LFS partial clone — only fetch artifacts needed

### 3. Dataset Repository (HELIX-IDS-datasets)
**Contents:** Raw canonical datasets and preprocessing scripts.

```
HELIX-IDS-datasets/
├── raw/                    # Canonical raw datasets (NSL-KDD, UNSW-NB15, CIC-IDS, etc.)
├── preprocessing/          # Feature harmonization scripts
├── checksums/              # Dataset integrity manifests
└── README.md               # Dataset provenance and citation
```

**GitHub (LFS):** `github.com/user/HELIX-IDS-datasets`
**Size target:** ~2.2GB (LFS)
**Access model:** Git LFS with sparse checkout for individual datasets

### 4. Release Archives
**Contents:** Versioned publication snapshots.

```
HELIX-IDS-releases/
├── v1.0/                   # Publication-quality code snapshot
│   ├── HELIX-IDS-v1.0.tar.xz
│   ├── HELIX-IDS-v1.0.pdf
│   └── checksums.txt
└── latest/                 # Symlink to latest release
```

## Migration Path

### Step 1: Create Research Repository (today)
```bash
# Copy results/ and archive manifests
git init HELIX-IDS-research
cp -r results/ HELIX-IDS-research/
cp -r cleanup/archives/ HELIX-IDS-research/artifacts/
cd HELIX-IDS-research
git lfs track "*.tar.xz"
git lfs track "*.npy"
git add -A
git commit -m "research artifacts"
```

### Step 2: Strip Research from Production (1-2 hours)
```bash
cd HELIX-IDS
git filter-repo --path results/ --invert-paths
git filter-repo --path artifacts/ --invert-paths
git filter-repo --path archive/ --invert-paths
git gc --aggressive --prune=now
```

### Step 3: Git LFS Setup for Datasets (2+ hours)
- Requires evaluating which datasets are canonical vs regenerable
- Each dataset's raw CSVs → LFS
- Preprocessing cache → regenerable (not versioned)

### Step 4: Configure Submodules (optional)
```
HELIX-IDS/
├── .gitmodules
├── src/              # core code
└── data/             # submodule → HELIX-IDS-datasets
```

## Verification Criteria

After separation, verify:
1. `PYTHONPATH=src pytest -q` — 76/76 tests pass
2. `ruff check src/` — lint passes
3. Training pipeline runs: `PYTHONPATH=src python scripts/training/train_helix_ids_full.py --config config/helix_config.yaml`
4. SHA256 verification of all archived artifacts
5. No broken imports between repositories

## Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| **Single monorepo** (current) | Simple, all-in-one | 5.8GB, clone takes minutes |
| **Multi-repo (recommended)** | Fast clones, clean boundaries | Submodule management, cross-repo refs |
| **Git LFS inline** | Single repo with large file pointers | LFS quota, hosting complexity |
| **External object storage** | Minimal repo size | Separate access control, S3 costs |

**Recommendation:** Multi-repo with Git LFS for datasets. Start with research separation (Step 1-2), defer dataset migration.
