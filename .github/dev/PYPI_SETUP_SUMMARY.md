# PyPI Setup Summary

DeepFense has been configured for PyPI distribution. Here's what was set up:

## Files Created/Modified

### ✅ New Files
1. **`deepfense/__init__.py`** - Package initialization with version info
2. **`MANIFEST.in`** - Specifies which files to include in the distribution
3. **`pyproject.toml`** - Modern Python packaging configuration (PEP 518/621)
4. **`PUBLISHING.md`** - Step-by-step guide for publishing to PyPI
5. **`INSTALLATION.md`** - Installation instructions for users

### ✅ Modified Files
1. **`setup.py`** - Updated to read version from `__init__.py` and include package data
2. **`README.md`** - Added PyPI installation instructions
3. **`requirements.txt`** - Added `omegaconf` dependency

## Package Structure

```
deepfense/
├── __init__.py          # Package version and metadata
├── config/              # Configuration files (included via package_data)
├── models/              # Model implementations
├── data/                # Data handling
├── training/            # Training utilities
└── utils/               # Utility functions
```

## Next Steps to Publish

### 1. Test Local Build
```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# This creates dist/ directory with:
# - deepfense-0.1.0.tar.gz
# - deepfense-0.1.0-py3-none-any.whl
```

### 2. Test Installation Locally
```bash
# Install from local build
pip install dist/deepfense-0.1.0-py3-none-any.whl

# Verify
python -c "import deepfense; print(deepfense.__version__)"
```

### 3. Create PyPI Accounts
- Test PyPI: https://test.pypi.org/account/register/
- PyPI: https://pypi.org/account/register/

### 4. Upload to Test PyPI (Recommended First)
```bash
python -m twine upload --repository testpypi dist/*
```

### 5. Publish to PyPI
```bash
python -m twine upload dist/*
```

## Important Notes

### Fairseq Dependency
- Fairseq cannot be included in the PyPI package due to complex build requirements
- Users must install it separately if they need fairseq-based frontends
- This is documented in the README and INSTALLATION.md

### Package Data
- Configuration files (`.yaml`, `.parquet`) are included via:
  - `MANIFEST.in` for source distributions
  - `package_data` in `setup.py` and `pyproject.toml` for wheels

### Excluded Files
- Large model files (`.pth`, `.pt`, `.ckpt`) are excluded to keep package size manageable
- Fairseq and WavLM modules are excluded (users install separately)

## Version Management

Update version in:
1. `deepfense/__init__.py` - `__version__ = "0.1.0"`
2. `pyproject.toml` - `version = "0.1.0"`
3. `setup.py` - Automatically reads from `__init__.py`

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

## Testing Checklist

Before publishing, verify:
- [ ] Package builds without errors: `python -m build`
- [ ] Can install locally: `pip install dist/deepfense-*.whl`
- [ ] Can import: `import deepfense`
- [ ] Version is correct: `deepfense.__version__`
- [ ] Config files are included: Check `deepfense/config/` after install
- [ ] All dependencies install correctly
- [ ] README renders correctly on PyPI (check Test PyPI first)

## After Publishing

Once published, users can install with:
```bash
pip install deepfense
```

And use it:
```python
import deepfense
from deepfense.models import *
from deepfense.utils.registry import build_detector
```

## Troubleshooting

See `PUBLISHING.md` for detailed troubleshooting guide.
