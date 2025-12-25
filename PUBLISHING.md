# Publishing DeepFense to PyPI

This guide explains how to publish DeepFense to PyPI so users can install it with `pip install deepfense`.

## Prerequisites

1. **PyPI Account**: Create accounts on:
   - Test PyPI: https://test.pypi.org/account/register/
   - PyPI: https://pypi.org/account/register/

2. **Install build tools**:
   ```bash
   pip install build twine
   ```

## Publishing Steps

### 1. Update Version

Update the version in `deepfense/__init__.py`:
```python
__version__ = "0.1.0"  # Update to new version
```

### 2. Build Distribution

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build source distribution and wheel
python -m build
```

This creates:
- `dist/deepfense-0.1.0.tar.gz` (source distribution)
- `dist/deepfense-0.1.0-py3-none-any.whl` (wheel)

### 3. Test on Test PyPI (Recommended)

```bash
# Upload to Test PyPI
python -m twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ deepfense
```

### 4. Publish to PyPI

Once tested, publish to the real PyPI:

```bash
python -m twine upload dist/*
```

You'll be prompted for your PyPI credentials.

### 5. Verify Installation

```bash
# Create a fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from PyPI
pip install deepfense

# Verify it works
python -c "import deepfense; print(deepfense.__version__)"
```

## Version Management

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

## Important Notes

1. **Fairseq Dependency**: Fairseq cannot be included in the PyPI package due to its complex build requirements. Users must install it separately if they need fairseq-based frontends.

2. **Package Data**: Configuration files (YAML, parquet) are included via `MANIFEST.in` and `package_data` in `setup.py`.

3. **Large Files**: The package excludes large model files (`.pth`, `.pt`, `.ckpt`) to keep the package size manageable.

## Troubleshooting

### "Package already exists"
- Update the version number
- Ensure you're not re-uploading the same version

### "Invalid distribution"
- Check that `setup.py` and `pyproject.toml` are valid
- Verify all required files are included

### "Missing dependencies"
- Ensure `requirements.txt` is up to date
- Check that all dependencies are available on PyPI

## Automated Publishing (Optional)

You can set up GitHub Actions to automatically publish on tags. Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [created]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install build tools
        run: pip install build twine
      - name: Build package
        run: python -m build
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: python -m twine upload dist/*
```
