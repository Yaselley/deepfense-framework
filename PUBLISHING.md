# Publishing DeepFense Library

This guide explains how to publish DeepFense as a Python package to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account at [pypi.org](https://pypi.org) and [test.pypi.org](https://test.pypi.org) for testing
2. **API Tokens**: Generate API tokens from your PyPI account settings
3. **Build Tools**: Install build tools
   ```bash
   pip install build twine
   ```

## Step 1: Prepare Your Package

### 1.1 Update Version Number

Update the version in:
- `pyproject.toml`: `version = "0.1.1"`
- `deepfense/cli/main.py`: `@click.version_option(version="0.1.1")`
- Consider using semantic versioning: MAJOR.MINOR.PATCH

### 1.2 Update Documentation

Ensure:
- README.md is up to date
- All docstrings are clear
- LICENSE file exists (Apache 2.0)
- CHANGELOG.md documents changes (optional but recommended)

### 1.3 Verify Setup

Check that all dependencies are correctly listed in `pyproject.toml`:
```bash
python -m pip install -e .  # Install in development mode
python -c "import deepfense; print('OK')"  # Verify imports
deepfense --help  # Verify CLI works
```

## Step 2: Build the Package

### 2.1 Clean Previous Builds

```bash
cd /netscratch/yelkheir/DeepFense/DeepFense
rm -rf build/ dist/ *.egg-info/
```

### 2.2 Build Distribution

```bash
# Build both wheel and source distribution
python -m build

# This creates:
# - dist/deepfense-0.1.0-py3-none-any.whl (wheel)
# - dist/deepfense-0.1.0.tar.gz (source distribution)
```

## Step 3: Test on Test PyPI (Recommended)

### 3.1 Upload to Test PyPI

```bash
python -m twine upload --repository testpypi dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your Test PyPI API token

### 3.2 Test Installation from Test PyPI

```bash
# Create a clean virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ deepfense

# Test it works
deepfense --help
python -c "import deepfense; print('OK')"
```

## Step 4: Publish to PyPI

### 4.1 Upload to Production PyPI

```bash
python -m twine upload dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your PyPI API token

### 4.2 Verify Installation

```bash
# Create a clean virtual environment
python -m venv prod_env
source prod_env/bin/activate

# Install from PyPI
pip install deepfense

# Test it works
deepfense --help
python -c "import deepfense; print('OK')"
```

## Step 5: Tag Release (Optional but Recommended)

After successful publication:

```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

## Updating the Package

For updates:

1. **Update version** in `pyproject.toml` and `deepfense/cli/main.py`
2. **Build again**: `python -m build`
3. **Upload**: `python -m twine upload dist/*`
4. **Tag release**: `git tag -a v0.1.1 -m "Release version 0.1.1"`

## Troubleshooting

### Common Issues

1. **Package already exists**: Increment version number
2. **Missing files**: Check `MANIFEST.in` if needed
3. **Import errors**: Verify all dependencies are in `pyproject.toml`
4. **CLI not found**: Verify entry point in `pyproject.toml`:
   ```toml
   [project.scripts]
   deepfense = "deepfense.cli.main:cli"
   ```

### Using .pypirc (Alternative Authentication)

Create `~/.pypirc`:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-token-here
```

Then upload with:
```bash
python -m twine upload --repository testpypi dist/*  # For test
python -m twine upload dist/*  # For production
```

## GitHub Actions (Automated Publishing)

Consider setting up GitHub Actions for automated publishing on tags:

Create `.github/workflows/publish.yml`:
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
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      - name: Build package
        run: python -m build
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

## Additional Resources

- [PyPI Packaging Guide](https://packaging.python.org/en/latest/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Python Packaging User Guide](https://packaging.python.org/guides/distributing-packages-using-setuptools/)

