# Step-by-Step Guide: Publishing DeepFense to PyPI

This is a quick reference guide for publishing DeepFense to PyPI.

## Prerequisites

1. **PyPI Account**: 
   - Create account at [pypi.org](https://pypi.org/account/register/)
   - Create account at [test.pypi.org](https://test.pypi.org/account/register/) (for testing)

2. **API Tokens**:
   - Log into PyPI → Account Settings → API tokens → Add API token
   - Name it (e.g., "deepfense-pypi") and set scope to "Entire account"
   - Copy the token (starts with `pypi-...`)
   - Do the same for Test PyPI

3. **Install Build Tools**:
   ```bash
   pip install build twine
   ```

4. **pip Version Compatibility**:
   - DeepFense supports **pip 24.1+** (latest versions recommended)
   - If installing locally with pip 24.1+, you may encounter an omegaconf 2.0.6 compatibility issue
   - See section 1.3 for handling this during local installation
   - Users installing from PyPI won't encounter this issue

## Step 1: Prepare the Package

### 1.1 Update Version (if needed)

Update version in `pyproject.toml`:
```toml
version = "0.1.0"  # Use semantic versioning: MAJOR.MINOR.PATCH
```

### 1.2 Verify Package Files

Ensure these files exist and are correct:
- ✅ `README.md` - Main readme
- ✅ `LICENSE` - License file (Apache 2.0)
- ✅ `pyproject.toml` - Package configuration
- ✅ `MANIFEST.in` - Extra files to include (if needed)

### 1.3 Test Local Installation

**Important**: If you're using pip 24.1 or higher, you may encounter an error with omegaconf 2.0.6. Follow these steps:

```bash
cd /netscratch/yelkheir/DeepFense/DeepFense

# Check pip version
pip --version

# Handle omegaconf 2.0.6 compatibility issue with pip 24.1+
# If you get an error about omegaconf having invalid requirements, uninstall it first:
pip uninstall omegaconf -y 2>/dev/null || true

# Install in development mode
pip install -e .

# Verify imports
python -c "import deepfense; print('Import OK')"

# Verify CLI
deepfense --help
deepfense list
```

**Note**: The omegaconf 2.0.6 uninstall step is only needed if you have an existing broken installation. Fresh installs from PyPI won't have this issue.

## Step 2: Clean and Build

### 2.1 Clean Previous Builds

```bash
cd /netscratch/yelkheir/DeepFense/DeepFense
rm -rf build/ dist/ *.egg-info/
```

### 2.2 Build Distribution Packages

```bash
# First, ensure build tools are installed (if not already)
pip install build twine

# Build wheel and source distribution
python -m build

# This creates:
# - dist/deepfense-0.1.0-py3-none-any.whl
# - dist/deepfense-0.1.0.tar.gz
```

### 2.3 Verify Build

```bash
# List what will be included
python -m twine check dist/*

# Should output: "Checking dist/deepfense-0.1.0-py3-none-any.whl: PASSED"
```

## Step 3: Test on Test PyPI (Highly Recommended)

### 3.1 Upload to Test PyPI

```bash
python -m twine upload --repository testpypi dist/*
```

When prompted:
- **Username**: `__token__`
- **Password**: Your Test PyPI API token (starts with `pypi-...`)

### 3.2 Test Installation from Test PyPI

```bash
# Create clean test environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ deepfense

# Test it works
deepfense --help
deepfense list
python -c "import deepfense; print('Success!')"
```

**Note**: You may need `--extra-index-url` because Test PyPI doesn't have all dependencies.

## Step 4: Publish to Production PyPI

### 4.1 Upload to PyPI

```bash
python -m twine upload dist/*
```

When prompted:
- **Username**: `__token__`
- **Password**: Your PyPI API token (starts with `pypi-...`)

### 4.2 Verify on PyPI

Visit: https://pypi.org/project/deepfense/

You should see your package listed!

### 4.3 Test Installation from PyPI

```bash
# Create clean environment
python -m venv prod_env
source prod_env/bin/activate  # On Windows: prod_env\Scripts\activate

# Install from PyPI
pip install deepfense

# Test it works
deepfense --help
deepfense list
python -c "import deepfense; print('Success!')"
```

## Step 5: Tag Release (Recommended)

After successful publication, create a Git tag:

```bash
cd /netscratch/yelkheir/DeepFense/DeepFense

# Tag the release
git tag -a v0.1.0 -m "Release version 0.1.0"

# Push tag to GitHub
git push origin v0.1.0
```

## Updating the Package

For future updates:

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.1.1"  # Increment version
   ```

2. **Clean, build, and upload**:
   ```bash
   rm -rf build/ dist/ *.egg-info/
   python -m build
   python -m twine upload dist/*
   ```

3. **Tag the new release**:
   ```bash
   git tag -a v0.1.1 -m "Release version 0.1.1"
   git push origin v0.1.1
   ```

## Quick Command Reference

```bash
# Full workflow
cd /netscratch/yelkheir/DeepFense/DeepFense
rm -rf build/ dist/ *.egg-info/
python -m build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*  # Test first
python -m twine upload dist/*  # Production
git tag -a v0.1.0 -m "Release v0.1.0" && git push origin v0.1.0
```

## Troubleshooting

### OmegaConf 2.0.6 Installation Error (pip 24.1+)

**Error Message**:
```
error: invalid-installed-package
× Cannot process installed package omegaconf 2.0.6 ... because it has an invalid requirement
│     PyYAML (>=5.1.*)
```

**Cause**: pip 24.1+ validates installed package metadata more strictly, and omegaconf 2.0.6 has invalid dependency specifications in its metadata.

**Solution**:
```bash
# Uninstall the broken omegaconf package
pip uninstall omegaconf -y

# Then install DeepFense (it will install omegaconf 2.0.6 correctly)
pip install -e .
```

**Important Notes**:
- This only affects **local installations** where omegaconf 2.0.6 is already installed with broken metadata
- The package configuration is correct (`omegaconf==2.0.6` in both `pyproject.toml` and `setup.py`)
- Users installing from PyPI won't encounter this issue
- The build process (`python -m build`) is not affected by this issue

### Build Module Not Found
If you get `No module named build`:
```bash
pip install build twine
```

### Package Already Exists
- Solution: Increment version number in `pyproject.toml`

### Missing Dependencies
- Solution: Ensure all dependencies are listed in `pyproject.toml` under `dependencies`

### Import Errors After Installation
- Solution: Verify `deepfense/__init__.py` exists and all imports are correct

### CLI Not Found
- Solution: Verify entry point in `pyproject.toml`:
  ```toml
  [project.scripts]
  deepfense = "deepfense.cli.main:cli"
  ```

### Authentication Issues
- Use `__token__` as username (with two underscores)
- Use your API token as password (starts with `pypi-...`)
- Make sure token has correct scope/permissions

## Security Best Practices

1. **Never commit API tokens** to git
2. **Use API tokens** instead of passwords
3. **Test on Test PyPI first** before production
4. **Use `.pypirc` file** for easier authentication (optional):

   Create `~/.pypirc`:
   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = pypi-your-production-token-here

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = pypi-your-test-token-here
   ```

   Then upload with:
   ```bash
   python -m twine upload --repository testpypi dist/*  # Uses .pypirc
   ```

## Next Steps After Publishing

1. ✅ Update website with PyPI link
2. ✅ Announce release on GitHub
3. ✅ Update documentation with installation instructions
4. ✅ Monitor PyPI project page for issues

## Additional Resources

- Full guide: [PUBLISHING.md](PUBLISHING.md)
- [PyPI Packaging Guide](https://packaging.python.org/en/latest/)
- [Twine Documentation](https://twine.readthedocs.io/)

