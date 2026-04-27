# Bootstrap Script Usage Guide

## Overview

`src/bootstrap.py` is a script runner that allows you to run any test script with zero cognitive overhead, automatically handling all environment setup.

## Basic Usage

```bash
uv run python src/bootstrap.py [script_path] [script_arguments...]
```

The script path can be:
- Relative to the project root: `unit_test/memory_manager_single_test.py`
- Relative to the current directory: `../tests/test_convert_rest.py`
- Absolute path: `/path/to/your/script.py`

## Usage Examples

### 1. Run Unit Tests

```bash
# Run memory manager test
uv run python src/bootstrap.py unit_test/memory_manager_single_test.py

# Run test with relative path
uv run python src/bootstrap.py ../tests/test_convert_rest.py

# Run test with arguments
uv run python src/bootstrap.py unit_test/memory_manager_single_test.py --verbose
```

### 2. Run Evaluation Scripts

```bash
# Run dynamic memory evaluation
uv run python src/bootstrap.py evaluation/dynamic_memory_evaluation/locomo_eval.py

# Run evaluation with dataset argument
uv run python src/bootstrap.py evaluation/dynamic_memory_evaluation/locomo_eval.py --dataset small
```

### 3. Run Algorithm Debugging Scripts

```bash
# Run algorithm debugging script
uv run python src/bootstrap.py tests/algorithms/debug_my_model.py

# Run script with config file
uv run python src/bootstrap.py tests/algorithms/debug_my_model.py --config config.json
```

### 4. Run Test Template

```bash
# Run Bootstrap test template to learn how to use DI and MongoDB
uv run python src/bootstrap.py tests/bootstrap_test_template.py
```

## Command Line Options

### `--env-file`
Specify the environment variable file to load (default: `.env`)

```bash
uv run python src/bootstrap.py your_script.py --env-file .env.test
```

### `--mock`
Enable Mock mode (for testing and development)

```bash
uv run python src/bootstrap.py your_script.py --mock
```

## Environment Variables

### `MOCK_MODE`
Set to `true` to enable Mock mode

```bash
MOCK_MODE=true python src/bootstrap.py your_script.py
```

## Test Template

The project provides a complete test template `tests/bootstrap_test_template.py` that demonstrates how to:

- **Use dependency injection**: Get singleton objects via `get_bean_by_type()` and `get_bean()`
- **Work with MongoDB**: Use the repository pattern for database queries and operations
- **Integration testing**: Combine multiple components for comprehensive testing

The template includes examples like:
```python
# Get MongoDB repository
from core.di.utils import get_bean_by_type
repo = get_bean_by_type(MemCellRawRepository)

# Query data
memcells = await repo.find_all(limit=5)
total_count = await repo.count_all()
```

## Best Practices

### 1. Daily Usage
Run your scripts directly:
```bash
uv run python src/bootstrap.py your_script.py
```

### 2. Learning Development
Start by running the test template to understand the project structure:
```bash
uv run python src/bootstrap.py tests/bootstrap_test_template.py
```

### 3. Development and Testing
Use Mock mode during development:
```bash
uv run python src/bootstrap.py your_script.py --mock
```

### 4. CI/CD Integration
Specify different environment files in continuous integration:
```bash
uv run python src/bootstrap.py test_script.py --env-file .env.ci
```

## Troubleshooting

### 1. Relative Import Errors
If you encounter `ImportError: attempted relative import with no known parent package` error:
- Bootstrap will automatically detect and switch to module mode
- No manual intervention needed, the script will automatically retry

### 2. Import Errors
If you encounter other module import errors, check:
- Whether you're executing from the project root directory
- Whether the `.env` file exists and is configured correctly

### 3. Script Execution Errors
If the target script fails to execute:
- Check if the script path is correct
- Confirm the script itself has no syntax errors

