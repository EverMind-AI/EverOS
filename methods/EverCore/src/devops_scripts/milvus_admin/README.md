# Milvus Admin Tools

## browse_collections.py

Browse Milvus collections and interactively delete them.

### Usage

```bash
# Run via bootstrap (from project root)
uv run python src/bootstrap.py src/devops_scripts/milvus_admin/browse_collections.py [OPTIONS]
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--filter TEXT` | `-f` | Filter collections by name substring (case-insensitive) |
| `--prefix TEXT` | `-p` | Match collections by name prefix (exact, case-sensitive) |
| `--delete` | `-d` | Enter interactive delete mode after listing |
| `--drop` | | Delete all matched collections (requires `--prefix` or `--filter`) |
| `--db TEXT` | | Specify Milvus database name (default: from env) |

### Examples

```bash
# List all collections
uv run python src/bootstrap.py src/devops_scripts/milvus_admin/browse_collections.py

# Filter by keyword
uv run python src/bootstrap.py src/devops_scripts/milvus_admin/browse_collections.py -f episodic

# List by prefix
uv run python src/bootstrap.py src/devops_scripts/milvus_admin/browse_collections.py -p v1_episodic

# Interactive delete (select by number after listing)
uv run python src/bootstrap.py src/devops_scripts/milvus_admin/browse_collections.py --delete

# Batch delete by prefix (requires 'yes' confirmation)
uv run python src/bootstrap.py src/devops_scripts/milvus_admin/browse_collections.py --prefix v1_episodic --drop

# Batch delete by filter
uv run python src/bootstrap.py src/devops_scripts/milvus_admin/browse_collections.py --filter episodic --drop

# Use a specific database
uv run python src/bootstrap.py src/devops_scripts/milvus_admin/browse_collections.py --db my_database
```

### Output

```
================================================================================
  Found 3 collection(s)
================================================================================

#    Collection Name                                          Rows Aliases
------------------------------------------------------------------------
1    v1_episodic_memory_tenant_a-20260301120000000000          1.2K v1_episodic_memory_tenant_a
2    v1_episodic_memory_tenant_b-20260315080000000000          3.5K v1_episodic_memory_tenant_b
3    v1_memcell_tenant_a-20260310150000000000                    89 v1_memcell_tenant_a
```

### Interactive Delete Mode

When using `--delete`, you can select collections by:

- Single numbers: `1,3,5`
- Ranges: `2-6`
- All: `all`
- Quit: `q`

All delete operations require typing `yes` to confirm.
