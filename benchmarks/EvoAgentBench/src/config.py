"""Configuration and registry.

config.yaml: who (agent), what (domain), where (output).
Domain-specific settings (split, trials, etc) live in each domain's yaml.
"""

import importlib
import os
import yaml
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SRC_DIR.parent


def _resolve_env_vars(obj):
    """Resolve ${ENV_VAR} placeholders in config values."""
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        return os.environ.get(obj[2:-1], obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj
_config = None
_domain_configs = {}


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve(val, base_dir: Path):
    """Resolve relative paths against base_dir. URLs, bare names, and absolute paths are left as-is."""
    if val and isinstance(val, str):
        if '://' in val or ' ' in val:
            return val
        if not val.startswith(('.', '/')):
            return val
        p = Path(val)
        if not p.is_absolute():
            return str((base_dir / p).resolve())
    return val


def _resolve_dict(d: dict, base_dir: Path) -> dict:
    """Resolve all string values in a dict that look like paths."""
    resolved = {}
    for k, v in d.items():
        if isinstance(v, str) and ('/' in v or '.' in v):
            resolved[k] = _resolve(v, base_dir)
        elif isinstance(v, dict):
            resolved[k] = _resolve_dict(v, base_dir)
        else:
            resolved[k] = v
    return resolved


# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

def load_config(config_path: str | None = None, agent_override: str | None = None) -> dict:
    """Load and cache global config from config.yaml."""
    global _config
    if _config is not None and agent_override is None:
        return _config

    path = Path(config_path) if config_path else _PROJECT_DIR / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    base_dir = path.resolve().parent

    # agent: can be a string (name) or dict (legacy: {name, config, ...})
    agent_field = agent_override or raw.get("agent", "openclaw")
    domain_raw = raw.get("domain", {})
    domain_config_path = domain_raw.get("config")

    # Normalize: string -> just a name, dict -> legacy format
    if isinstance(agent_field, str):
        agent_name = agent_field
        agent_inline = {}
    else:
        agent_inline = dict(agent_field)
        agent_name = agent_inline.pop("name", None) or agent_inline.pop("config", "openclaw")
        # Legacy: "config" pointed to yaml path directly
        if "config" in agent_inline:
            agent_inline.pop("config")

    # Load agent yaml via agent_configs map > default convention
    agents_map = raw.get("agent_configs", {})
    if agent_name in agents_map:
        agent_yaml_path = Path(_resolve(agents_map[agent_name], base_dir))
    else:
        agent_yaml_path = _SRC_DIR / "agents" / agent_name / f"{agent_name}.yaml"

    agent_raw = {}
    if agent_yaml_path.exists():
        with open(agent_yaml_path) as af:
            agent_raw = yaml.safe_load(af) or {}
    # Inline overrides from config.yaml take precedence
    agent_raw.update(agent_inline)

    # Resolve ${ENV_VAR} in values

    providers = _resolve_env_vars(agent_raw.get("providers", {}))

    _config = {
        "agent": {
            "name": agent_raw.get("name", "openclaw"),
            "command": agent_raw.get("command", agent_raw.get("name", "openclaw")),
            "model": agent_raw.get("model"),
            "provider": agent_raw.get("provider"),
            "providers": providers,
            "max_concurrent": agent_raw.get("max_concurrent"),
            "retry_timeout": agent_raw.get("retry_timeout"),
            "tools": agent_raw.get("tools"),
            "maxTokens": agent_raw.get("maxTokens"),
            "contextWindowTokens": agent_raw.get("contextWindowTokens"),
            "temperature": agent_raw.get("temperature"),
            "reasoningEffort": agent_raw.get("reasoningEffort"),
        },
        "domain": {
            "name": domain_raw.get("name", "information_retrieval"),
            "config": _resolve(domain_config_path, base_dir) if domain_config_path else None,
        },
        "job_dir": _resolve(raw.get("job_dir", "./jobs"), base_dir),
        "trials": raw.get("trials", 1),
        "parallel": raw.get("parallel", 1),
        "max_retries": raw.get("max_retries", 2),
        "live": raw.get("live", False),
        "disk_budget": raw.get("disk_budget"),
        "docker": _resolve(raw.get("docker", "docker"), base_dir),
        "docker_sock": raw.get("docker_sock", "/var/run/docker.sock"),
        "_base_dir": str(base_dir),
    }
    return _config


def get_config() -> dict:
    """Get cached global config, loading defaults if needed."""
    if _config is None:
        return load_config()
    return _config


# ---------------------------------------------------------------------------
# Per-domain config
# ---------------------------------------------------------------------------

def get_domain_config(domain_name: str) -> dict:
    """Load and cache a domain-specific config file.

    Lookup order:
    1. Path specified in config.yaml under domain.config
    2. Default: src/domains/{name}/{name}.yaml
    """
    if domain_name in _domain_configs:
        return _domain_configs[domain_name]

    cfg = get_config()
    explicit_path = None
    if cfg["domain"]["name"] == domain_name:
        explicit_path = cfg["domain"].get("config")

    if explicit_path:
        domain_path = Path(explicit_path)
    else:
        # Try {name}.yaml first, then scan for any .yaml in the directory
        domain_dir = _SRC_DIR / "domains" / domain_name
        domain_path = domain_dir / f"{domain_name}.yaml"
        if not domain_path.exists() and domain_dir.is_dir():
            yamls = list(domain_dir.glob("*.yaml"))
            if yamls:
                domain_path = yamls[0]

    if not domain_path.exists():
        raise FileNotFoundError(
            f"Domain config not found in: src/domains/{domain_name}/\n"
            f"To add a new domain, create a .yaml config file there."
        )

    with open(domain_path) as f:
        raw = yaml.safe_load(f) or {}

    base_dir = domain_path.resolve().parent
    resolved = _resolve_env_vars(_resolve_dict(raw, base_dir))

    _domain_configs[domain_name] = resolved
    return resolved


# ---------------------------------------------------------------------------
# Registry: agent and domain lookup
# ---------------------------------------------------------------------------

_DOMAINS = {}
_AGENTS = {}


def register_domain(name: str, cls):
    _DOMAINS[name] = cls


def register_agent(name: str, cls):
    _AGENTS[name] = cls


def get_domain(name: str):
    """Get domain adapter by name. Auto-imports from domains/{name}/."""
    if name not in _DOMAINS:
        domain_dir = _SRC_DIR / "domains" / name
        if domain_dir.is_dir():
            for py in sorted(domain_dir.glob("*.py")):
                if py.name.startswith("_"):
                    continue
                try:
                    importlib.import_module(f"domains.{name}.{py.stem}")
                    if name in _DOMAINS:
                        break
                except ImportError:
                    continue
    if name not in _DOMAINS:
        raise ValueError(f"Unknown domain: {name}. Available: {list(_DOMAINS.keys())}")
    return _DOMAINS[name]()


def get_agent(name: str):
    """Get agent adapter by name. Auto-imports agents/{name}/{name}.py or agents/{name}.py."""
    if name not in _AGENTS:
        try:
            importlib.import_module(f"agents.{name}.{name}")
        except ModuleNotFoundError:
            importlib.import_module(f"agents.{name}")
    if name not in _AGENTS:
        raise ValueError(f"Unknown agent: {name}. Available: {list(_AGENTS.keys())}")
    return _AGENTS[name]()
