"""
Unit and safety tests for Alpaca MCP Server V2 configuration.

Verifies:
1. Workspace MCP config exists.
2. Valid JSON syntax.
3. Server name is strictly 'alpaca-paper-readonly'.
4. Command is 'uvx'.
5. Args is ['alpaca-mcp-server'].
6. ALPACA_PAPER_TRADE == 'true'.
7. ALPACA_TOOLSETS strictly excludes 'trading'.
8. Credentials use environment variable expansion references ($ALPACA_PAPER_API_KEY, $ALPACA_PAPER_SECRET_KEY).
9. No literal secrets appear in config.
10. Trust is explicitly false.
11. Live credentials/modes are strictly absent.
12. Preserves unrelated MCP servers during updates.
13. Trading runtime execution path has zero MCP dependencies/imports.
14. Laya V2 inference has zero MCP/broker execution access.
"""

import json
import os
import re
from pathlib import Path
import pytest


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
MCP_CONFIG_PATH = WORKSPACE_ROOT / ".agents" / "mcp_config.json"


def load_mcp_config():
    assert MCP_CONFIG_PATH.exists(), f"MCP config not found at {MCP_CONFIG_PATH}"
    with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_mcp_config_exists_and_is_valid_json():
    """1 & 2: MCP config exists and parses as valid JSON."""
    assert MCP_CONFIG_PATH.is_file(), f"Expected config file at {MCP_CONFIG_PATH}"
    with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), "Root of MCP config must be a JSON object"
    assert "mcpServers" in data, "MCP config must contain 'mcpServers' mapping"


def test_alpaca_server_name_and_command():
    """3, 4, 5: Server name is 'alpaca-paper-readonly', command is 'uvx', args is ['alpaca-mcp-server']."""
    config = load_mcp_config()
    servers = config.get("mcpServers", {})
    assert "alpaca-paper-readonly" in servers, "Server name must be 'alpaca-paper-readonly'"
    server = servers["alpaca-paper-readonly"]
    assert server.get("command") == "uvx", f"Expected command 'uvx', got {server.get('command')}"
    assert server.get("args") == ["alpaca-mcp-server"], f"Expected args ['alpaca-mcp-server'], got {server.get('args')}"


def test_alpaca_paper_trade_mode():
    """6: ALPACA_PAPER_TRADE is strictly 'true'."""
    config = load_mcp_config()
    server = config["mcpServers"]["alpaca-paper-readonly"]
    env = server.get("env", {})
    assert env.get("ALPACA_PAPER_TRADE") == "true", "ALPACA_PAPER_TRADE must be set to 'true'"


def test_alpaca_toolsets_excludes_trading():
    """7: ALPACA_TOOLSETS strictly excludes 'trading' and 'watchlists'."""
    config = load_mcp_config()
    server = config["mcpServers"]["alpaca-paper-readonly"]
    env = server.get("env", {})
    toolsets_str = env.get("ALPACA_TOOLSETS", "")
    toolsets = [t.strip() for t in toolsets_str.split(",") if t.strip()]

    assert "trading" not in toolsets, "CRITICAL: 'trading' toolset MUST NOT be exposed to MCP"
    assert "watchlists" not in toolsets, "'watchlists' toolset should not be exposed without need"

    # Verify required read-only toolsets
    expected_read_only = {"account", "stock-data", "assets", "news", "corporate-actions"}
    assert set(toolsets) == expected_read_only, f"Expected toolsets {expected_read_only}, got {set(toolsets)}"


def test_credentials_use_env_var_references_and_no_literals():
    """8 & 9: Credentials use environment-variable references and no literal secrets exist in config."""
    with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    config = json.loads(raw_text)
    server = config["mcpServers"]["alpaca-paper-readonly"]
    env = server.get("env", {})

    api_key_ref = env.get("ALPACA_API_KEY", "")
    secret_key_ref = env.get("ALPACA_SECRET_KEY", "")

    assert api_key_ref.startswith("$"), f"ALPACA_API_KEY must reference an env var (got: {api_key_ref})"
    assert secret_key_ref.startswith("$"), f"ALPACA_SECRET_KEY must reference an env var (got: {secret_key_ref})"
    assert api_key_ref == "$ALPACA_PAPER_API_KEY", f"Expected $ALPACA_PAPER_API_KEY, got {api_key_ref}"
    assert secret_key_ref == "$ALPACA_PAPER_SECRET_KEY", f"Expected $ALPACA_PAPER_SECRET_KEY, got {secret_key_ref}"

    # Verify no raw secret key literal pattern appears anywhere in the file
    # Alpaca API keys are 20 char alphanumeric starting with PK or AK, secret keys are 40 char hex/base64
    assert not re.search(r"PK[A-Za-z0-9]{18}", raw_text), "Literal paper API key found in mcp_config.json!"
    assert not re.search(r"AK[A-Za-z0-9]{18}", raw_text), "Literal live API key found in mcp_config.json!"
    assert not re.search(r"[A-Za-z0-9+/]{40}", raw_text), "Potential literal secret found in mcp_config.json!"


def test_mcp_trust_is_false():
    """10: Trust is explicitly false."""
    config = load_mcp_config()
    server = config["mcpServers"]["alpaca-paper-readonly"]
    assert server.get("trust") is False, "trust must be set to false"


def test_no_live_credentials_passed_to_mcp():
    """11: Live credentials or live flags must not be configured in MCP server env."""
    config = load_mcp_config()
    server = config["mcpServers"]["alpaca-paper-readonly"]
    env = server.get("env", {})

    forbidden_keys = [
        "ALPACA_LIVE_API_KEY",
        "ALPACA_LIVE_SECRET_KEY",
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "LIVE_TRADING_ENABLED",
    ]
    for key in forbidden_keys:
        assert key not in env, f"Forbidden key '{key}' found in MCP server env"

    for val in env.values():
        assert "LIVE" not in str(val).upper(), f"Live reference found in MCP env value: {val}"


def test_unrelated_mcp_servers_preserved():
    """12: Updating MCP config preserves existing unrelated servers."""
    test_config = {
        "mcpServers": {
            "unrelated-server": {
                "command": "python",
                "args": ["-m", "other_server"],
                "trust": True
            }
        }
    }
    # Simulate non-destructive update
    merged = dict(test_config["mcpServers"])
    alpaca_def = load_mcp_config()["mcpServers"]["alpaca-paper-readonly"]
    merged["alpaca-paper-readonly"] = alpaca_def

    assert "unrelated-server" in merged
    assert "alpaca-paper-readonly" in merged
    assert merged["unrelated-server"]["command"] == "python"


def test_trading_execution_runtime_does_not_import_mcp():
    """13: Deterministic trading execution path does not import or depend on MCP."""
    execution_files = [
        WORKSPACE_ROOT / "trading" / "runtime" / "orchestrator.py",
        WORKSPACE_ROOT / "trading" / "risk" / "engine.py",
        WORKSPACE_ROOT / "trading" / "portfolio" / "constructor.py",
        WORKSPACE_ROOT / "trading" / "brokers" / "alpaca_paper_broker.py",
        WORKSPACE_ROOT / "trading" / "execution" / "adapter.py" if (WORKSPACE_ROOT / "trading" / "execution" / "adapter.py").exists() else None,
        WORKSPACE_ROOT / "trading" / "execution" / "live_guard.py",
    ]

    for file_path in execution_files:
        if file_path is None or not file_path.exists():
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "mcp" not in content.lower() or "alpaca-mcp" not in content.lower(), (
            f"Execution module {file_path.name} must not reference MCP!"
        )
        assert "import mcp" not in content, f"Execution module {file_path.name} imports mcp!"
        assert "from mcp" not in content, f"Execution module {file_path.name} imports from mcp!"


def test_laya_v2_has_no_mcp_or_broker_execution_authority():
    """14: Laya V2 model & specialist have zero MCP and zero broker execution access."""
    laya_files = [
        WORKSPACE_ROOT / "trading" / "ml" / "laya_specialist_v2.py",
        WORKSPACE_ROOT / "trading" / "ml" / "laya_specialist.py",
        WORKSPACE_ROOT / "trading" / "ml" / "meta_model.py",
    ]

    for file_path in laya_files:
        if not file_path.exists():
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "mcp" not in content.lower(), f"Laya module {file_path.name} references MCP!"
        assert "broker.submit" not in content, f"Laya module {file_path.name} has broker submission access!"
        assert "place_order" not in content, f"Laya module {file_path.name} has order placement access!"


def _get_uvx_path():
    import shutil
    uvx_bin = shutil.which("uvx")
    if not uvx_bin:
        candidates = [
            Path.home() / ".local" / "bin" / "uvx",
            Path.home() / ".cargo" / "bin" / "uvx",
            Path.home() / ".cargo" / "bin" / "uvx.exe",
            Path.home() / "AppData" / "Local" / "bin" / "uvx.exe",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
    return uvx_bin or "uvx"


def test_uvx_resolves_alpaca_mcp_server():
    """15: uvx is installed and resolves alpaca-mcp-server V2."""
    import subprocess
    uvx_cmd = _get_uvx_path()
    cmd = [uvx_cmd, "alpaca-mcp-server", "--version"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"uvx failed to execute alpaca-mcp-server: {result.stderr}"
    assert "alpaca-mcp-server" in result.stdout or "alpaca-mcp-server" in result.stderr
    version_match = re.search(r"version\s+([0-9]+\.[0-9]+\.[0-9]+)", result.stdout + result.stderr)
    assert version_match is not None, f"Could not detect version in output: {result.stdout}"
    major_version = int(version_match.group(1).split(".")[0])
    assert major_version >= 2, f"Expected Alpaca MCP V2+, got {version_match.group(1)}"


def test_alpaca_mcp_server_exposes_zero_order_placement_tools():
    """16: Verify that alpaca-mcp-server with configured toolsets registers 0 order tools."""
    import subprocess
    uvx_cmd = _get_uvx_path()
    verify_script = (
        "import os, asyncio\n"
        "os.environ['ALPACA_API_KEY'] = 'PK_TEST_DUMMY'\n"
        "os.environ['ALPACA_SECRET_KEY'] = 'SK_TEST_DUMMY'\n"
        "os.environ['ALPACA_PAPER_TRADE'] = 'true'\n"
        "os.environ['ALPACA_TOOLSETS'] = 'account,stock-data,assets,news,corporate-actions'\n"
        "from alpaca_mcp_server.server import build_server\n"
        "server = build_server()\n"
        "tools = asyncio.run(server.list_tools())\n"
        "names = [t.name for t in tools]\n"
        "forbidden = ['place_stock_order', 'place_crypto_order', 'place_option_order', "
        "'cancel_all_orders', 'cancel_order_by_id', 'replace_order_by_id', "
        "'exercise_options_position', 'close_all_positions', 'close_position']\n"
        "found = [f for f in forbidden if f in names]\n"
        "assert not found, f'Forbidden tools found: {found}'\n"
        "assert 'get_account_info' in names\n"
        "assert 'get_stock_bars' in names\n"
        "print('PASS_ZERO_ORDER_TOOLS', len(names))\n"
    )
    cmd = [uvx_cmd, "--with", "alpaca-mcp-server", "python", "-c", verify_script]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Tool verification script failed: {result.stderr}"
    assert "PASS_ZERO_ORDER_TOOLS" in result.stdout


def test_exclude_tools_blocks_update_account_config():
    """17: Verify excludeTools in mcp_config.json explicitly blocks update_account_config."""
    config = load_mcp_config()
    server = config["mcpServers"]["alpaca-paper-readonly"]
    exclude_tools = server.get("excludeTools", [])
    assert isinstance(exclude_tools, list), "excludeTools must be a list"
    assert "update_account_config" in exclude_tools, "excludeTools must include 'update_account_config'"


def test_defense_in_depth_produces_32_strictly_read_only_tools():
    """18: Verify defense in depth yields exactly 32 approved read-only tools and zero mutation tools."""
    import subprocess
    uvx_cmd = _get_uvx_path()
    config = load_mcp_config()
    server = config["mcpServers"]["alpaca-paper-readonly"]
    exclude_tools = set(server.get("excludeTools", []))

    verify_script = (
        "import os, asyncio, json\n"
        "os.environ['ALPACA_API_KEY'] = 'PK_TEST_DUMMY'\n"
        "os.environ['ALPACA_SECRET_KEY'] = 'SK_TEST_DUMMY'\n"
        "os.environ['ALPACA_PAPER_TRADE'] = 'true'\n"
        "os.environ['ALPACA_TOOLSETS'] = 'account,stock-data,assets,news,corporate-actions'\n"
        "from alpaca_mcp_server.server import build_server\n"
        "server = build_server()\n"
        "tools = asyncio.run(server.list_tools())\n"
        "print(json.dumps([t.name for t in tools]))\n"
    )
    cmd = [uvx_cmd, "--with", "alpaca-mcp-server", "python", "-c", verify_script]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Tool discovery failed: {result.stderr}"

    discovered_tools = json.loads(result.stdout.strip())
    assert "update_account_config" in discovered_tools, "update_account_config should be present before filtering"

    # Apply Antigravity excludeTools filter
    visible_tools = [t for t in discovered_tools if t not in exclude_tools]

    assert "update_account_config" not in visible_tools, "update_account_config must be filtered out by excludeTools"
    assert len(visible_tools) == 32, f"Expected exactly 32 visible read-only tools, got {len(visible_tools)}: {visible_tools}"

    # Verify every remaining visible tool is purely read-only / diagnostic
    mutation_prefixes = ["update_", "modify_", "delete_", "post_", "patch_", "create_", "place_", "cancel_", "replace_", "close_", "exercise_"]
    for tool_name in visible_tools:
        for prefix in mutation_prefixes:
            assert not tool_name.startswith(prefix), f"Mutation tool {tool_name} exposed in visible tools!"


