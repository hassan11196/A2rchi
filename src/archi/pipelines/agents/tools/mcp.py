from __future__ import annotations
import os
from typing import List, Any, Tuple, Optional

import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.tools import BaseTool

from src.utils.config_access import get_mcp_servers_config, get_full_config
from src.utils.logging import get_logger
from src.archi.pipelines.agents.utils.skill_utils import load_skill

logger = get_logger(__name__)

_CERN_CA_BUNDLE = "/etc/ssl/certs/tls-ca-bundle.pem"

# archi-only config keys that the MCP client doesn't understand.  These are
# consumed elsewhere (compose template, stdio install, post-load tool
# customization, SSO injection) and must be stripped before handing the dict
# to MultiServerMCPClient.
_ARCHI_ONLY_FIELDS = {
    "env_from_secrets", "host_file_mounts", "build_context", "image",
    "path", "skill", "sso_auth",
}


def _make_httpx_factory(ca_bundle: str):
    """Return an httpx_client_factory that uses the given CA bundle for SSL verification."""
    def factory(
        headers: dict | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers=headers or {},
            timeout=timeout,
            auth=auth,
            verify=ca_bundle,
            follow_redirects=True,
        )
    return factory


async def initialize_mcp_client(
    user_id: Optional[str] = None,
) -> Tuple[Optional[MultiServerMCPClient], List[BaseTool], str]:
    """
    Initializes the MCP client and fetches tool definitions.

    Args:
        user_id: SSO user ID used to look up a valid MCP OAuth token from the DB
                 for servers configured with sso_auth: true.

    Returns:
        client: The active client instance (must be kept alive by the caller).
        tools: The list of LangChain-compatible tools.
        skills_text: Concatenated skill content from all MCP servers that declare
            a `skill`. Empty string if no server has a skill. The caller is
            responsible for appending this to the agent's system prompt — we inject
            here only once per agent rather than into each tool description, so
            the content doesn't multiply by tool count.
    """
    from src.utils.mcp_oauth_service import MCPOAuthService

    mcp_servers = get_mcp_servers_config()
    _mcp_oauth = MCPOAuthService()

    _use_cern_ca = os.path.exists(_CERN_CA_BUNDLE)
    if _use_cern_ca:
        logger.info(f"Using CERN CA bundle for MCP SSL verification: {_CERN_CA_BUNDLE}")

    client_configs: dict[str, dict] = {}
    server_skills: dict[str, str] = {}
    full_config = get_full_config()

    for name, server_cfg in mcp_servers.items():
        # SSO-gated server: skip when we have no valid token. At boot
        # (user_id=None) the server is picked up later by the per-user
        # _build_mcp_tools() call, which passes the chat user's user_id so
        # the token is fetched from mcp_oauth_tokens. Registering without a
        # token would 401 on the MCP initialize handshake.
        requires_sso = server_cfg.get('sso_auth', False)
        access_token = (
            _mcp_oauth.get_access_token(user_id, name)
            if requires_sso and user_id
            else None
        )
        if requires_sso and not access_token:
            logger.info(
                f"Skipping MCP server '{name}': sso_auth=true but no valid "
                f"token for user_id={user_id!r}"
            )
            continue

        # Load any declared skill so we can append it to the agent system prompt.
        skill_name = server_cfg.get("skill")
        if skill_name:
            skill_content = load_skill(skill_name, full_config)
            if skill_content:
                server_skills[name] = skill_content

        # Strip archi-only fields the MCP client doesn't understand.
        cfg = {k: v for k, v in server_cfg.items() if k not in _ARCHI_ONLY_FIELDS}

        if requires_sso:
            cfg.setdefault('headers', {})['Authorization'] = f'Bearer {access_token}'

        transport = cfg.get("transport")
        if transport == "stdio":
            # stdio subprocesses inherit nothing by default (mcp.client.stdio uses
            # an empty env). Forward the parent process env so stdio MCP servers
            # see what they need.
            cfg["env"] = {**os.environ, **(cfg.get("env") or {})}
        else:
            # For HTTP-based transports, `env` is for the sidecar container
            # (compose), not the MCP client connection — drop it here.
            cfg.pop("env", None)

        # Inject CERN CA bundle for SSE / streamable_http transports.
        if _use_cern_ca and transport in ('sse', 'streamable_http'):
            cfg['httpx_client_factory'] = _make_httpx_factory(_CERN_CA_BUNDLE)

        client_configs[name] = cfg

    logger.info(f"Configuring MCP client with servers: {list(client_configs.keys())}")
    client = MultiServerMCPClient(client_configs)

    all_tools: List[BaseTool] = []
    failed_servers: dict[str, str] = {}

    for name in client_configs.keys():
        try:
            tools = await client.get_tools(server_name=name)
            for tool in tools:
                # Return error messages to the LLM instead of crashing the agent chain.
                tool.handle_tool_error = True
                # Tag with originating MCP server so downstream guardrails can
                # look up per-server policy from mcp_servers_config.
                try:
                    tool._archi_server_name = name  # type: ignore[attr-defined]
                except Exception:
                    pass
                logger.info(f"Loaded tool from MCP server '{name}': {tool.name} - {tool.description}")
            all_tools.extend(tools)
        except Exception as e:
            logger.error(f"Failed to fetch tools from MCP server '{name}': {e}")
            failed_servers[name] = str(e)

    logger.info(f"Active MCP servers: {[n for n in client_configs if n not in failed_servers]}")
    logger.warning(f"Failed MCP servers: {list(failed_servers.keys())}")

    # Build a single combined skills block keyed by server name — this is appended
    # to the agent's system prompt once, rather than duplicated across every tool.
    skills_parts: List[str] = []
    for name, skill_content in server_skills.items():
        if name not in failed_servers:
            skills_parts.append(
                f"\n--- {name} MCP Server Domain Knowledge ---\n{skill_content}"
            )
    skills_text = "".join(skills_parts)

    return client, all_tools, skills_text
