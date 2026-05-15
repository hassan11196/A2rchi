"""MCP tool guardrails: classifier + decision helpers.

Background
----------
MCP tools served by remote servers (filesystem, GitHub, Postgres, kubectl, ...)
can perform write or execute operations on systems users care about.  Before
archi invokes such a tool on a user's behalf the user should explicitly
authorize it, the way the Claude Code CLI prompts for permission before
sensitive shell commands.

This module is the pure-logic half of that gate.  It classifies an MCP tool
as one of three sensitivity levels — ``safe``, ``write``, ``execute`` —
combining a heuristic name/description scan with per-server and per-tool
overrides read from the deployment config.  It does **not** itself prompt or
interrupt; that lives in the runtime wrapper that consults the classifier
plus the approval service.

Configuration
-------------
Each entry in ``mcp_servers_config`` (already JSONB on ``static_config``) may
carry optional keys read by this module:

    my_server:
      transport: streamable_http
      url: https://...
      requires_approval: true            # alias for "write" (covers write + execute)
      # or: requires_approval: "execute" / "write" / "all" / false
      tool_overrides:                    # exact tool-name → classification
        list_files: safe
        delete_file: execute
      auto_approve_principals:           # user_ids that bypass approval entirely
        - admin-bot

Classification rules
--------------------
1. If ``tool_overrides[tool.name]`` is set, return it.
2. If ``requires_approval`` is ``"all"`` or ``True`` or matches the heuristic
   class (e.g. heuristic says ``write`` and config says ``"write"``), return
   the heuristic class but never weaker than the server-wide floor.
3. Otherwise return the heuristic class derived from name + description.

The heuristic is deliberately conservative: anything matching a write/execute
keyword in name OR description is flagged.  False positives are acceptable;
false negatives are not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

Sensitivity = Literal["safe", "write", "execute"]

_VALID_SENSITIVITIES: tuple[Sensitivity, ...] = ("safe", "write", "execute")

# Heuristic regexes — applied to lowercased tool.name and tool.description.
# Ordering matters: execute > write > safe.  The "execute" patterns cover
# commands that run external processes or modify infrastructure; "write"
# covers mutations of stored state.
#
# Tool names are typically snake_case (``delete_user``, ``run_shell``).
# Python's ``\b`` does not treat ``_`` as a boundary because ``_`` is a word
# character — so ``\bdelete\b`` fails on ``delete_user``.  We use explicit
# lookarounds that treat any non-letter (including ``_``, digit, end-of-string)
# as a boundary.
_NAME_BOUNDARY_PREFIX = r"(?:^|[^a-z])"
_NAME_BOUNDARY_SUFFIX = r"(?=$|[^a-z])"

_EXECUTE_NAME_RE = re.compile(
    _NAME_BOUNDARY_PREFIX
    + r"(?:"
    r"exec(?:ute)?|run|spawn|shell|bash|command|deploy|invoke|"
    r"restart|reboot|kill|terminate|launch|kubectl|ssh"
    r")"
    + _NAME_BOUNDARY_SUFFIX
)
_WRITE_NAME_RE = re.compile(
    _NAME_BOUNDARY_PREFIX
    + r"(?:"
    r"create|update|delete|insert|write|patch|put|post|set|"
    r"modify|edit|append|upload|push|merge|rename|move|copy|"
    r"rm|drop|truncate|revoke|grant|enable|disable|"
    r"send|email|sms|pay|charge|refund"
    r")"
    + _NAME_BOUNDARY_SUFFIX
)
_EXECUTE_DESC_RE = re.compile(
    r"\b(execute|runs?\s+a\s+command|spawns?|starts?\s+a\s+process|"
    r"side[-\s]*effect|deletes?\s+files?|restarts?)\b",
    re.IGNORECASE,
)
_WRITE_DESC_RE = re.compile(
    r"\b(creates?|updates?|deletes?|writes?|modifies?|inserts?|"
    r"sends?|uploads?|appends?|persists?\s+to)\b",
    re.IGNORECASE,
)


def _coerce_sensitivity(value: Any) -> Optional[Sensitivity]:
    """Coerce a config value to a Sensitivity literal, else None."""
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _VALID_SENSITIVITIES:
            return v  # type: ignore[return-value]
    return None


def _heuristic_classify(name: str, description: str) -> Sensitivity:
    """Classify a tool by its name and description text alone."""
    name = (name or "").lower()
    desc = description or ""
    if _EXECUTE_NAME_RE.search(name) or _EXECUTE_DESC_RE.search(desc):
        return "execute"
    if _WRITE_NAME_RE.search(name) or _WRITE_DESC_RE.search(desc):
        return "write"
    return "safe"


def _server_floor(server_cfg: Optional[Dict[str, Any]]) -> Sensitivity:
    """Return the minimum classification implied by ``requires_approval``.

    A floor of ``write`` means ``safe`` heuristics get bumped up to ``write``,
    but tools heuristically classified as ``execute`` stay ``execute``.
    """
    if not server_cfg:
        return "safe"
    raw = server_cfg.get("requires_approval", False)
    if raw is True or (isinstance(raw, str) and raw.lower() == "all"):
        return "write"
    coerced = _coerce_sensitivity(raw)
    return coerced or "safe"


def _max_sensitivity(*levels: Sensitivity) -> Sensitivity:
    """Return the strongest of ``levels``."""
    rank = {"safe": 0, "write": 1, "execute": 2}
    best: Sensitivity = "safe"
    best_rank = -1
    for lvl in levels:
        if rank[lvl] > best_rank:
            best, best_rank = lvl, rank[lvl]
    return best


@dataclass(frozen=True)
class Classification:
    """Outcome of classifying one MCP tool against one server's config."""

    sensitivity: Sensitivity
    reason: str
    auto_approve_principals: frozenset[str]

    @property
    def requires_approval(self) -> bool:
        return self.sensitivity != "safe"


def classify_tool(
    tool_name: str,
    tool_description: str,
    server_cfg: Optional[Dict[str, Any]],
) -> Classification:
    """Classify a single MCP tool.

    Resolution order:
      1. ``tool_overrides[tool_name]`` — exact-match override
      2. heuristic on name + description
      3. floor implied by ``requires_approval`` on the server
    """
    server_cfg = server_cfg or {}

    overrides = server_cfg.get("tool_overrides") or {}
    explicit = _coerce_sensitivity(overrides.get(tool_name) if isinstance(overrides, dict) else None)
    if explicit is not None:
        return Classification(
            sensitivity=explicit,
            reason=f"tool_overrides[{tool_name!r}]={explicit!r}",
            auto_approve_principals=_extract_principals(server_cfg),
        )

    heuristic = _heuristic_classify(tool_name, tool_description)
    floor = _server_floor(server_cfg)
    final = _max_sensitivity(heuristic, floor)
    reason = (
        f"heuristic={heuristic!r}"
        + (f", floor={floor!r}" if floor != "safe" else "")
    )
    return Classification(
        sensitivity=final,
        reason=reason,
        auto_approve_principals=_extract_principals(server_cfg),
    )


def _extract_principals(server_cfg: Dict[str, Any]) -> frozenset[str]:
    principals = server_cfg.get("auto_approve_principals") or []
    if isinstance(principals, str):
        principals = [principals]
    if not isinstance(principals, Iterable):
        return frozenset()
    return frozenset(str(p) for p in principals if p)


def is_auto_approved(classification: Classification, user_id: Optional[str]) -> bool:
    """Return True if *user_id* is allowed to bypass the approval prompt."""
    if not classification.requires_approval:
        return True
    if not user_id:
        return False
    return user_id in classification.auto_approve_principals


# ---------------------------------------------------------------------------
# Approval-command parsing (used by Mattermost / API / future surfaces)
# ---------------------------------------------------------------------------

# The bot includes the approval_id in its message when it gates a tool call.
# Users reply with one of:
#     approve <id>
#     deny <id>
#     /approve <id>          (slash-command style)
# matching is case-insensitive and tolerant of leading/trailing whitespace.

_APPROVAL_COMMAND_RE = re.compile(
    r"^\s*/?(?P<decision>approve|approved|deny|denied)\b[\s:]+(?P<approval_id>[A-Za-z0-9_-]{8,64})\b",
    re.IGNORECASE,
)


def parse_approval_command(text: str) -> Optional[tuple[str, str]]:
    """Parse a free-text reply into a ``(decision, approval_id)`` pair.

    ``decision`` is always normalised to ``"approved"`` or ``"denied"``.
    Returns ``None`` for any text that doesn't start with an approval verb.
    """
    if not text:
        return None
    m = _APPROVAL_COMMAND_RE.match(text)
    if not m:
        return None
    raw = m.group("decision").lower()
    decision = "approved" if raw.startswith("approv") else "denied"
    return decision, m.group("approval_id")
