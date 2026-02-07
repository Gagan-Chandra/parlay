# loan_app/src/llm.py
from __future__ import annotations
import os
import json
from typing import List, Dict, Any, Optional, Tuple

# Optional deps — we handle missing modules gracefully.
try:
    import openai  # openai>=1.0
except Exception:
    openai = None

try:
    import anthropic  # anthropic SDK
except Exception:
    anthropic = None


class LLMUnavailable(Exception):
    pass


def llm_provider_available() -> bool:
    """Return True if *either* OpenAI or Anthropic is available & configured."""
    oai = (openai is not None) and bool(os.getenv("OPENAI_API_KEY"))
    claude = (anthropic is not None) and bool(os.getenv("ANTHROPIC_API_KEY"))
    return bool(oai or claude)


def _call_openai_chat(
    system_prompt: str,
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict[str, Any]]] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> Dict[str, Any]:
    if openai is None or not os.getenv("OPENAI_API_KEY"):
        raise LLMUnavailable("OpenAI not configured")

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    oai_msgs = [{"role": "system", "content": system_prompt}] + messages

    resp = client.chat.completions.create(
        model=model,
        messages=oai_msgs,
        tools=tools or None,
        tool_choice="auto" if tools else None,
        temperature=temperature,
    )
    choice = resp.choices[0]
    return {
        "provider": "openai",
        "content": choice.message.content or "",
        "tool_calls": getattr(choice.message, "tool_calls", []),
        "raw": resp,
    }


def _call_anthropic_chat(
    system_prompt: str,
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict[str, Any]]] = None,
    model: str = "claude-3-5-sonnet-20240620",
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Simple Anthropic call wrapper (no native tool-calling here).
    We emulate tool routing by letting the model output JSON markers if needed.
    """
    if anthropic is None or not os.getenv("ANTHROPIC_API_KEY"):
        raise LLMUnavailable("Anthropic not configured")

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Convert to Anthropic-style messages
    anth_msgs = []
    # Put system prompt into first user msg as a prefix
    sys_txt = f"[SYSTEM]\n{system_prompt}\n[/SYSTEM]"
    for m in messages:
        anth_msgs.append({"role": m["role"], "content": m.get("content", "")})
    if len(anth_msgs) == 0 or anth_msgs[0]["role"] != "system":
        anth_msgs = [{"role": "user", "content": sys_txt}] + anth_msgs

    resp = client.messages.create(
        model=model,
        max_tokens=1000,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
    )
    # Anthropic returns a structured object; we join text blocks
    content_text = ""
    try:
        for blk in resp.content:
            if blk.type == "text":
                content_text += blk.text
    except Exception:
        content_text = str(resp)

    return {
        "provider": "anthropic",
        "content": content_text,
        "tool_calls": [],  # No built-in function calling here
        "raw": resp,
    }


def call_llm(
    system_prompt: str,
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict[str, Any]]] = None,
    provider_priority: Tuple[str, str] = ("openai", "anthropic"),
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Try OpenAI first then Anthropic, unless provider_priority says otherwise.
    Returns dict with: { provider, content, tool_calls, raw }
    Raises LLMUnavailable if neither provider is configured.
    """
    provs = list(provider_priority)

    last_err = None
    for prov in provs:
        try:
            if prov == "openai":
                return _call_openai_chat(system_prompt, messages, tools=tools, temperature=temperature)
            elif prov == "anthropic":
                return _call_anthropic_chat(system_prompt, messages, tools=tools, temperature=temperature)
        except LLMUnavailable as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue

    raise LLMUnavailable(f"No LLM provider available: {last_err}")


# ----------------------------
# Tool schema helpers for OpenAI function calling
# ----------------------------
def tool_schema_compute_eligibility() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "compute_eligibility_for_dataset",
            "description": "Compute rule-based eligibility (multi-loan) for a set of applicants.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rules": {"type": "object", "description": "Rules dictionary to apply"},
                    "applicant_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Indices of applicants to evaluate. If empty, evaluate all."
                    }
                },
                "required": ["rules"]
            }
        }
    }


def tool_schema_applicant_summary() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "summarize_applicant",
            "description": "Return strengths, risks, score, label, and recommendation for one applicant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_index": {"type": "integer"}
                },
                "required": ["applicant_index"]
            }
        }
    }


def parse_tool_call(tool_call: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Extract function name + JSON args from an OpenAI tool call object.
    """
    fn = getattr(tool_call, "function", None)
    if not fn:
        return "", {}
    name = getattr(fn, "name", "")
    raw_args = getattr(fn, "arguments", "{}")
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
    except Exception:
        args = {}
    return name, args
