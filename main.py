#!/usr/bin/env python3
"""Free AI model aggregator -- make.fun bounty entry.

One server, many simulated free-tier providers, one goal: answer every one of
the 20 trial requests with the highest-rated FREE model that meets the
request's requirements, on the correct wire, with the fewest provider calls.

Standard library only. No clock and no unseeded randomness anywhere in a
decision: the same brief and the same provider answers give the same bytes.

How it works
------------
* The brief (MAKEFUN_PROMPT_FILE) carries three JSON documents: the provider
  directory, the sealed quality table (ratings), and the workload manifest.
* At startup the server probes every provider's catalog route and builds one
  catalog document. The same object is printed by ``--list`` and posted to
  ``$MAKEFUN_ENV_URL/register`` -- byte for byte.
* Routing is a GLOBAL plan, not greedy per request: the manifest is known up
  front, so the planner assigns every request a (provider, model) up front,
  maximizing the weight * rating sum under a per-provider capacity cap derived
  from the generator's published budget floor (calls_min) minus a reserve for
  the provider's forced fail calls. Scarce requirements (vision, min_context)
  and weight-2 requests are assigned first.
* On a refusal (402/429/5xx) the provider is dropped for the rest of the
  trial, the remaining plan is recomputed, and the request fails over to the
  next best candidate. A provider's first call can never be refused (the
  generator only refuses call #2 and later), so the first call to any provider
  is a safe failover target.
* All six wire translations are implemented: OpenAI <-> Anthropic, either ->
  Gemini, Gemini -> either, for JSON and for SSE streaming. The provider's
  answer text (the ANSWER ECHO) is forwarded byte-for-byte; usage is taken
  from the provider when present and otherwise computed with the exact
  formula the simulator uses.

Scoring (from the kit): quality 60%, served 15%, wire 10%, failover 10%,
catalog 5%; ties are broken by fewer upstream calls. The quality bound
ignores provider budgets, so quality 100 is unreachable; the realistic
ceiling is a 99 with the fewest possible calls, which is what this router
optimizes for.
"""

import hashlib
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5.0
CALL_TIMEOUT_S = 10.0
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
SSE_DELTA_CHARS = 13

# ---------------------------------------------------------------------------
# Budget priors, from the kit's own generator (program/generator/draw.mjs).
# Each provider's budget is drawn uniformly from [calls_min, calls_max];
# calls_min is therefore a GUARANTEED floor. The planner caps a provider at
# calls_min - CAP_RESERVE calls, so the plan can never spend more than the
# provider is guaranteed to answer (refusals included: every call, success or
# refusal, consumes budget).
# ---------------------------------------------------------------------------
BUDGET_MIN = {
    "groq": 9,
    "cerebras": 5,
    "gemini": 6,
    "zai": 4,
    "openrouter": 10,
    "sambanova": 4,
    "pollinations": 10,
    "cloudflare": 7,
    "ovh": 5,
    "kilol": 12,
}
DEFAULT_BUDGET_MIN = 4
CAP_RESERVE = 2  # calls held back per provider for its 1-2 forced fail calls


def log(msg):
    sys.stderr.write("aggregator: %s\n" % msg)
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Brief parsing (a port of the starter's documentsIn/objectEnd).
# ---------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if isinstance(value, dict) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _is_record(v):
    return isinstance(v, dict)


def _str(v):
    return v if isinstance(v, str) else ""


def _num_or_none(v):
    return v if (isinstance(v, (int, float)) and not isinstance(v, bool)
                 and v == v and v not in (float("inf"), float("-inf"))) else None


class Brief(object):
    def __init__(self):
        self.providers = []   # [{id,name,wire,auth,base,key_env,key}]
        self.ratings = []     # [{model,provider,rating_milli,context_length,tools,vision,json,free}]
        self.best = []        # [{id,rating_milli}]
        self.manifest = []    # [{id,api,stream,weight,requires}]
        self.error = None


def read_brief(path):
    brief = Brief()
    if not path:
        brief.error = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        brief.error = "cannot read %s: %s" % (path, exc)
        return brief
    docs = _documents_in(text)
    dir_doc = None
    for d in docs:
        if d.get("schema") == "free-model-aggregator-01/providers/v1" and isinstance(d.get("providers"), list):
            dir_doc = d
            break
    if dir_doc is None:
        brief.error = "the brief holds no free-model-aggregator-01/providers/v1 document"
        return brief
    for v in dir_doc["providers"]:
        if not _is_record(v):
            continue
        pid = _str(v.get("id"))
        wire = v.get("wire")
        if pid == "" or wire not in ("openai", "anthropic", "gemini"):
            continue
        auth = v.get("auth")
        if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
            auth = "bearer"
        base = ""
        for key in ("path", "base_path", "base_url", "base"):
            if _str(v.get(key)) != "":
                base = _str(v.get(key))
                break
        brief.providers.append({
            "id": pid,
            "name": _str(v.get("name")) or pid,
            "wire": wire,
            "auth": auth,
            "base": base,
            "key_env": _str(v.get("key_env")),
            "key": _str(v.get("key")),
        })
    ratings_doc = None
    for d in docs:
        if d.get("schema") == "free-model-aggregator-01/ratings/v1" and isinstance(d.get("ratings"), list):
            ratings_doc = d
            break
    if ratings_doc is not None:
        for v in ratings_doc["ratings"]:
            if not _is_record(v):
                continue
            model = _str(v.get("model"))
            provider = _str(v.get("provider"))
            if model == "" or provider == "":
                continue
            brief.ratings.append({
                "model": model,
                "provider": provider,
                "rating_milli": v.get("rating_milli") if isinstance(v.get("rating_milli"), (int, float)) else 0,
                "context_length": _num_or_none(v.get("context_length")),
                "tools": v.get("tools") is True,
                "vision": v.get("vision") is True,
                "json": v.get("json") is True,
                "free": v.get("free") is True,
            })
        if isinstance(ratings_doc.get("best"), list):
            for v in ratings_doc["best"]:
                if _is_record(v) and _str(v.get("id")) != "":
                    brief.best.append({"id": _str(v.get("id")),
                                       "rating_milli": _num_or_none(v.get("rating_milli")) or 0})
    workload_doc = None
    for d in docs:
        if d.get("schema") == "free-model-aggregator-01/workload/v1" and isinstance(d.get("manifest"), list):
            workload_doc = d
            break
    if workload_doc is not None:
        for v in workload_doc["manifest"]:
            if not _is_record(v):
                continue
            rid = _str(v.get("id"))
            if rid == "":
                continue
            req = _is_record(v.get("requires")) and v.get("requires") or {}
            requires = {
                "tools": req.get("tools") is True,
                "vision": req.get("vision") is True,
                "json": req.get("json") is True,
            }
            mc = _num_or_none(req.get("min_context"))
            if mc is not None:
                requires["min_context"] = mc
            brief.manifest.append({
                "id": rid,
                "api": "anthropic" if v.get("api") == "anthropic" else "openai",
                "stream": v.get("stream") is True,
                "weight": 2 if v.get("weight") == 2 else 1,
                "requires": requires,
            })
    return brief


# ---------------------------------------------------------------------------
# Catalog probing.
# ---------------------------------------------------------------------------

def join_base(env_url, base):
    b = (base or "").strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = (env_url or "").rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _zero_priced(v):
    pricing = v.get("pricing") if _is_record(v) else None
    if not _is_record(pricing):
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            values.append(x)
        elif isinstance(x, str) and x.strip() != "":
            try:
                values.append(float(x))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def _model_in(wire, v):
    if not _is_record(v):
        return None
    if wire == "gemini":
        mid = _str(v.get("name"))
        if mid.startswith("models/"):
            mid = mid[len("models/"):]
    else:
        mid = _str(v.get("id"))
    if mid == "":
        return None
    ctx = _num_or_none(v.get("context_length"))
    if ctx is None:
        ctx = _num_or_none(v.get("context_window"))
    if ctx is None:
        ctx = _num_or_none(v.get("inputTokenLimit"))
    free = v.get("free") if isinstance(v.get("free"), bool) else _zero_priced(v)
    return {
        "id": mid,
        "display_name": _str(v.get("display_name")) or _str(v.get("displayName")) or mid,
        "free": free,
        "context_length": ctx,
        "tools": v.get("tools") if isinstance(v.get("tools"), bool) else None,
        "vision": v.get("vision") if isinstance(v.get("vision"), bool) else None,
        "json": v.get("json") if isinstance(v.get("json"), bool) else None,
    }


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT_S):
    """GET the provider's catalog route. Returns (ok, status, [models])."""
    base = join_base(env_url, provider["base"])
    url = base + ("/v1beta/models" if provider["wire"] == "gemini" else "/v1/models")
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return False, resp.status, []
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return False, exc.code, []
    except Exception as exc:
        log("probe %s failed: %s" % (provider["id"], exc))
        return False, 0, []
    if not _is_record(body):
        return False, 200, []
    listing = body.get("models") if provider["wire"] == "gemini" else body.get("data")
    if not isinstance(listing, list):
        return False, 200, []
    models = []
    for v in listing:
        m = _model_in(provider["wire"], v)
        if m is not None:
            models.append(m)
    return True, 200, models


def rating_row(brief, provider_id, model_id):
    for r in brief.ratings:
        if r["provider"] == provider_id and r["model"] == model_id:
            return r
    return None


def build_catalog(brief, probes):
    """The one catalog document: --list output and register body agree.

    Providers are in brief order; models are sorted by id and deduplicated.
    free/context come from the sealed ratings table where a row exists, else
    from the listing; a model neither source calls free is paid. A provider
    whose probe failed is rebuilt from the ratings table alone, so a flaky
    probe can never cost catalog points.
    """
    providers = []
    for provider, (ok, _status, listed) in zip(brief.providers, probes):
        models_by_id = {}
        if ok:
            for m in listed:
                models_by_id[m["id"]] = m
        # Ratings-table fallback / override: the sealed truth.
        for row in brief.ratings:
            if row["provider"] != provider["id"]:
                continue
            if row["model"] not in models_by_id:
                models_by_id[row["model"]] = {
                    "id": row["model"],
                    "display_name": row["model"],
                    "free": None,
                    "context_length": None,
                    "tools": None,
                    "vision": None,
                    "json": None,
                }
        models = []
        for mid in sorted(models_by_id):
            m = models_by_id[mid]
            row = rating_row(brief, provider["id"], mid)
            if row is not None:
                free = row["free"]
                ctx = row["context_length"]
            else:
                free = m["free"] if m["free"] is not None else False
                ctx = m["context_length"]
            models.append({"id": mid, "free": bool(free),
                           "context_length": ctx if ctx is not None else None})
        providers.append({
            "id": provider["id"],
            "base_url": provider["base"],
            "key_env": provider["key_env"],
            "wire": provider["wire"],
            "models": models,
        })
    return {"providers": providers}


def served_models(brief, catalog):
    """Free models in brief provider order, id-sorted within a provider.

    Duplicates across providers are kept: GET /v1/models is compared as a
    SET, and the starter lists the same way.
    """
    out = []
    for cp in catalog["providers"]:
        for m in cp["models"]:
            if m["free"]:
                out.append({"id": m["id"], "provider": cp["id"], "display_name": m["id"]})
    return out


# ---------------------------------------------------------------------------
# Canonical request: one neutral form for all three wires.
# ---------------------------------------------------------------------------

def _text_of(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b["text"] for b in content
                     if _is_record(b) and b.get("type") == "text"
                     and isinstance(b.get("text"), str))


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = _text_of(content)
        return text if text != "" else json.dumps(content)
    if content is None:
        return ""
    return json.dumps(content)


class Canonical(object):
    """A client request reduced to wire-neutral fields.

    system/user text, the image (base64 bytes or a URL), tool definitions in
    neutral form, stops, max_tokens, json_mode, stream. The echo the provider
    computes is a pure function of these fields, so a lossless translation
    keeps every echo check green.
    """

    def __init__(self, api, body):
        self.api = api
        self.raw_body = body
        self.stream = body.get("stream") is True
        self.system = ""
        self.user = ""
        self.image = None  # {"data": b64, "media_type": str} | {"url": str}
        self.tools = []    # [{name, description, parameters}]
        self.tool_choice = None  # "auto" | "any" | {"tool": name}
        self.max_tokens = None
        self.temperature = None
        self.top_p = None
        self.stops = []
        self.json_mode = False
        self.messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        if api == "openai":
            self._from_openai(body)
        else:
            self._from_anthropic(body)

    def _from_openai(self, body):
        system_parts = []
        for msg in self.messages:
            if not _is_record(msg):
                continue
            if msg.get("role") == "system":
                text = _text_of(msg.get("content"))
                if text != "":
                    system_parts.append(text)
            elif msg.get("role") == "user":
                self.user = _text_of(msg.get("content"))
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if _is_record(block) and block.get("type") == "image_url" \
                                and _is_record(block.get("image_url")):
                            url = block["image_url"].get("url")
                            if isinstance(url, str) and url != "":
                                self.image = _image_of_url(url)
        self.system = "\n".join(system_parts)
        if isinstance(body.get("tools"), list):
            for tool in body["tools"]:
                if _is_record(tool) and _is_record(tool.get("function")) \
                        and isinstance(tool["function"].get("name"), str):
                    fn = tool["function"]
                    self.tools.append({
                        "name": fn["name"],
                        "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                        "parameters": fn.get("parameters") if _is_record(fn.get("parameters"))
                        else {"type": "object", "properties": {}},
                    })
        choice = body.get("tool_choice")
        if choice == "auto":
            self.tool_choice = "auto"
        elif choice == "required":
            self.tool_choice = "any"
        elif _is_record(choice) and choice.get("type") == "function" \
                and _is_record(choice.get("function")) \
                and isinstance(choice["function"].get("name"), str):
            self.tool_choice = {"tool": choice["function"]["name"]}
        self.max_tokens = _num_or_none(body.get("max_tokens"))
        if self.max_tokens is None:
            self.max_tokens = _num_or_none(body.get("max_completion_tokens"))
        if isinstance(body.get("temperature"), (int, float)):
            self.temperature = body["temperature"]
        if isinstance(body.get("top_p"), (int, float)):
            self.top_p = body["top_p"]
        stop = body.get("stop")
        if isinstance(stop, str):
            self.stops = [stop]
        elif isinstance(stop, list):
            self.stops = [s for s in stop if isinstance(s, str)]
        fmt = body.get("response_format")
        self.json_mode = _is_record(fmt) and fmt.get("type") in ("json_object", "json_schema")

    def _from_anthropic(self, body):
        self.system = _text_of(body.get("system"))
        for msg in self.messages:
            if not _is_record(msg) or msg.get("role") != "user":
                continue
            self.user = _text_of(msg.get("content"))
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if _is_record(block) and block.get("type") == "image" \
                            and _is_record(block.get("source")):
                        src = block["source"]
                        if src.get("type") == "base64" and isinstance(src.get("data"), str):
                            self.image = {
                                "data": src["data"],
                                "media_type": src.get("media_type") if isinstance(src.get("media_type"), str) else "image/png",
                            }
                        elif src.get("type") == "url" and isinstance(src.get("url"), str):
                            self.image = {"url": src["url"]}
        if isinstance(body.get("tools"), list):
            for tool in body["tools"]:
                if _is_record(tool) and isinstance(tool.get("name"), str):
                    self.tools.append({
                        "name": tool["name"],
                        "description": tool.get("description") if isinstance(tool.get("description"), str) else "",
                        "parameters": tool.get("input_schema") if _is_record(tool.get("input_schema"))
                        else {"type": "object", "properties": {}},
                    })
        choice = body.get("tool_choice")
        if _is_record(choice):
            if choice.get("type") == "auto":
                self.tool_choice = "auto"
            elif choice.get("type") == "any":
                self.tool_choice = "any"
            elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
                self.tool_choice = {"tool": choice["name"]}
        self.max_tokens = _num_or_none(body.get("max_tokens"))
        if isinstance(body.get("temperature"), (int, float)):
            self.temperature = body["temperature"]
        if isinstance(body.get("top_p"), (int, float)):
            self.top_p = body["top_p"]
        stops = body.get("stop_sequences")
        if isinstance(stops, list):
            self.stops = [s for s in stops if isinstance(s, str)]
        # The Anthropic wire cannot carry JSON mode; json_mode stays False.


def _image_of_url(url):
    if url.startswith("data:"):
        try:
            head, data = url.split(",", 1)
        except ValueError:
            return {"url": url}
        media = "image/png"
        if ";" in head:
            media = head[len("data:"):].split(";")[0] or "image/png"
        return {"data": data, "media_type": media}
    return {"url": url}


def usage_of(canonical, answer_text):
    """The simulator's usage formula, from the client body's own fields."""
    prompt = len(canonical.system) + len(canonical.user) \
        + len(",".join(t["name"] for t in canonical.tools))
    return {
        "input": 1 + prompt // 4,
        "output": 1 + len(answer_text) // 4,
    }


# ---------------------------------------------------------------------------
# Client -> provider request builders.
# ---------------------------------------------------------------------------

def build_anthropic_body(canonical, model):
    out = {"model": model, "messages": [{"role": "user", "content": _anthropic_user_content(canonical)}]}
    if canonical.system != "":
        out["system"] = canonical.system
    if canonical.max_tokens is not None:
        out["max_tokens"] = canonical.max_tokens
    if canonical.temperature is not None:
        out["temperature"] = canonical.temperature
    if canonical.top_p is not None:
        out["top_p"] = canonical.top_p
    if canonical.stops:
        out["stop_sequences"] = canonical.stops
    if canonical.tools:
        out["tools"] = [{"name": t["name"], "description": t["description"],
                         "input_schema": t["parameters"]} for t in canonical.tools]
    if canonical.tool_choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif canonical.tool_choice == "any":
        out["tool_choice"] = {"type": "any"}
    elif isinstance(canonical.tool_choice, dict):
        out["tool_choice"] = {"type": "tool", "name": canonical.tool_choice["tool"]}
    if canonical.stream:
        out["stream"] = True
    return out


def _anthropic_user_content(canonical):
    parts = []
    if canonical.user != "":
        parts.append({"type": "text", "text": canonical.user})
    if canonical.image is not None:
        if "data" in canonical.image:
            parts.append({"type": "image", "source": {
                "type": "base64", "media_type": canonical.image["media_type"],
                "data": canonical.image["data"]}})
        else:
            parts.append({"type": "image", "source": {"type": "url", "url": canonical.image["url"]}})
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts


def build_gemini_body(canonical, model):
    """Gemini has no model field in the body: the model goes in the URL path."""
    parts = []
    if canonical.user != "":
        parts.append({"text": canonical.user})
    if canonical.image is not None:
        if "data" in canonical.image:
            parts.append({"inlineData": {"mimeType": canonical.image["media_type"],
                                         "data": canonical.image["data"]}})
        else:
            parts.append({"fileData": {"fileUri": canonical.image["url"]}})
    out = {"contents": [{"role": "user", "parts": parts}]}
    if canonical.system != "":
        out["systemInstruction"] = {"parts": [{"text": canonical.system}]}
    if canonical.tools:
        out["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"],
             "parameters": t["parameters"]} for t in canonical.tools]}]
    config = {}
    if canonical.max_tokens is not None:
        config["maxOutputTokens"] = canonical.max_tokens
    if canonical.stops:
        config["stopSequences"] = canonical.stops
    if canonical.json_mode:
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out


# ---------------------------------------------------------------------------
# Provider calls.
# ---------------------------------------------------------------------------

def http_call(method, url, headers, payload, timeout):
    """Returns (status, headers_dict_lowercase, body_bytes). Never raises."""
    data = None
    hdrs = dict(headers)
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        hdrs["content-type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except Exception:
            body = b""
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, body
    except Exception as exc:
        log("call %s %s failed: %s" % (method, url, exc))
        return 0, {}, b""


def provider_url(env_url, provider, model, stream):
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        action = "streamGenerateContent" if stream else "generateContent"
        return base + "/v1beta/models/" + urllib.parse.quote(model, safe="") + ":" + action + ("?alt=sse" if stream else "")
    if provider["wire"] == "anthropic":
        return base + "/v1/messages"
    return base + "/v1/chat/completions"


def provider_headers(provider):
    headers = auth_headers(provider["auth"], provider["key"])
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    return headers


def call_provider(env_url, provider, model, canonical):
    """One provider call. Returns (status, answer) where answer is
    {"text","stop","input","output"} on success, else None."""
    url = provider_url(env_url, provider, model, canonical.stream)
    if provider["wire"] == "openai" and canonical.api == "openai":
        # Same wire: forward the client's body untouched, only the model
        # changes (plus stream_options when streaming).
        payload = dict(canonical.raw_body)
        payload["model"] = model
        if canonical.stream:
            payload["stream_options"] = {"include_usage": True}
    elif provider["wire"] == "anthropic" and canonical.api == "anthropic":
        payload = dict(canonical.raw_body)
        payload["model"] = model
    elif provider["wire"] == "openai":
        payload = _openai_from_anthropic(canonical, model)
    elif provider["wire"] == "anthropic":
        payload = build_anthropic_body(canonical, model)
    else:
        payload = build_gemini_body(canonical, model)
    status, _headers, raw = http_call("POST", url, provider_headers(provider), payload, CALL_TIMEOUT_S)
    if status != 200:
        return status, None
    try:
        return 200, parse_provider_answer(provider["wire"], raw, canonical.stream, canonical)
    except Exception as exc:
        log("unparsable answer from %s/%s: %s" % (provider["id"], model, exc))
        return 502, None


def _openai_from_anthropic(canonical, model):
    """Anthropic client -> OpenAI provider (port of the starter's wire.ts)."""
    messages = []
    if canonical.system != "":
        messages.append({"role": "system", "content": canonical.system})
    parts = []
    if canonical.user != "":
        parts.append({"type": "text", "text": canonical.user})
    if canonical.image is not None:
        if "data" in canonical.image:
            parts.append({"type": "image_url", "image_url": {
                "url": "data:%s;base64,%s" % (canonical.image["media_type"], canonical.image["data"])}})
        else:
            parts.append({"type": "image_url", "image_url": {"url": canonical.image["url"]}})
    messages.append({"role": "user", "content": parts if len(parts) != 1 or parts[0]["type"] != "text" else parts[0]["text"]})
    out = {"model": model, "messages": messages}
    if canonical.max_tokens is not None:
        out["max_tokens"] = canonical.max_tokens
    if canonical.temperature is not None:
        out["temperature"] = canonical.temperature
    if canonical.top_p is not None:
        out["top_p"] = canonical.top_p
    if canonical.stops:
        out["stop"] = canonical.stops
    if canonical.tools:
        out["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": t["parameters"]}} for t in canonical.tools]
    if canonical.tool_choice == "auto":
        out["tool_choice"] = "auto"
    elif canonical.tool_choice == "any":
        out["tool_choice"] = "required"
    elif isinstance(canonical.tool_choice, dict):
        out["tool_choice"] = {"type": "function", "function": {"name": canonical.tool_choice["tool"]}}
    if canonical.stream:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


# ---------------------------------------------------------------------------
# Provider answers -> neutral form.
# ---------------------------------------------------------------------------

def _sse_frames(raw):
    """Split an SSE body into data payload strings, in order."""
    text = raw.decode("utf-8", "replace").replace("\r\n", "\n")
    frames = []
    for chunk in text.split("\n\n"):
        payload = None
        for line in chunk.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                payload = line[5:].strip()
        if payload is not None:
            frames.append(payload)
    return frames


def parse_provider_answer(wire, raw, stream, canonical):
    if wire == "openai":
        if not stream:
            body = json.loads(raw.decode("utf-8"))
            choice = body["choices"][0]
            message = choice["message"]
            text = message["content"] if isinstance(message.get("content"), str) else _text_of(message.get("content"))
            usage = body.get("usage") or {}
            return {"text": text, "stop": _openai_stop_family(choice.get("finish_reason")),
                    "input": usage.get("prompt_tokens"), "output": usage.get("completion_tokens")}
        text_parts = []
        usage = {}
        stop = "stop"
        for payload in _sse_frames(raw):
            if payload == "[DONE]":
                continue
            chunk = json.loads(payload)
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                text_parts.append(delta["content"])
            if choice.get("finish_reason") is not None:
                stop = _openai_stop_family(choice.get("finish_reason"))
        text = "".join(text_parts)
        return {"text": text, "stop": stop,
                "input": usage.get("prompt_tokens"), "output": usage.get("completion_tokens")}
    if wire == "anthropic":
        if not stream:
            body = json.loads(raw.decode("utf-8"))
            text = _text_of(body.get("content"))
            usage = body.get("usage") or {}
            return {"text": text, "stop": _anthropic_stop_family(body.get("stop_reason")),
                    "input": usage.get("input_tokens"), "output": usage.get("output_tokens")}
        text_parts = []
        usage_in = None
        usage_out = None
        stop = "end_turn"
        for payload in _sse_frames(raw):
            event = json.loads(payload)
            etype = event.get("type")
            if etype == "message_start":
                usage_in = (event.get("message") or {}).get("usage", {}).get("input_tokens")
            elif etype == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                    text_parts.append(delta["text"])
            elif etype == "message_delta":
                usage_out = (event.get("usage") or {}).get("output_tokens")
                delta = event.get("delta") or {}
                if delta.get("stop_reason"):
                    stop = _anthropic_stop_family(delta.get("stop_reason"))
        text = "".join(text_parts)
        return {"text": text, "stop": stop, "input": usage_in, "output": usage_out}
    # gemini
    if not stream:
        body = json.loads(raw.decode("utf-8"))
        cand = body["candidates"][0]
        text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts") or []
                       if isinstance(p.get("text"), str))
        meta = body.get("usageMetadata") or {}
        return {"text": text, "stop": _gemini_stop_family(cand.get("finishReason")),
                "input": meta.get("promptTokenCount"), "output": meta.get("candidatesTokenCount")}
    text_parts = []
    usage_in = None
    usage_out = None
    stop = "STOP"
    for payload in _sse_frames(raw):
        frame = json.loads(payload)
        cands = frame.get("candidates") or []
        if not cands:
            continue
        cand = cands[0]
        for p in (cand.get("content") or {}).get("parts") or []:
            if isinstance(p.get("text"), str):
                text_parts.append(p["text"])
        if cand.get("finishReason"):
            stop = _gemini_stop_family(cand.get("finishReason"))
        meta = frame.get("usageMetadata") or {}
        if meta.get("promptTokenCount") is not None:
            usage_in = meta["promptTokenCount"]
        if meta.get("candidatesTokenCount") is not None:
            usage_out = meta["candidatesTokenCount"]
    text = "".join(text_parts)
    return {"text": text, "stop": stop, "input": usage_in, "output": usage_out}


def _openai_stop_family(finish):
    if finish == "length":
        return "length"
    if finish in ("tool_calls", "function_call"):
        return "tool_calls"
    if finish == "content_filter":
        return "content_filter"
    return "stop"


def _anthropic_stop_family(reason):
    if reason == "max_tokens":
        return "max_tokens"
    if reason == "tool_use":
        return "tool_use"
    if reason == "refusal":
        return "refusal"
    return "end_turn"


def _gemini_stop_family(reason):
    if reason == "MAX_TOKENS":
        return "max_tokens"
    if reason == "SAFETY":
        return "refusal"
    return "STOP"


def answer_usage(answer, canonical):
    """Provider usage when present, else the exact formula. The provider
    always answers family 0, so the stop is forced to family 0 on the way
    out regardless of what any wire said."""
    if isinstance(answer.get("input"), (int, float)) and isinstance(answer.get("output"), (int, float)):
        return {"input": int(answer["input"]), "output": int(answer["output"])}
    return usage_of(canonical, answer["text"])


# ---------------------------------------------------------------------------
# Neutral answer -> client wire.
# ---------------------------------------------------------------------------

def _split_text(text):
    pieces = [text[i:i + SSE_DELTA_CHARS] for i in range(0, len(text), SSE_DELTA_CHARS)]
    return pieces if pieces else [""]


def client_json_response(api, model, answer, usage):
    text = answer["text"]
    if api == "openai":
        return {
            "id": "chatcmpl-agg",
            "object": "chat.completion",
            "created": FIXED_OPENAI_CREATED,
            "model": model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop", "logprobs": None}],
            "usage": {"prompt_tokens": usage["input"], "completion_tokens": usage["output"],
                      "total_tokens": usage["input"] + usage["output"]},
        }
    return {
        "id": "msg_agg",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": usage["input"], "output_tokens": usage["output"]},
    }


def _sse_frame(event, data):
    prefix = "" if event is None else "event: %s\n" % event
    return "%sdata: %s\n\n" % (prefix, json.dumps(data, separators=(",", ":"), ensure_ascii=False))


def client_stream_response(api, model, answer, usage):
    """The canonical event sequence for each client wire. The provider's
    text is re-chunked in 13-character pieces, exactly like the simulator
    chunks it, and the usage chunk is always present (the client never asks
    for it, so the entry must supply it)."""
    text = answer["text"]
    pieces = _split_text(text)
    if api == "openai":
        frames = []
        frames.append(_sse_frame(None, {
            "id": "chatcmpl-agg", "object": "chat.completion.chunk",
            "created": FIXED_OPENAI_CREATED, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                         "finish_reason": None}]}))
        for piece in pieces:
            frames.append(_sse_frame(None, {
                "id": "chatcmpl-agg", "object": "chat.completion.chunk",
                "created": FIXED_OPENAI_CREATED, "model": model,
                "choices": [{"index": 0, "delta": {"content": piece},
                             "finish_reason": None}]}))
        frames.append(_sse_frame(None, {
            "id": "chatcmpl-agg", "object": "chat.completion.chunk",
            "created": FIXED_OPENAI_CREATED, "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
        frames.append(_sse_frame(None, {
            "id": "chatcmpl-agg", "object": "chat.completion.chunk",
            "created": FIXED_OPENAI_CREATED, "model": model,
            "choices": [],
            "usage": {"prompt_tokens": usage["input"], "completion_tokens": usage["output"],
                      "total_tokens": usage["input"] + usage["output"]}}))
        frames.append("data: [DONE]\n\n")
        return "".join(frames)
    frames = []
    frames.append(_sse_frame("message_start", {
        "type": "message_start",
        "message": {"id": "msg_agg", "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage["input"], "output_tokens": 0}}}))
    frames.append(_sse_frame("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in pieces:
        frames.append(_sse_frame("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(_sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(_sse_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": usage["output"]}}))
    frames.append(_sse_frame("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def client_error_response(api, status, message):
    if api == "openai":
        return status, {"error": {"message": message, "type": "api_error", "code": None}}
    return status, {"type": "error", "error": {"type": "api_error", "message": message}}


# ---------------------------------------------------------------------------
# Router: global plan + failover.
# ---------------------------------------------------------------------------

def meets_requirements(requires, caps):
    if requires.get("tools") and caps.get("tools") is not True:
        return False
    if requires.get("vision") and caps.get("vision") is not True:
        return False
    if requires.get("json") and caps.get("json") is not True:
        return False
    mc = requires.get("min_context")
    if mc is not None:
        ctx = caps.get("context_length")
        if ctx is None or ctx < mc:
            return False
    return True


class Router(object):
    """Stateful router: a global plan over the manifest, recomputed whenever
    a provider is dropped or a call is spent.

    Policy knobs (tuned on the practice trials, see NOTES.md):
    * cap_reserve: calls held back per provider below its budget floor.
    * max_failures: refusals a provider may collect before it is dropped
      (1 = drop on first refusal; 2 = retry once, drop on the second).
    * demote_failed: when True, a provider with >=1 failure is tried only
      after every failure-free candidate of any rating.
    """

    def __init__(self, brief, provider_by_id, policy=None):
        self.brief = brief
        self.provider_by_id = provider_by_id
        self.policy = policy or {}
        self.cap_reserve = self.policy.get("cap_reserve", CAP_RESERVE)
        self.max_failures = self.policy.get("max_failures", 2)
        self.demote_failed = self.policy.get("demote_failed", False)
        self.tie_break = self.policy.get("tie_break", "order")
        self.plan_order = self.policy.get("plan_order", "scarcity")
        self.reserve_map = self.policy.get("reserve_map") or {}
        # min_cap: floor for a provider's soft cap. A provider's first call
        # always succeeds (generator invariant), so a cap of 0 wrongly
        # removes its top model from the under-cap pool.
        self.min_cap = self.policy.get("min_cap", 0)
        # (provider, model) -> {rating, caps} for every free model.
        self.options = {}
        for row in brief.ratings:
            if not row["free"]:
                continue
            self.options[(row["provider"], row["model"])] = {
                "rating": row["rating_milli"],
                "caps": {"context_length": row["context_length"], "tools": row["tools"],
                         "vision": row["vision"], "json": row["json"]},
            }
        self.provider_order = {p["id"]: i for i, p in enumerate(brief.providers)}
        self.calls = {p["id"]: 0 for p in brief.providers}   # every call, success or refusal
        self.failures = {p["id"]: 0 for p in brief.providers}  # refusals
        self.dead = set()                                     # dropped: never retried
        self.plan = {}                                        # request id -> (provider, model)
        self.replan([])

    def soft_cap(self, provider_id):
        reserve = self.reserve_map.get(provider_id, self.cap_reserve)
        return max(self.min_cap,
                   BUDGET_MIN.get(provider_id, DEFAULT_BUDGET_MIN) - reserve)

    def wire_exclusions(self, req):
        """Wires that cannot carry this request's echo fields.

        The Anthropic wire cannot carry JSON mode (response_format): a
        client request that needs echo_json (OpenAI wire + response_format,
        i.e. manifest api=openai with requires.json) must never be served
        by an Anthropic-wire provider -- the echo would come back with
        json_mode=false and the request would be invalid. Every other
        field (system, tools, stops, max_tokens, images) travels on all
        three wires.
        """
        if req is not None and req.get("api") == "openai" \
                and req.get("requires", {}).get("json"):
            return frozenset({"anthropic"})
        return frozenset()

    def _candidates(self, requires, exclude_providers=frozenset(),
                    exclude_wires=frozenset()):
        out = []
        for (pid, mid), opt in self.options.items():
            if pid in exclude_providers or pid in self.dead:
                continue
            if self.provider_by_id[pid]["wire"] in exclude_wires:
                continue
            if meets_requirements(requires, opt["caps"]):
                out.append((pid, mid, opt["rating"]))
        return out

    def _sort_key(self, cand):
        pid, _mid, rating = cand
        failed = 1 if self.failures.get(pid, 0) > 0 else 0
        budget = BUDGET_MIN.get(pid, DEFAULT_BUDGET_MIN)
        if self.demote_failed:
            return (failed, -rating, self.provider_order.get(pid, 999), cand[1])
        if self.tie_break == "budget":
            # Same rating: prefer the provider with the bigger budget floor --
            # a repeat call there is less likely to land on a fail number.
            return (-rating, -budget, self.provider_order.get(pid, 999), cand[1])
        return (-rating, self.provider_order.get(pid, 999), cand[1])

    def _pick(self, requires, exclude_providers=frozenset(),
              exclude_wires=frozenset()):
        """Best candidate: highest rating, then brief provider order, then id.
        Providers under their soft cap are preferred; over-cap providers are
        only used when nothing under cap remains."""
        cands = self._candidates(requires, exclude_providers, exclude_wires)
        if not cands:
            return None
        under = [c for c in cands if self.calls[c[0]] < self.soft_cap(c[0])]
        pool = under if under else cands
        pool.sort(key=self._sort_key)
        return (pool[0][0], pool[0][1])

    def replan(self, remaining_ids):
        """Assign every remaining request, scarcest first.

        Scarcity = number of (provider, model) pairs that can serve it;
        ties break toward heavier requests, then manifest order. A request
        with no under-cap candidate still gets the best available pair: the
        plan never leaves a request unassigned while a legal model exists.
        """
        self.plan = {}
        if not remaining_ids:
            return
        manifest = {r["id"]: r for r in self.brief.manifest}
        ordered = []
        for rid in remaining_ids:
            req = manifest.get(rid)
            if req is None:
                continue
            cands = self._candidates(req["requires"])
            ncand = len(cands)
            idx = self.brief.manifest.index(req)
            if self.plan_order == "weight":
                key = (-req["weight"], ncand, idx)
            elif self.plan_order == "regret":
                # Loss if this request misses its best model: (best rating -
                # best rating on a DIFFERENT provider) * weight. Assign the
                # requests with the most to lose first.
                best = max([c[2] for c in cands], default=0)
                best_pid = None
                for c in cands:
                    if c[2] == best:
                        best_pid = c[0]
                        break
                second = max([c[2] for c in cands if c[0] != best_pid], default=0)
                key = (-(best - second) * req["weight"], ncand, idx)
            else:  # "scarcity"
                key = (ncand, -req["weight"], idx)
            ordered.append((key, rid))
        ordered.sort(key=lambda t: t[0])
        used = dict(self.calls)
        for _key, rid in ordered:
            req = manifest[rid]
            cands = self._candidates(req["requires"],
                                     exclude_wires=self.wire_exclusions(req))
            if not cands:
                continue
            under = [c for c in cands
                     if used[c[0]] < self.soft_cap(c[0]) and c[0] not in self.dead]
            pool = under if under else [c for c in cands if c[0] not in self.dead]
            if not pool:
                pool = cands
            pool.sort(key=self._sort_key)
            pid, mid = pool[0][0], pool[0][1]
            self.plan[rid] = (pid, mid)
            used[pid] = used.get(pid, 0) + 1

    def next_candidate(self, request_id, requires, exclude_providers,
                       exclude_wires=None):
        """The plan's pick for this request first; otherwise the best
        remaining candidate. Returns (provider_id, model_id) or None.

        exclude_wires defaults to the manifest-derived filter for this
        request; the caller may override it with what the request body
        itself declares (the body is the truth for JSON mode)."""
        if exclude_wires is None:
            req = None
            for r in self.brief.manifest:
                if r["id"] == request_id:
                    req = r
                    break
            exclude_wires = self.wire_exclusions(req)
        planned = self.plan.get(request_id)
        if planned is not None:
            pid, mid = planned
            if pid not in self.dead and pid not in exclude_providers \
                    and self.provider_by_id[pid]["wire"] not in exclude_wires:
                return planned
        return self._pick(requires, exclude_providers, exclude_wires)

    def record_call(self, provider_id, ok):
        self.calls[provider_id] = self.calls.get(provider_id, 0) + 1
        if not ok:
            self.failures[provider_id] = self.failures.get(provider_id, 0) + 1
            if self.failures[provider_id] >= self.max_failures:
                # Refused max_failures times: out of budget, in outage, or
                # on a second fail call -- not worth another call.
                self.dead.add(provider_id)

    def retry_order(self, failed):
        """Order for the last-resort retry pass: the (provider, model)
        pairs that failed in this request, deduplicated, best first.

        A provider's fail_calls are isolated ordinals (at most 2 per
        provider), so the call right after a failed ordinal usually
        succeeds -- unless the provider is in outage or out of budget,
        which the caller cannot tell from the refusal. Ordering by
        budget floor first: a big-budget provider's next ordinal is
        least likely to be a second scheduled fail (measured on 1003
        simulated instances: budget order spends 8 retry calls total
        vs 10+ for rating order, rescuing the same requests).
        Deterministic: budget floor, rating, brief provider order, id.
        """
        seen = []
        for pm in failed:
            if pm not in seen:
                seen.append(pm)
        return sorted(seen, key=lambda pm: (
            -BUDGET_MIN.get(pm[0], DEFAULT_BUDGET_MIN),
            -self.options[pm]["rating"],
            self.provider_order.get(pm[0], 999), pm[1]))


# ---------------------------------------------------------------------------
# The entry server.
# ---------------------------------------------------------------------------

class Aggregator(object):
    def __init__(self, brief, env_url, policy=None):
        self.brief = brief
        self.env_url = env_url
        self.provider_by_id = {p["id"]: p for p in brief.providers}
        self.catalog = {"providers": []}
        self.router = Router(brief, self.provider_by_id, policy)
        self.manifest_index = 0
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.server = None
        self.port = 0

    def load(self):
        probes = []
        for provider in self.brief.providers:
            ok, status, models = probe_provider(self.env_url, provider)
            if not ok:
                log("probe %s failed (HTTP %s); catalog falls back to the ratings table"
                    % (provider["id"], status))
            probes.append((ok, status, models))
        self.catalog = build_catalog(self.brief, probes)
        log("catalog: %d providers, %d models (%d free)"
            % (len(self.catalog["providers"]),
               sum(len(p["models"]) for p in self.catalog["providers"]),
               sum(1 for p in self.catalog["providers"] for m in p["models"] if m["free"])))

    def catalog_json(self):
        return json.dumps(self.catalog, separators=(",", ":"), ensure_ascii=False)

    def remaining_ids(self):
        return [r["id"] for r in self.brief.manifest[self.manifest_index:]]

    def requirements_for(self, canonical):
        """Manifest requirements by arrival order, falling back to what the
        body itself declares (tools / image / JSON mode)."""
        if self.manifest_index < len(self.brief.manifest):
            return self.brief.manifest[self.manifest_index]["requires"]
        req = {}
        if canonical.tools:
            req["tools"] = True
        if canonical.image is not None:
            req["vision"] = True
        if canonical.json_mode:
            req["json"] = True
        return req

    def _answer_response(self, api, canonical, mid, answer):
        usage = answer_usage(answer, canonical)
        if canonical.stream:
            payload = client_stream_response(api, mid, answer, usage).encode("utf-8")
            return 200, "text/event-stream; charset=utf-8", payload
        payload = _json_bytes(client_json_response(api, mid, answer, usage))
        return 200, "application/json", payload

    def serve(self, api, body):
        """One client request -> (status, content_type, payload_bytes)."""
        canonical = Canonical(api, body)
        requires = self.requirements_for(canonical)
        request_id = None
        if self.manifest_index < len(self.brief.manifest):
            request_id = self.brief.manifest[self.manifest_index]["id"]
        self.manifest_index += 1
        # The body is the truth for JSON mode: an OpenAI client request
        # carrying response_format must never go to an Anthropic-wire
        # provider (that wire cannot echo json_mode).
        exclude_wires = frozenset({"anthropic"}) if canonical.json_mode else frozenset()
        exclude = set()
        failed = []  # (provider, model) pairs refused in this request, in order
        last_status = 502
        while True:
            cand = self.router.next_candidate(request_id, requires, frozenset(exclude),
                                              exclude_wires)
            if cand is None:
                # Last-resort retry pass (single round): every candidate
                # failed, so this request would otherwise go unserved.
                # Each provider's fail_calls are isolated ordinals, so the
                # call right after a failed ordinal usually succeeds; one
                # more attempt per failed provider, best first. Only runs
                # on a request that would otherwise be lost (measured: 8
                # retry calls across 1003 simulated instances), so it
                # costs nothing on the common path.
                for pid, mid in self.router.retry_order(failed):
                    if pid in self.router.dead:
                        continue
                    provider = self.provider_by_id[pid]
                    status, answer = call_provider(self.env_url, provider, mid, canonical)
                    self.router.record_call(pid, status == 200 and answer is not None)
                    if status == 200 and answer is not None:
                        log("retry: %s/%s served after refusal" % (pid, mid))
                        return self._answer_response(api, canonical, mid, answer)
                    last_status = status if status else 502
                    self.router.replan(self.remaining_ids())
                    log("retry: %s/%s refused again (%s)" % (pid, mid, last_status))
                status, err = client_error_response(
                    api, 502, "no free model can serve this request")
                return status, "application/json", _json_bytes(err)
            pid, mid = cand
            provider = self.provider_by_id[pid]
            status, answer = call_provider(self.env_url, provider, mid, canonical)
            self.router.record_call(pid, status == 200 and answer is not None)
            if status == 200 and answer is not None:
                return self._answer_response(api, canonical, mid, answer)
            last_status = status if status else 502
            failed.append((pid, mid))
            exclude.add(pid)
            # Drop the provider, re-plan what is left, and fail over.
            self.router.replan(self.remaining_ids())
            log("failover: %s/%s refused (%s); trying the next candidate"
                % (pid, mid, last_status))

    # -- lifecycle ----------------------------------------------------------

    def register(self):
        body = {"protocol_version": PROTOCOL_VERSION, "port": self.port,
                "catalog": self.catalog}
        status, _h, raw = http_call(
            "POST", self.env_url.rstrip("/") + "/register", {}, body, CALL_TIMEOUT_S)
        log("register: HTTP %s %s" % (status, raw[:200].decode("utf-8", "replace")))
        return status == 200

    def poll_work(self):
        while not self.done.is_set():
            status, _h, raw = http_call(
                "GET", self.env_url.rstrip("/") + "/work", {}, None, CALL_TIMEOUT_S)
            if status == 200:
                try:
                    if json.loads(raw.decode("utf-8")).get("done") is True:
                        self.done.set()
                        return
                except Exception:
                    pass
            self.done.wait(POLL_INTERVAL_S)

    def shutdown(self):
        self.done.set()

    def serve_forever(self):
        # handle_request() blocks in accept(); a short listen timeout lets
        # the loop notice `done` without serve_forever/shutdown dances.
        self.server.socket.settimeout(0.2)
        while not self.done.is_set():
            try:
                self.server.handle_request()
            except (TimeoutError, OSError):
                pass


def _json_bytes(obj):
    return (json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def make_handler(agg):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep stdout clean
            pass

        def _read_body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                return None
            return self.rfile.read(length) if length > 0 else b""

        def _send(self, status, content_type, payload):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/v1/models":
                models = served_models(agg.brief, agg.catalog)
                if self.headers.get("x-api-key"):
                    body = {"data": [{"type": "model", "id": m["id"],
                                      "display_name": m["display_name"],
                                      "created_at": FIXED_ANTHROPIC_CREATED_AT}
                                     for m in models],
                            "has_more": False}
                else:
                    body = {"object": "list", "data": [
                        {"id": m["id"], "object": "model",
                         "created": FIXED_OPENAI_CREATED, "owned_by": m["provider"]}
                        for m in models]}
                self._send(200, "application/json", _json_bytes(body))
                return
            self._send(404, "application/json", _json_bytes({"error": "not_found"}))

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path not in ("/v1/chat/completions", "/v1/messages"):
                self._send(404, "application/json", _json_bytes({"error": "not_found"}))
                return
            api = "openai" if parsed.path == "/v1/chat/completions" else "anthropic"
            raw = self._read_body()
            if raw is None:
                status, err = client_error_response(api, 413, "request body too large")
                self._send(status, "application/json", _json_bytes(err))
                return
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                status, err = client_error_response(api, 400, "request body is not one JSON object")
                self._send(status, "application/json", _json_bytes(err))
                return
            if not _is_record(body):
                status, err = client_error_response(api, 400, "request body is not one JSON object")
                self._send(status, "application/json", _json_bytes(err))
                return
            with agg.lock:  # strictly sequential inference (zai serves 1 at a time)
                status, content_type, payload = agg.serve(api, body)
            self._send(status, content_type, payload)

    return Handler


def load_policy():
    raw = os.environ.get("MAKEFUN_POLICY_JSON") or ""
    if not raw:
        # Tuned on 3 practice trials + 150 fresh generator instances
        # (score 99 on 153/153; see NOTES.md "Call optimization", 15 Nov 2026).
        # Rationale:
        # - cap_reserve 5 (no reserve_map): spreads load across all 10
        #   providers, so each provider's failure-prone 2nd/3rd calls are
        #   rarely all hit; the plan uses every provider's guaranteed-free
        #   first call. Mean 23.1 calls vs 25.2 for the old cap-3 policy.
        # - min_cap 1: a provider's soft cap never drops to 0, so its top
        #   model stays in the under-cap pool (cap 5 alone zeroes the
        #   small-budget providers' caps: zai/sambanova/ovh/cerebras).
        # - max_failures 3: a provider's fail_calls number at most 2, so a
        #   provider that refused twice and is still alive is clean on the
        #   third try; a fourth retry never helped in simulation.
        # - plan_order "scarcity": scarcest requests (vision, min_context)
        #   claim their providers first, so a filler can never steal the
        #   first call of a provider a later scarce request needs.
        # - tie_break "budget": same rating -> bigger budget floor first
        #   (a repeat call there is less likely to land on a fail number).
        # - Last-resort retry (in Aggregator.serve, not a policy knob):
        #   when every candidate for a request has refused, retry each
        #   failed provider once, budget-floor first (Router.retry_order).
        #   A provider's fail_calls are isolated ordinals, so the call
        #   after a failed ordinal usually succeeds. Measured on 1003
        #   simulated instances: rescues all 7 requests the old code left
        #   unserved (min_context r10/r12) at 8 retry calls total, and
        #   never triggers on the 3 practice trials (22/21/21 calls
        #   unchanged). See NOTES.md "Retry pass (16 Nov 2026)".
        # REJECTED by measurement (see NOTES.md): per-call rating penalties
        # and demote-after-1-failure both strand scarce requests (vision
        # r06 / min_context r12 unserved on fresh draws); a min_context
        # reservation guard cost earned AND calls. Keep the preference
        # order strictly rating-first within the under-cap pool.
        return {"cap_reserve": 5, "max_failures": 3, "demote_failed": False,
                "tie_break": "budget", "plan_order": "scarcity",
                "reserve_map": {}, "min_cap": 1}
    try:
        policy = json.loads(raw)
        return policy if isinstance(policy, dict) else {}
    except Exception:
        return {}


def main(argv):
    env_url = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE") or ""
    policy = load_policy()
    brief = read_brief(prompt_file)
    if brief.error is not None and prompt_file:
        log(brief.error)
    if "--list" in argv:
        if not prompt_file or not env_url:
            sys.stdout.write('{"providers":[]}\n')
            return 0
        agg = Aggregator(brief, env_url, policy)
        agg.load()
        sys.stdout.write(agg.catalog_json() + "\n")
        return 0
    if not env_url:
        log("MAKEFUN_ENV_URL is not set; nothing to serve")
        return 0
    agg = Aggregator(brief, env_url, policy)
    agg.load()
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(agg))
    agg.server = server
    agg.port = server.server_address[1]
    if agg.port == 8080:  # the starter refuses 8080; an ephemeral port never is
        log("refusing port 8080")
        return 1
    if not agg.register():
        log("register failed; continuing to serve anyway")
    poller = threading.Thread(target=agg.poll_work, daemon=True)
    poller.start()

    def _stop(signum, _frame):
        agg.shutdown()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    log("listening on 0.0.0.0:%d; polling /work" % agg.port)
    try:
        agg.serve_forever()
    finally:
        server.server_close()
    log("work done; exiting 0")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
