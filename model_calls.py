"""Model calls for the math track: one run loop per provider, each returning the same run record.

run_model drives Claude through the Messages API with Anthropic's code execution tool.
run_openai drives OpenAI models through the Responses API with OpenAI's code interpreter.
"""

import asyncio
import threading
import time

import anthropic
import httpx2
import openai

CODE_EXECUTION_TOOL = {"type": "code_execution_20260521", "name": "code_execution"}
# OpenAI's sandbox, with outbound network off to match the prompt's "no internet".
OPENAI_CODE_TOOL = {
    "type": "code_interpreter",
    "container": {"type": "auto", "network_policy": {"type": "disabled"}},
}
API_KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
MAX_TOKENS_PER_RESPONSE = 64000  # same per-response cap for both providers
STREAM_RESUMES = 5  # reconnects to a dropped OpenAI stream before the run gives up
# SDK-typed terminal events; incomplete and failed carry their output too.
OPENAI_TERMINAL_EVENTS = {
    "response.completed",
    "response.incomplete",
    "response.failed",
}


# --- Shared ----------------------------------------------------------------------------------


def budget_exhausted(config, started, tool_calls, output_tokens):
    """Which budget has run out ('wall_clock', 'tool_calls', 'output_tokens'), or None."""
    if (time.monotonic() - started) / 60 >= config["wall_clock_min"]:
        return "wall_clock"
    if tool_calls >= config["max_tool_calls"]:
        return "tool_calls"
    if config["max_output_tokens"] - output_tokens < 1024:
        return "output_tokens"
    return None


def usage_status(usage, provider, complete=True):
    """Provider-reported counts, not token estimates or a final billing total.

    Keep raw provider semantics in JSON. Anthropic's input excludes cache reads
    and writes; OpenAI's input includes them already. Thinking is part of output.
    """
    input_tokens = usage["input"]
    if provider == "anthropic":
        input_tokens += usage["cache_read"] + usage["cache_write"]
    suffix = "" if complete else " (incomplete; latest reported counts)"
    return (
        f"reported tokens: input {input_tokens:,} · output {usage['output']:,}"
        f" · cache read {usage['cache_read']:,} · cache write {usage['cache_write']:,}"
        f"{suffix}"
    )


class UsageTracker:
    """Sum requests, replacing cumulative stream snapshots instead of adding them."""

    def __init__(self, provider):
        self.provider = provider
        self.totals = self.empty()
        self.current = self.empty()
        self.in_flight = False
        self.complete = True
        self.lock = threading.Lock()

    @staticmethod
    def empty():
        return dict(input=0, output=0, cache_read=0, cache_write=0, thinking=0)

    def begin(self):
        with self.lock:
            self.current = self.empty()
            self.in_flight = True

    def observe(self, usage):
        if usage is None:
            return
        fields = {
            "input": getattr(usage, "input_tokens", None),
            "output": getattr(usage, "output_tokens", None),
        }
        if self.provider == "anthropic":
            fields.update(
                cache_read=getattr(usage, "cache_read_input_tokens", None),
                cache_write=getattr(usage, "cache_creation_input_tokens", None),
                thinking=getattr(
                    getattr(usage, "output_tokens_details", None),
                    "thinking_tokens",
                    None,
                ),
            )
        else:
            details = getattr(usage, "input_tokens_details", None)
            fields.update(
                cache_read=getattr(details, "cached_tokens", None),
                cache_write=getattr(details, "cache_write_tokens", None),
                thinking=getattr(
                    getattr(usage, "output_tokens_details", None),
                    "reasoning_tokens",
                    None,
                ),
            )
        with self.lock:
            for field, value in fields.items():
                if value is not None:
                    self.current[field] = value

    def finish(self, usage=None):
        self.observe(usage)
        complete = (
            usage is not None
            and getattr(usage, "input_tokens", None) is not None
            and getattr(usage, "output_tokens", None) is not None
        )
        with self.lock:
            snapshot = self.current.copy()
            for key, value in snapshot.items():
                self.totals[key] += value
            self.current = self.empty()
            self.complete = self.complete and complete
            self.in_flight = False
        return {"usage": snapshot, "usage_complete": complete}

    def snapshot(self):
        with self.lock:
            return {
                key: value + self.current[key] for key, value in self.totals.items()
            }

    def summary(self):
        with self.lock:
            usage = {
                key: value + self.current[key] for key, value in self.totals.items()
            }
            complete = self.complete and not self.in_flight
        return usage_status(usage, self.provider, complete)


def status(started, message, tracker=None):
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.monotonic() - started))
    if tracker is not None:
        message += f" · {tracker.summary()}"
    print(f"[{elapsed}] {message}", flush=True)


def heartbeat(started, activity, finished, every_s=120):
    """Every couple of quiet minutes, say how long it's been since the stream sent anything.

    Long code runs in the sandbox are silent, so a quiet stretch alone isn't a stall;
    a number that keeps climbing past ~15 min is.
    """
    while not finished.wait(every_s):
        quiet_min = (time.monotonic() - activity["last"]) / 60
        if quiet_min >= every_s / 60:
            status(
                started,
                f"still waiting · {quiet_min:.0f} min since last update",
                activity.get("tracker"),
            )


# --- Claude ----------------------------------------------------------------------------------


def run_model(config, prompt):
    """Synchronous entry point for the Anthropic benchmark."""
    return asyncio.run(run_anthropic_async(config, prompt))


async def run_anthropic_async(config, prompt):
    async with anthropic.AsyncAnthropic() as client:
        return await anthropic_run(client, config, prompt)


async def anthropic_run(client, config, prompt):
    """Stop the client stream at the deadline, including during silent tool runs.

    Closing the stream is not an acknowledgement of server-side cancellation or
    final billing. Preserve reported usage and mark interrupted requests incomplete.
    """
    messages = [{"role": "user", "content": prompt}]
    tracker = UsageTracker("anthropic")
    responses = []
    tool_calls = 0
    container_id = None
    stop = {"reason": None, "budget_exhausted": None, "error": None}
    served_model = None
    started = time.monotonic()
    deadline = started + config["wall_clock_min"] * 60
    activity = {"last": started, "tracker": tracker}
    finished = threading.Event()
    threading.Thread(
        target=heartbeat, args=(started, activity, finished), daemon=True
    ).start()
    status(
        started, f"run started · {config['model']} · effort {config['effort']}", tracker
    )

    try:
        while True:
            usage = tracker.snapshot()
            stop["budget_exhausted"] = budget_exhausted(
                config, started, tool_calls, usage["output"]
            )
            if stop["budget_exhausted"]:
                break

            request = dict(
                model=config["model"],
                max_tokens=min(
                    MAX_TOKENS_PER_RESPONSE,
                    config["max_output_tokens"] - usage["output"],
                ),
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": config["effort"]},
                tools=[CODE_EXECUTION_TOOL],
                messages=messages,
            )
            if container_id:
                request["container"] = container_id

            blocks = []
            response = None
            message_stopped = False
            tracker.begin()
            try:
                remaining = max(0, deadline - time.monotonic())
                # Disable implicit SDK retries: they could duplicate work/spend and
                # reuse a timeout after the run's deadline has passed.
                async with asyncio.timeout(remaining):
                    attempt_client = client.with_options(
                        timeout=remaining, max_retries=0
                    )
                    async with attempt_client.messages.stream(**request) as stream:
                        async for event in stream:
                            activity["last"] = time.monotonic()
                            if event.type == "message_start":
                                served_model = event.message.model
                                tracker.observe(event.message.usage)
                            elif event.type == "message_delta":
                                tracker.observe(event.usage)
                            elif event.type == "message_stop":
                                message_stopped = True
                            elif event.type == "content_block_stop":
                                block = event.content_block
                                blocks.append(block_record(block))
                                if block.type == "server_tool_use":
                                    tool_calls += 1
                                note = block_status(block)
                                if note:
                                    status(started, note, tracker)
                                if tool_calls >= config["max_tool_calls"]:
                                    stop["budget_exhausted"] = "tool_calls"
                                    break
                            if time.monotonic() >= deadline:
                                stop["budget_exhausted"] = "wall_clock"
                                break
                        if message_stopped and not stop["budget_exhausted"]:
                            response = await stream.get_final_message()
                        elif not stop["budget_exhausted"]:
                            stop["error"] = (
                                "StreamLost: stream closed before message_stop"
                            )
            except (TimeoutError, anthropic.APITimeoutError):
                if time.monotonic() >= deadline:
                    stop["budget_exhausted"] = "wall_clock"
                else:
                    stop["error"] = (
                        "APITimeoutError: stream timed out before the run deadline"
                    )
            except anthropic.APIError as error:
                stop["error"] = f"{type(error).__name__}: {error}"
            except httpx2.TransportError as error:
                # The SDK wraps request errors, but streaming body errors can
                # escape directly from the HTTP transport after partial output.
                if time.monotonic() >= deadline:
                    stop["budget_exhausted"] = "wall_clock"
                else:
                    stop["error"] = f"{type(error).__name__}: {error}"

            response_usage = tracker.finish(
                response.usage if response is not None else None
            )
            if response is None:
                stop["reason"] = "cut off" if stop["budget_exhausted"] else "error"
                responses.append(
                    {"stop_reason": stop["reason"], "blocks": blocks, **response_usage}
                )
                break

            served_model = response.model
            if response.container:
                container_id = response.container.id
            responses.append(
                {
                    "stop_reason": response.stop_reason,
                    "blocks": [block_record(b) for b in response.content],
                    **response_usage,
                }
            )
            messages.append({"role": "assistant", "content": response.content})
            stop["reason"] = response.stop_reason
            status(
                started,
                f"response {len(responses)} · {response.stop_reason} · {tool_calls} code runs",
                tracker,
            )
            if response.stop_reason == "pause_turn":
                continue
            if response.stop_reason == "max_tokens":
                messages.append(
                    {
                        "role": "user",
                        "content": "You hit the per-response output limit. Continue concisely and report your best verified answer.",
                    }
                )
                continue
            break
    finally:
        finished.set()
    status(
        started,
        f"run ended · {stop['budget_exhausted'] or stop['error'] or stop['reason']}",
        tracker,
    )
    return {
        "served_model": served_model,
        "usage": tracker.snapshot(),
        "usage_complete": tracker.complete,
        "tool_calls": tool_calls,
        "wall_clock_min": round((time.monotonic() - started) / 60, 2),
        "stop": stop,
        "responses": responses,
    }


def block_status(block):
    """A short progress note for a finished response block, or None."""
    if block.type == "thinking":
        return "thinking done"
    if block.type == "text":
        first_line = block.text.strip().splitlines()[0] if block.text.strip() else ""
        return f"model: {first_line[:100]}"
    if block.type == "server_tool_use":
        if block.name == "bash_code_execution":
            command = str(block.input.get("command", "")).strip().splitlines()
            return f"running: {command[0][:100] if command else ''}"
        return f"editing file: {block.input.get('path', '')}"
    if block.type == "bash_code_execution_tool_result":
        code = getattr(block.content, "return_code", None)
        return (
            f"code finished (exit {code})" if code is not None else "code failed to run"
        )
    return None


def block_record(block):
    """Reduce an API content block to the plain dict we log and render."""
    if block.type == "thinking":
        return {"type": "thinking", "text": block.thinking}
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "server_tool_use":
        return {"type": "tool_call", "tool": block.name, "input": block.input}
    if block.type == "bash_code_execution_tool_result":
        result = block.content
        if result.type == "bash_code_execution_result":
            return {
                "type": "tool_output",
                "return_code": result.return_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        return {
            "type": "tool_output",
            "error": getattr(result, "error_code", str(result)),
        }
    if block.type == "text_editor_code_execution_tool_result":
        return {"type": "tool_output", "editor": block.content.to_dict()}
    return {"type": block.type, "raw": block.to_dict()}


# --- OpenAI ----------------------------------------------------------------------------------


def run_openai(config, prompt):
    """Synchronous entry point for the OpenAI benchmark."""
    return asyncio.run(run_openai_async(config, prompt))


async def run_openai_async(config, prompt):
    async with openai.AsyncOpenAI() as client:
        return await openai_run(client, config, prompt)


async def openai_run(client, config, prompt):
    """run_model's counterpart for OpenAI: the Responses API with its code interpreter.

    The server runs the whole tool loop inside one response (there is no pause_turn), so the
    wall clock is also checked while streaming, and max_tool_calls is enforced by the API.
    """
    next_input = prompt
    previous_id = (
        None  # the server keeps the conversation; each request points at the last one
    )
    tracker = UsageTracker("openai")
    responses = []
    tool_calls = 0
    stop = {"reason": None, "budget_exhausted": None, "error": None}
    served_model = None
    started = time.monotonic()
    activity = {"last": started, "tracker": tracker}
    finished = threading.Event()
    threading.Thread(
        target=heartbeat, args=(started, activity, finished), daemon=True
    ).start()
    status(
        started, f"run started · {config['model']} · effort {config['effort']}", tracker
    )

    try:
        while True:
            usage = tracker.snapshot()
            stop["budget_exhausted"] = budget_exhausted(
                config, started, tool_calls, usage["output"]
            )
            if stop["budget_exhausted"]:
                break

            remaining_tokens = config["max_output_tokens"] - usage["output"]
            request = dict(
                model=config["model"],
                input=next_input,
                reasoning={"effort": config["effort"], "summary": "auto"},
                tools=[OPENAI_CODE_TOOL],
                include=["code_interpreter_call.outputs"],
                max_output_tokens=min(MAX_TOKENS_PER_RESPONSE, remaining_tokens),
                max_tool_calls=config["max_tool_calls"] - tool_calls,
            )
            if previous_id:
                request["previous_response_id"] = previous_id

            # Items are recorded as they finish, so a run cut off mid-response keeps its transcript.
            blocks = []
            response = None
            response_id = None
            deadline = started + config["wall_clock_min"] * 60
            events = openai_events(client, request, started, deadline, tracker)
            tracker.begin()
            try:
                # Socket timeouts restart on each read. This deadline also interrupts
                # a silent read or transport keepalives that never yield model events.
                async with asyncio.timeout(max(0, deadline - time.monotonic())):
                    async for event in events:
                        activity["last"] = time.monotonic()
                        if event.type == "response.created":
                            response_id = event.response.id
                        if event.type == "response.output_item.done":
                            blocks += openai_item_records(event.item)
                            if event.item.type == "code_interpreter_call":
                                tool_calls += 1
                            note = openai_item_status(event.item)
                            if note:
                                status(started, note, tracker)
                        if event.type in OPENAI_TERMINAL_EVENTS:
                            response = event.response
                        if time.monotonic() >= deadline:
                            stop["budget_exhausted"] = "wall_clock"
                            break
            except TimeoutError:
                stop["budget_exhausted"] = "wall_clock"
            except (openai.APIError, StreamLost) as error:
                stop["error"] = f"{type(error).__name__}: {error}"
            finally:
                await events.aclose()
                # A background response outlives the connection, so stop it rather than leave it billing.
                if response is None and response_id:
                    await cancel_openai_response(client, response_id)

            response_usage = tracker.finish(
                response.usage if response is not None else None
            )
            if response is None:
                if not stop["error"] and not stop["budget_exhausted"]:
                    stop["budget_exhausted"] = (
                        "wall_clock"  # openai_events ran out of time reconnecting
                    )
                stop["reason"] = "cut off" if stop["budget_exhausted"] else "error"
                responses.append(
                    {"stop_reason": stop["reason"], "blocks": blocks, **response_usage}
                )
                break

            served_model = response.model
            reason = response.status
            if response.incomplete_details and response.incomplete_details.reason:
                reason = f"incomplete: {response.incomplete_details.reason}"
            if response.error:
                stop["error"] = f"{response.error.code}: {response.error.message}"
            responses.append(
                {"stop_reason": reason, "blocks": blocks, **response_usage}
            )
            stop["reason"] = reason
            status(
                started,
                f"response {len(responses)} · {reason} · {tool_calls} code runs",
                tracker,
            )

            if reason == "incomplete: max_output_tokens":
                previous_id = response.id
                next_input = "You hit the per-response output limit. Continue where you left off."
                continue
            break  # completed, failed, content filter, or anything unexpected

    finally:
        finished.set()
    status(
        started,
        f"run ended · {stop['budget_exhausted'] or stop['error'] or stop['reason']}",
        tracker,
    )
    return {
        "served_model": served_model,
        "usage": tracker.snapshot(),
        "usage_complete": tracker.complete,
        "tool_calls": tool_calls,
        "wall_clock_min": round((time.monotonic() - started) / 60, 2),
        "stop": stop,
        "responses": responses,
    }


class StreamLost(Exception):
    """The stream kept dropping and every reconnect attempt failed."""


async def openai_events(client, request, started, deadline, tracker=None):
    """Yield every event of one background response, reconnecting if the stream drops.

    A run is one response that streams for half an hour or more, mostly silent while code
    runs, and that connection can close before the terminal event arrives. Background mode
    keeps the response alive server-side, so the stream is reopened by id and the SDK skips
    events at or before the last sequence number already seen. Reconnecting stops at
    `deadline` (a time.monotonic() value), so backoff never outlasts the wall-clock budget.
    """
    response_id = None
    last_seq = None
    for attempt in range(STREAM_RESUMES + 1):
        # Recheck after backoff too: sleeping can consume the entire remaining budget.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        # Apply the remaining timeout to both creation and resumption requests.
        # Only this loop retries; SDK retries would reuse a stale timeout budget.
        attempt_client = client.with_options(timeout=remaining, max_retries=0)
        if response_id is None:
            manager = attempt_client.responses.stream(**request, background=True)
        else:
            # Resume by id without repeating the response creation arguments.
            manager = attempt_client.responses.stream(
                response_id=response_id,
                starting_after=last_seq,
            )
        try:
            async with manager as stream:
                async for event in stream:
                    last_seq = event.sequence_number
                    if event.type == "response.created":
                        response_id = event.response.id
                    yield event
                    if event.type in OPENAI_TERMINAL_EVENTS:
                        return
            dropped = "stream closed before the response finished"
        except (openai.APIConnectionError, httpx2.TransportError) as error:
            dropped = f"{type(error).__name__}: {error}"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        if response_id is None:
            raise StreamLost(f"{dropped} (before a response id arrived)")
        if attempt == STREAM_RESUMES:
            break
        status(started, f"{dropped} · reconnecting after event {last_seq}", tracker)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        await asyncio.sleep(min(2**attempt, 30, remaining))
    raise StreamLost(f"gave up after {STREAM_RESUMES} reconnects to {response_id}")


async def cancel_openai_response(client, response_id):
    try:
        # Cancellation needs its own short allowance once the run budget is exhausted.
        async with asyncio.timeout(10):
            await client.with_options(timeout=10, max_retries=0).responses.cancel(
                response_id
            )
    except (openai.APIError, TimeoutError):
        pass  # already finished, or the cancel itself failed; nothing more to do here


def openai_item_records(item):
    """Reduce a Responses API output item to the same plain dicts block_record produces."""
    if item.type == "reasoning":
        text = "\n\n".join(part.text for part in item.summary)
        return [{"type": "thinking", "text": text}]
    if item.type == "message":
        return [
            {
                "type": "text",
                "text": part.text
                if part.type == "output_text"
                else f"[refusal] {part.refusal}",
            }
            for part in item.content
        ]
    if item.type == "code_interpreter_call":
        outputs = item.outputs or []
        logs = "".join(o.logs for o in outputs if o.type == "logs")
        images = [f"[image: {o.url}]" for o in outputs if o.type == "image"]
        return [
            {
                "type": "tool_call",
                "tool": "code_interpreter",
                "input": {"code": item.code},
            },
            {
                "type": "tool_output",
                "return_code": None,  # the code interpreter reports a status, not an exit code
                "status": item.status,
                "stdout": "\n".join([logs, *images]).strip(),
                "stderr": "",
            },
        ]
    return [{"type": item.type, "raw": item.to_dict()}]


def openai_item_status(item):
    """A short progress note for a finished output item, or None."""
    if item.type == "reasoning":
        return "thinking done"
    if item.type == "message":
        text = "".join(getattr(part, "text", "") for part in item.content).strip()
        return f"model: {text.splitlines()[0][:100] if text else ''}"
    if item.type == "code_interpreter_call":
        code = (item.code or "").strip().splitlines()
        return f"ran code ({item.status}): {code[0][:100] if code else ''}"
    return None
