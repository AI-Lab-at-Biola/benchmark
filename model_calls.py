"""Model calls for the math track: one run loop per provider, each returning the same run record.

run_model drives Claude through the Messages API with Anthropic's code execution tool.
run_openai drives OpenAI models through the Responses API with OpenAI's code interpreter.
"""

import asyncio
import threading
import time

import anthropic
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


def status(started, message, usage=None):
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.monotonic() - started))
    if usage:
        approx = "~" if usage.get("is_estimate") else ""
        message += f" · in {usage['input']:,} · out {approx}{usage['output']:,}"
        if usage.get("incomplete"):
            message += " (usage incomplete)"
    print(f"[{elapsed}] {message}", flush=True)


def heartbeat(started, activity, finished, current_usage, every_s=120):
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
                current_usage(),
            )


# --- Claude ----------------------------------------------------------------------------------


def run_model(config, prompt):
    """Synchronous entry point for the Claude benchmark."""
    return asyncio.run(run_model_async(config, prompt))


async def run_model_async(config, prompt):
    async with anthropic.AsyncAnthropic() as client:
        return await claude_run(client, config, prompt)


async def claude_run(client, config, prompt):
    """Drive the conversation until the model finishes or a budget runs out. Returns the run record.

    The server runs its code-execution loop inside one response, so like openai_run the wall
    clock is also enforced while streaming.
    """
    messages = [{"role": "user", "content": prompt}]
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "thinking": 0}
    responses = []  # one entry per API request, in order
    tool_calls = 0
    container_id = None
    stop = {"reason": None, "budget_exhausted": None, "error": None}
    served_model = None
    started = time.monotonic()
    activity = {"last": started}  # when the stream last delivered anything
    in_flight = {"input": 0, "chars": 0}  # the response currently streaming
    finished = threading.Event()
    threading.Thread(
        target=heartbeat,
        args=(started, activity, finished, lambda: live_usage(usage, in_flight)),
        daemon=True,
    ).start()
    status(started, f"run started · {config['model']} · effort {config['effort']}")
    deadline = started + config["wall_clock_min"] * 60

    while True:
        stop["budget_exhausted"] = budget_exhausted(
            config, started, tool_calls, usage["output"]
        )
        if stop["budget_exhausted"]:
            break

        remaining_tokens = config["max_output_tokens"] - usage["output"]
        request = dict(
            model=config["model"],
            max_tokens=min(MAX_TOKENS_PER_RESPONSE, remaining_tokens),
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": config["effort"]},
            tools=[CODE_EXECUTION_TOOL],
            messages=messages,
        )
        if container_id:
            request["container"] = container_id

        # Blocks are recorded as they finish, so a run cut off mid-response keeps its transcript.
        blocks = []
        response = None
        partial_response = None
        remaining = max(0, deadline - time.monotonic())
        # Only the deadline bounds the request; SDK retries would reuse a stale timeout budget.
        attempt_client = client.with_options(timeout=remaining, max_retries=0)
        try:
            # Socket timeouts restart on each read, so this also covers a stream that keeps
            # sending events past the deadline.
            async with asyncio.timeout(remaining):
                async with attempt_client.messages.stream(**request) as stream:
                    async for event in stream:
                        activity["last"] = time.monotonic()
                        track_in_flight(in_flight, event)
                        if event.type in {"message_start", "message_delta"}:
                            partial_response = stream.current_message_snapshot
                        if event.type == "content_block_stop":
                            blocks.append(block_record(event.content_block))
                            if event.content_block.type == "server_tool_use":
                                tool_calls += 1
                            note = block_status(event.content_block)
                            if note:
                                status(started, note, live_usage(usage, in_flight))
                    response = await stream.get_final_message()
        except TimeoutError:
            stop["budget_exhausted"] = "wall_clock"
        except anthropic.APIError as error:
            stop["error"] = f"{type(error).__name__}: {error}"
        finally:
            in_flight.update(input=0, chars=0)

        if response is None:
            # Observed counts are a lower bound without the final response usage.
            usage["incomplete"] = True
            if partial_response is not None:
                served_model = partial_response.model
                add_usage(usage, partial_response.usage)
            stop["reason"] = "cut off" if stop["budget_exhausted"] else "error"
            responses.append({"stop_reason": stop["reason"], "blocks": blocks})
            break

        served_model = response.model
        if response.container:
            container_id = response.container.id
        add_usage(usage, response.usage)
        responses.append(
            {
                "stop_reason": response.stop_reason,
                "blocks": [block_record(b) for b in response.content],
            }
        )
        messages.append({"role": "assistant", "content": response.content})
        stop["reason"] = response.stop_reason
        status(
            started,
            f"response {len(responses)} · {response.stop_reason} · {tool_calls} code runs",
            usage,
        )

        if response.stop_reason == "pause_turn":
            # The server's tool loop hit its iteration limit; resending the history resumes it.
            continue
        if response.stop_reason == "max_tokens":
            # Prefill isn't allowed, so a cut-off turn continues through a user message.
            messages.append(
                {
                    "role": "user",
                    "content": "You hit the per-response output limit. Continue where you left off.",
                }
            )
            continue
        break  # end_turn, refusal, or anything unexpected

    finished.set()
    status(
        started,
        f"run ended · {stop['budget_exhausted'] or stop['error'] or stop['reason']}",
        usage,
    )
    return {
        "served_model": served_model,
        "usage": usage,
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


def track_in_flight(in_flight, event):
    """Usage only arrives at message_start (input) and message_delta (end), and a whole
    server-side code-execution loop is one response, so count streamed characters meanwhile."""
    if event.type == "message_start":
        in_flight["input"] = event.message.usage.input_tokens or 0
    elif event.type == "content_block_delta":
        delta = event.delta
        text = getattr(delta, "text", None) or getattr(delta, "thinking", None)
        text = text or getattr(delta, "partial_json", None) or ""
        in_flight["chars"] += len(text)


def live_usage(totals, in_flight):
    """Run totals plus the streaming response: exact input, output estimated at ~4 chars/token."""
    if not in_flight["input"] and not in_flight["chars"]:
        return totals
    return {
        **totals,
        "input": totals["input"] + in_flight["input"],
        "output": totals["output"] + in_flight["chars"] // 4,
        "is_estimate": True,
    }


def add_usage(totals, usage):
    totals["input"] += usage.input_tokens or 0
    totals["output"] += usage.output_tokens or 0
    totals["cache_read"] += usage.cache_read_input_tokens or 0
    totals["cache_write"] += usage.cache_creation_input_tokens or 0
    details = getattr(usage, "output_tokens_details", None)
    totals["thinking"] += getattr(details, "thinking_tokens", 0) or 0


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
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "thinking": 0}
    responses = []
    tool_calls = 0
    stop = {"reason": None, "budget_exhausted": None, "error": None}
    served_model = None
    started = time.monotonic()
    activity = {"last": started}
    finished = threading.Event()
    threading.Thread(
        target=heartbeat,
        args=(started, activity, finished, lambda: usage),
        daemon=True,
    ).start()
    status(started, f"run started · {config['model']} · effort {config['effort']}")

    try:
        while True:
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
            events = openai_events(client, request, started, deadline)
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
                                status(started, note, usage)
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

            if response is None:
                if not stop["error"] and not stop["budget_exhausted"]:
                    stop["budget_exhausted"] = (
                        "wall_clock"  # openai_events ran out of time reconnecting
                    )
                stop["reason"] = "cut off" if stop["budget_exhausted"] else "error"
                responses.append({"stop_reason": stop["reason"], "blocks": blocks})
                break

            served_model = response.model
            add_openai_usage(usage, response.usage)
            reason = response.status
            if response.incomplete_details and response.incomplete_details.reason:
                reason = f"incomplete: {response.incomplete_details.reason}"
            if response.error:
                stop["error"] = f"{response.error.code}: {response.error.message}"
            responses.append({"stop_reason": reason, "blocks": blocks})
            stop["reason"] = reason
            status(
                started,
                f"response {len(responses)} · {reason} · {tool_calls} code runs",
                usage,
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
        usage,
    )
    return {
        "served_model": served_model,
        "usage": usage,
        "tool_calls": tool_calls,
        "wall_clock_min": round((time.monotonic() - started) / 60, 2),
        "stop": stop,
        "responses": responses,
    }


class StreamLost(Exception):
    """The stream kept dropping and every reconnect attempt failed."""


async def openai_events(client, request, started, deadline):
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
        except openai.APIConnectionError as error:
            dropped = f"{type(error).__name__}: {error}"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        if response_id is None:
            raise StreamLost(f"{dropped} (before a response id arrived)")
        if attempt == STREAM_RESUMES:
            break
        status(started, f"{dropped} · reconnecting after event {last_seq}")
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


def add_openai_usage(totals, usage):
    """Like add_usage. OpenAI's input count already includes cached tokens; Anthropic's doesn't."""
    if usage is None:
        return
    totals["input"] += usage.input_tokens or 0
    totals["output"] += usage.output_tokens or 0
    totals["cache_read"] += getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
    totals["cache_write"] += (
        getattr(usage.input_tokens_details, "cache_write_tokens", 0) or 0
    )
    totals["thinking"] += (
        getattr(usage.output_tokens_details, "reasoning_tokens", 0) or 0
    )


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
