import asyncio
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import anthropic
import httpx2
import openai

import model_calls


def response_event(event_type, sequence):
    return {
        "type": event_type,
        "sequence_number": sequence,
        "response": {
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "model": "test-model",
            "status": (
                "completed" if event_type == "response.completed" else "in_progress"
            ),
            "output": [],
            "error": None,
            "incomplete_details": None,
            "usage": None,
        },
    }


class EventStream(httpx2.AsyncByteStream):
    def __init__(self, events):
        self.events = events

    async def __aiter__(self):
        for event in self.events:
            yield f"data: {json.dumps(event)}\n\n".encode()


class OpenAIDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = 0.0
        self.requests = []
        self.streams = []
        self.actions = []
        self.client = openai.AsyncOpenAI(
            api_key="test-key",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.handle)),
        )
        self.addAsyncCleanup(self.client.close)
        for target, options in (
            ("model_calls.time", {"wraps": time}),
            ("model_calls.asyncio.sleep", {"side_effect": self.advance}),
            ("model_calls.status", {}),
        ):
            patcher = patch(target, **options)
            mocked = patcher.start()
            self.addCleanup(patcher.stop)
            if target == "model_calls.time":
                mocked.monotonic.side_effect = lambda: self.now
            if target.endswith("sleep"):
                self.sleep = mocked

    def advance(self, seconds):
        self.now += seconds

    async def handle(self, request):
        self.requests.append(request)
        action = self.actions.pop(0)
        if callable(action):
            response = await action(request)
        else:
            response = httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=EventStream(action),
            )
        self.streams.append(response)
        return response

    def events(self, deadline):
        return model_calls.openai_events(
            self.client, {"model": "test-model", "input": "test"}, 0, deadline
        )

    async def cancelled_response(self, request):
        return httpx2.Response(200, json={"id": "resp_test"})

    async def timeout(self, request):
        seconds = request.extensions["timeout"]["read"]
        self.advance(seconds)
        raise httpx2.ReadTimeout("timed out", request=request)

    async def body_timeout(self, request):
        advance = self.advance

        class TimeoutStream(httpx2.AsyncByteStream):
            async def __aiter__(self):
                event = response_event("response.created", 0)
                yield f"data: {json.dumps(event)}\n\n".encode()
                advance(request.extensions["timeout"]["read"])
                raise httpx2.ReadTimeout("stream timed out", request=request)

        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=TimeoutStream(),
        )

    async def run_model(self, deadline):
        finished = Mock()
        with patch("model_calls.threading") as threads:
            threads.Event.return_value = finished
            result = await model_calls.openai_run(
                self.client,
                {
                    "model": "test-model",
                    "effort": "high",
                    "wall_clock_min": deadline / 60,
                    "max_tool_calls": 10,
                    "max_output_tokens": 4096,
                },
                "test",
            )
        finished.set.assert_called_once_with()
        self.assertTrue(all(stream.is_closed for stream in self.streams))
        return result

    async def test_expired_deadline_does_not_create_response(self):
        for deadline in (0, -1):
            self.assertEqual([e async for e in self.events(deadline)], [])
        self.assertEqual(self.requests, [])

    async def test_backoff_reaching_or_overshooting_deadline_does_not_reconnect(self):
        for overshoot in (0, 0.1):
            with self.subTest(overshoot=overshoot):
                self.now = 0
                self.requests.clear()
                self.actions = [[response_event("response.created", 0)]]
                self.sleep.side_effect = lambda seconds: self.advance(
                    seconds + overshoot
                )
                self.assertEqual(len([e async for e in self.events(1)]), 1)
                self.assertEqual(len(self.requests), 1)
                self.assertTrue(self.streams[-1].is_closed)

    async def test_reconnect_uses_remaining_timeout_and_skips_seen_events(self):
        created = response_event("response.created", 0)
        self.actions = [[created], [created, response_event("response.completed", 1)]]
        events = [e async for e in self.events(3)]
        self.assertEqual([event.sequence_number for event in events], [0, 1])
        self.assertEqual([request.method for request in self.requests], ["POST", "GET"])
        self.assertEqual(self.requests[1].url.path, "/v1/responses/resp_test")
        for request, remaining in zip(self.requests, (3, 2)):
            self.assertTrue(
                all(
                    value == remaining
                    for value in request.extensions["timeout"].values()
                )
            )
        self.assertTrue(all(stream.is_closed for stream in self.streams))

    async def test_deadline_during_reconnect_timeout_cancels_and_reports_wall_clock(
        self,
    ):
        self.actions = [
            [response_event("response.created", 0)],
            self.timeout,
            self.cancelled_response,
        ]
        result = await self.run_model(3)
        self.assertEqual(
            result["stop"],
            {"reason": "cut off", "budget_exhausted": "wall_clock", "error": None},
        )
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(self.requests[-1].url.path, "/v1/responses/resp_test/cancel")
        self.assertEqual(self.requests[-1].extensions["timeout"]["read"], 10)
        self.sleep.assert_called_once_with(1)

    async def test_backoff_deadline_cancels_even_if_cancellation_times_out(self):
        self.actions = [[response_event("response.created", 0)], self.timeout]
        result = await self.run_model(1)
        self.assertEqual(result["stop"]["budget_exhausted"], "wall_clock")
        self.assertIsNone(result["stop"]["error"])
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[-1].url.path, "/v1/responses/resp_test/cancel")

    async def test_initial_request_timeout_at_deadline_reports_wall_clock(self):
        self.actions = [self.timeout]
        result = await self.run_model(1)
        self.assertEqual(result["stop"]["budget_exhausted"], "wall_clock")
        self.assertIsNone(result["stop"]["error"])
        self.assertEqual(len(self.requests), 1)
        self.sleep.assert_not_called()

    async def test_timeout_reading_stream_body_closes_and_cancels(self):
        for reconnect in (False, True):
            with self.subTest(reconnect=reconnect):
                self.now = 0
                self.requests.clear()
                self.actions = (
                    [[response_event("response.created", 0)]] if reconnect else []
                ) + [
                    self.body_timeout,
                    self.cancelled_response,
                ]
                result = await self.run_model(3)
                self.assertEqual(result["stop"]["budget_exhausted"], "wall_clock")
                self.assertIsNone(result["stop"]["error"])
                self.assertEqual(len(self.requests), 3 if reconnect else 2)
                self.assertEqual(
                    self.requests[-1].url.path, "/v1/responses/resp_test/cancel"
                )

    async def test_timeout_before_deadline_can_reconnect(self):
        async def early_timeout(request):
            self.advance(0.25)
            raise httpx2.ReadTimeout("timed out", request=request)

        self.actions = [
            [response_event("response.created", 0)],
            early_timeout,
            [
                response_event("response.created", 0),
                response_event("response.completed", 1),
            ],
        ]
        self.assertEqual(len([e async for e in self.events(10)]), 2)
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(self.requests[-1].extensions["timeout"]["read"], 6.75)

    async def test_raw_transport_error_reconnects_same_response(self):
        class BrokenStream(EventStream):
            async def __aiter__(self):
                async for chunk in super().__aiter__():
                    yield chunk
                raise httpx2.ReadError("connection reset")

        async def broken(request):
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=BrokenStream([response_event("response.created", 0)]),
            )

        self.actions = [
            broken,
            [
                response_event("response.created", 0),
                response_event("response.completed", 1),
            ],
        ]
        result = await self.run_model(10)
        self.assertEqual(result["stop"]["reason"], "completed")
        self.assertEqual([request.method for request in self.requests], ["POST", "GET"])
        self.assertEqual(self.requests[1].url.path, "/v1/responses/resp_test")

    async def test_retry_limit_reports_stream_lost_without_final_sleep(self):
        self.actions = [[response_event("response.created", 0)], [], []]
        with patch("model_calls.STREAM_RESUMES", 2):
            with self.assertRaises(model_calls.StreamLost):
                [e async for e in self.events(100)]
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(self.sleep.call_count, 2)
        self.assertEqual(self.now, 3)

    async def test_completed_response_does_not_cancel(self):
        self.actions = [
            [
                response_event("response.created", 0),
                response_event("response.completed", 1),
            ]
        ]
        result = await self.run_model(3)
        self.assertEqual(result["stop"]["reason"], "completed")
        self.assertEqual(len(self.requests), 1)

    async def test_closing_events_closes_active_stream_without_reconnecting(self):
        self.actions = [[response_event("response.created", 0)]]
        events = self.events(3)
        await anext(events)
        self.assertFalse(self.streams[0].is_closed)
        await events.aclose()
        self.assertTrue(self.streams[0].is_closed)
        self.assertEqual(len(self.requests), 1)
        self.sleep.assert_not_called()


class OpenAIBlockedStreamTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Warm the SDK's lazy event decoder outside the sub-second deadline.
        # A cold schema build can otherwise consume the entire test budget before
        # the checkpoint arrives (also reproducible on the unmodified runner).
        async def handle(request):
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=EventStream(
                    [
                        response_event("response.created", 0),
                        response_event("response.completed", 1),
                    ]
                ),
            )

        async with openai.AsyncOpenAI(
            api_key="test-key",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)),
        ) as client:
            now = time.monotonic()
            async for _ in model_calls.openai_events(
                client, {"model": "test-model", "input": "test"}, now, now + 10
            ):
                pass

    async def run_stream(self, mode):
        requests = []
        streams = []
        closed = asyncio.Event()

        class SlowStream(httpx2.AsyncByteStream):
            async def __aiter__(self):
                events = [
                    response_event("response.created", 0),
                    {
                        "type": "response.output_item.done",
                        "sequence_number": 1,
                        "output_index": 0,
                        "item": {
                            "id": "msg_test",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "checkpoint",
                                    "annotations": [],
                                }
                            ],
                        },
                    },
                ]
                for event in events:
                    yield f"data: {json.dumps(event)}\n\n".encode()
                if mode == "backoff":
                    return
                await asyncio.sleep(0.05)
                yield f"data: {json.dumps(response_event('response.in_progress', 2))}\n\n".encode()
                if mode == "keepalives":
                    while True:
                        await asyncio.sleep(0.02)
                        yield b": ping\n\n"
                else:
                    await asyncio.sleep(2)

            async def aclose(self):
                closed.set()

        async def handle(request):
            requests.append(request)
            if request.url.path.endswith("/cancel"):
                return httpx2.Response(200, json={"id": "resp_test"})
            if mode == "headers":
                await asyncio.sleep(2)
            response = httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, stream=SlowStream()
            )
            streams.append(response)
            return response

        async with openai.AsyncOpenAI(
            api_key="test-key",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)),
        ) as client:
            started = time.monotonic()
            with patch("model_calls.threading") as threads, patch("model_calls.status"):
                result = await asyncio.wait_for(
                    model_calls.openai_run(
                        client,
                        {
                            "model": "test-model",
                            "effort": "high",
                            "wall_clock_min": 0.3 / 60,
                            "max_tool_calls": 10,
                            "max_output_tokens": 4096,
                        },
                        "test",
                    ),
                    timeout=1.5,
                )
                threads.Event.return_value.set.assert_called_once_with()
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.9)
        self.assertEqual(
            result["stop"],
            {
                "reason": "cut off",
                "budget_exhausted": "wall_clock",
                "error": None,
            },
        )
        self.assertTrue(all(stream.is_closed for stream in streams))
        if mode == "headers":
            self.assertEqual(len(requests), 1)
        else:
            self.assertTrue(closed.is_set())
            self.assertEqual(len(requests), 2)
            self.assertTrue(requests[-1].url.path.endswith("/resp_test/cancel"))
            self.assertEqual(
                result["responses"][0]["blocks"],
                [
                    {"type": "text", "text": "checkpoint"},
                ],
            )

    async def test_silence_after_progress_cannot_extend_deadline(self):
        await self.run_stream("silent")

    async def test_transport_keepalives_cannot_extend_deadline(self):
        await self.run_stream("keepalives")

    async def test_waiting_for_headers_cannot_extend_deadline(self):
        await self.run_stream("headers")

    async def test_deadline_interrupts_reconnect_backoff(self):
        await self.run_stream("backoff")


class OpenAISyncEntryTests(unittest.TestCase):
    def test_entry_point_closes_client_after_completed_response(self):
        async def handle(request):
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=EventStream(
                    [
                        response_event("response.created", 0),
                        response_event("response.completed", 1),
                    ]
                ),
            )

        client = openai.AsyncOpenAI(
            api_key="test-key",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)),
        )
        with (
            patch("model_calls.openai.AsyncOpenAI", return_value=client),
            patch("model_calls.status"),
        ):
            result = model_calls.run_openai(
                {
                    "model": "test-model",
                    "effort": "high",
                    "wall_clock_min": 1,
                    "max_tool_calls": 10,
                    "max_output_tokens": 4096,
                },
                "test",
            )
        self.assertEqual(result["stop"]["reason"], "completed")
        self.assertTrue(client.is_closed())


class UsageTests(unittest.TestCase):
    def test_anthropic_cumulative_snapshots_are_not_added(self):
        tracker = model_calls.UsageTracker("anthropic")
        tracker.begin()
        tracker.observe(
            SimpleNamespace(
                input_tokens=100,
                output_tokens=1,
                cache_read_input_tokens=20,
                cache_creation_input_tokens=30,
            )
        )
        tracker.observe(SimpleNamespace(output_tokens=10))
        tracker.observe(SimpleNamespace(output_tokens=20))
        self.assertEqual(tracker.snapshot()["output"], 20)
        self.assertIn("input 150", tracker.summary())
        self.assertIn("incomplete", tracker.summary())
        final = SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=20,
            cache_creation_input_tokens=30,
        )
        self.assertTrue(tracker.finish(final)["usage_complete"])
        tracker.begin()
        tracker.finish(SimpleNamespace(input_tokens=50, output_tokens=7))
        self.assertEqual(
            tracker.snapshot(),
            dict(input=150, output=27, cache_read=20, cache_write=30, thinking=0),
        )
        self.assertIn("input 200", tracker.summary())
        self.assertNotIn("incomplete", tracker.summary())

    def test_openai_cache_and_thinking_are_subsets_not_extra_tokens(self):
        tracker = model_calls.UsageTracker("openai")
        tracker.begin()
        tracker.finish(
            SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                input_tokens_details=SimpleNamespace(cached_tokens=50),
                output_tokens_details=SimpleNamespace(reasoning_tokens=15),
            )
        )
        self.assertIn("input 100", tracker.summary())
        self.assertIn("output 20", tracker.summary())
        self.assertEqual(tracker.snapshot()["thinking"], 15)

    def test_missing_usage_stays_incomplete_after_later_complete_response(self):
        tracker = model_calls.UsageTracker("anthropic")
        tracker.begin()
        tracker.observe(SimpleNamespace(input_tokens=80, output_tokens=5))
        self.assertFalse(tracker.finish()["usage_complete"])
        tracker.begin()
        tracker.finish(SimpleNamespace(input_tokens=10, output_tokens=2))
        self.assertEqual(tracker.snapshot()["output"], 7)
        self.assertFalse(tracker.complete)
        self.assertIn("incomplete", tracker.summary())

    def test_heartbeat_and_status_include_latest_reported_input_output(self):
        tracker = model_calls.UsageTracker("anthropic")
        tracker.begin()
        tracker.observe(SimpleNamespace(input_tokens=80, output_tokens=5))
        finished = Mock()
        finished.wait.side_effect = [False, True]
        with (
            patch("model_calls.time.monotonic", return_value=180),
            patch("builtins.print") as output,
        ):
            model_calls.heartbeat(0, {"last": 0, "tracker": tracker}, finished)
        message = output.call_args.args[0]
        self.assertIn("input 80", message)
        self.assertIn("output 5", message)
        self.assertIn("incomplete", message)


def anthropic_events(stop="end_turn", input_tokens=100, output_tokens=20, tool=False):
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-fable-5-2",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 10,
                    "cache_creation_input_tokens": 5,
                },
            },
        }
    ]
    if tool:
        block = {
            "type": "server_tool_use",
            "id": "srvtoolu_test",
            "name": "bash_code_execution",
            "input": {"command": "echo test"},
        }
        delta = {"type": "input_json_delta", "partial_json": ""}
    else:
        block = {"type": "text", "text": ""}
        delta = {"type": "text_delta", "text": "checkpoint"}
    events += [
        {"type": "content_block_start", "index": 0, "content_block": block},
        {"type": "content_block_delta", "index": 0, "delta": delta},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
        {"type": "message_stop"},
    ]
    return events


class AnthropicEventStream(EventStream):
    async def __aiter__(self):
        for event in self.events:
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


class AnthropicStreamTests(unittest.IsolatedAsyncioTestCase):
    async def run_events(self, batches, **changes):
        requests, streams, updates = [], [], []

        async def handle(request):
            requests.append(request)
            batch = batches.pop(0)
            if callable(batch):
                response = await batch(request)
            else:
                response = httpx2.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=AnthropicEventStream(batch),
                )
            streams.append(response)
            return response

        async with anthropic.AsyncAnthropic(
            api_key="test-key",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)),
        ) as client:
            with patch(
                "model_calls.status",
                side_effect=lambda started, text, tracker=None: updates.append(
                    text + (" " + tracker.summary() if tracker else "")
                ),
            ):
                result = await model_calls.anthropic_run(
                    client,
                    {
                        "model": "claude-fable-5-2",
                        "effort": "high",
                        "wall_clock_min": 1,
                        "max_output_tokens": 16000,
                        "max_tool_calls": 30,
                        **changes,
                    },
                    "test",
                )
        self.assertTrue(all(stream.is_closed for stream in streams))
        return result, requests, updates

    async def test_final_usage_replaces_snapshots_and_continuations_sum(self):
        result, requests, updates = await self.run_events(
            [
                anthropic_events("pause_turn"),
                anthropic_events(input_tokens=200, output_tokens=30),
            ]
        )
        self.assertEqual(len(requests), 2)
        self.assertEqual(result["usage"]["input"], 300)
        self.assertEqual(result["usage"]["output"], 50)
        self.assertEqual(result["usage"]["cache_read"], 20)
        self.assertTrue(result["usage_complete"])
        second = json.loads(requests[1].content)
        self.assertEqual(second["max_tokens"], 15980)
        self.assertEqual(len(second["messages"]), 2)
        self.assertIn("input 330", updates[-1])
        self.assertIn("output 50", updates[-1])
        self.assertIn("input 115", updates[1])
        self.assertIn("incomplete", updates[1])

    async def test_dropped_stream_keeps_reported_usage_and_checkpoint(self):
        result, requests, _ = await self.run_events([anthropic_events()[:-1]])
        self.assertEqual(len(requests), 1)
        self.assertFalse(result["usage_complete"])
        self.assertEqual(result["usage"]["output"], 20)
        self.assertIn("StreamLost", result["stop"]["error"])
        self.assertEqual(
            result["responses"][0]["blocks"], [{"type": "text", "text": "checkpoint"}]
        )

    async def test_transport_reset_keeps_partial_usage_and_transcript(self):
        class BrokenStream(AnthropicEventStream):
            async def __aiter__(self):
                async for chunk in super().__aiter__():
                    yield chunk
                raise httpx2.ReadError("connection reset")

        async def broken(request):
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=BrokenStream(anthropic_events()[:4]),
            )

        result, requests, updates = await self.run_events([broken])
        self.assertEqual(len(requests), 1)
        self.assertFalse(result["usage_complete"])
        self.assertEqual(result["usage"]["input"], 100)
        self.assertEqual(result["usage"]["output"], 1)
        self.assertIn("ReadError", result["stop"]["error"])
        self.assertEqual(result["responses"][0]["blocks"][0]["text"], "checkpoint")
        self.assertIn("incomplete", updates[-1])

    async def test_tool_cap_interrupts_inside_response(self):
        result, requests, _ = await self.run_events(
            [anthropic_events(tool=True)], max_tool_calls=1
        )
        self.assertEqual(result["stop"]["budget_exhausted"], "tool_calls")
        self.assertEqual(result["tool_calls"], 1)
        self.assertFalse(result["usage_complete"])
        self.assertEqual(len(requests), 1)

    async def test_output_cap_does_not_start_continuation(self):
        result, requests, _ = await self.run_events(
            [anthropic_events("max_tokens", output_tokens=16000)]
        )
        self.assertEqual(result["stop"]["budget_exhausted"], "output_tokens")
        self.assertEqual(len(requests), 1)
        self.assertTrue(result["usage_complete"])

    async def test_silent_stream_and_headers_obey_deadline(self):
        # Warm Anthropic's lazy decoder before measuring short deadlines.
        await self.run_events([anthropic_events()])
        for mode in ("silent", "keepalives", "headers"):
            with self.subTest(mode=mode):

                class SlowStream(httpx2.AsyncByteStream):
                    async def __aiter__(self):
                        for event in anthropic_events()[:4]:
                            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
                        if mode == "keepalives":
                            while True:
                                await asyncio.sleep(0.01)
                                yield b": ping\n\n"
                        await asyncio.sleep(2)

                async def slow(request):
                    if mode == "headers":
                        await asyncio.sleep(2)
                    return httpx2.Response(
                        200,
                        headers={"content-type": "text/event-stream"},
                        stream=SlowStream(),
                    )

                start = time.monotonic()
                result, requests, updates = await asyncio.wait_for(
                    self.run_events([slow], wall_clock_min=0.15 / 60), timeout=1.2
                )
                self.assertLess(time.monotonic() - start, 0.9)
                self.assertEqual(len(requests), 1)
                self.assertEqual(result["stop"]["budget_exhausted"], "wall_clock")
                self.assertFalse(result["usage_complete"])
                self.assertIn("incomplete", updates[-1])
                if mode != "headers":
                    self.assertEqual(result["usage"]["input"], 100)
                    self.assertEqual(
                        result["responses"][0]["blocks"][0]["text"], "checkpoint"
                    )

    async def test_expired_deadline_starts_no_request(self):
        result, requests, _ = await self.run_events([], wall_clock_min=0)
        self.assertEqual(requests, [])
        self.assertEqual(result["stop"]["budget_exhausted"], "wall_clock")


class OpenAIUsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_usage_and_missing_usage_are_distinguished(self):
        for with_usage in (True, False):
            with self.subTest(with_usage=with_usage):
                event = response_event("response.completed", 1)
                if with_usage:
                    event["response"]["usage"] = {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "total_tokens": 120,
                        "input_tokens_details": {"cached_tokens": 50},
                        "output_tokens_details": {"reasoning_tokens": 15},
                    }

                async def handle(request):
                    return httpx2.Response(
                        200,
                        headers={"content-type": "text/event-stream"},
                        stream=EventStream(
                            [response_event("response.created", 0), event]
                        ),
                    )

                async with openai.AsyncOpenAI(
                    api_key="test-key",
                    http_client=httpx2.AsyncClient(
                        transport=httpx2.MockTransport(handle)
                    ),
                ) as client:
                    with patch("model_calls.status"):
                        result = await model_calls.openai_run(
                            client,
                            {
                                "model": "test-model",
                                "effort": "high",
                                "wall_clock_min": 1,
                                "max_tool_calls": 30,
                                "max_output_tokens": 16000,
                            },
                            "test",
                        )
                self.assertEqual(result["usage_complete"], with_usage)
                self.assertEqual(result["usage"]["input"], 100 if with_usage else 0)
                self.assertEqual(result["usage"]["output"], 20 if with_usage else 0)
                self.assertEqual(result["responses"][0]["usage_complete"], with_usage)


if __name__ == "__main__":
    unittest.main()
