import json
import unittest
from unittest.mock import Mock, patch

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


class OpenAIDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.requests = []
        self.streams = []
        self.actions = []
        self.client = openai.OpenAI(
            api_key="test-key",
            http_client=httpx2.Client(transport=httpx2.MockTransport(self.handle)),
        )
        self.addCleanup(self.client.close)
        for target, options in (
            ("model_calls.time.monotonic", {"side_effect": lambda: self.now}),
            ("model_calls.time.sleep", {"side_effect": self.advance}),
            ("model_calls.status", {}),
        ):
            patcher = patch(target, **options)
            mocked = patcher.start()
            self.addCleanup(patcher.stop)
            if target.endswith("sleep"):
                self.sleep = mocked

    def advance(self, seconds):
        self.now += seconds

    def handle(self, request):
        self.requests.append(request)
        action = self.actions.pop(0)
        if callable(action):
            response = action(request)
        else:
            payload = "".join(f"data: {json.dumps(event)}\n\n" for event in action)
            response = httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=httpx2.ByteStream(payload.encode()),
            )
        self.streams.append(response)
        return response

    def events(self, deadline):
        return model_calls.openai_events(
            self.client, {"model": "test-model", "input": "test"}, 0, deadline
        )

    def timeout(self, request):
        seconds = request.extensions["timeout"]["read"]
        self.advance(seconds)
        raise httpx2.ReadTimeout("timed out", request=request)

    def body_timeout(self, request):
        advance = self.advance

        class TimeoutStream(httpx2.SyncByteStream):
            def __iter__(self):
                event = response_event("response.created", 0)
                yield f"data: {json.dumps(event)}\n\n".encode()
                advance(request.extensions["timeout"]["read"])
                raise httpx2.ReadTimeout("stream timed out", request=request)

        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=TimeoutStream(),
        )

    def run_model(self, deadline):
        finished = Mock()
        with (
            patch("model_calls.openai.OpenAI", return_value=self.client),
            patch("model_calls.threading.Thread"),
            patch("model_calls.threading.Event", return_value=finished),
        ):
            result = model_calls.run_openai(
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

    def test_expired_deadline_does_not_create_response(self):
        for deadline in (0, -1):
            self.assertEqual(list(self.events(deadline)), [])
        self.assertEqual(self.requests, [])

    def test_backoff_reaching_or_overshooting_deadline_does_not_reconnect(self):
        for overshoot in (0, 0.1):
            with self.subTest(overshoot=overshoot):
                self.now = 0
                self.requests.clear()
                self.actions = [[response_event("response.created", 0)]]
                self.sleep.side_effect = lambda seconds: self.advance(
                    seconds + overshoot
                )
                self.assertEqual(len(list(self.events(1))), 1)
                self.assertEqual(len(self.requests), 1)
                self.assertTrue(self.streams[-1].is_closed)

    def test_reconnect_uses_remaining_timeout_and_skips_seen_events(self):
        created = response_event("response.created", 0)
        self.actions = [[created], [created, response_event("response.completed", 1)]]
        events = list(self.events(3))
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

    def test_deadline_during_reconnect_timeout_cancels_and_reports_wall_clock(self):
        self.actions = [
            [response_event("response.created", 0)],
            self.timeout,
            lambda request: httpx2.Response(200, json={"id": "resp_test"}),
        ]
        result = self.run_model(3)
        self.assertEqual(
            result["stop"],
            {"reason": "cut off", "budget_exhausted": "wall_clock", "error": None},
        )
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(self.requests[-1].url.path, "/v1/responses/resp_test/cancel")
        self.assertEqual(self.requests[-1].extensions["timeout"]["read"], 10)
        self.sleep.assert_called_once_with(1)

    def test_backoff_deadline_cancels_even_if_cancellation_times_out(self):
        self.actions = [[response_event("response.created", 0)], self.timeout]
        result = self.run_model(1)
        self.assertEqual(result["stop"]["budget_exhausted"], "wall_clock")
        self.assertIsNone(result["stop"]["error"])
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[-1].url.path, "/v1/responses/resp_test/cancel")

    def test_initial_request_timeout_at_deadline_reports_wall_clock(self):
        self.actions = [self.timeout]
        result = self.run_model(1)
        self.assertEqual(result["stop"]["budget_exhausted"], "wall_clock")
        self.assertIsNone(result["stop"]["error"])
        self.assertEqual(len(self.requests), 1)
        self.sleep.assert_not_called()

    def test_timeout_reading_stream_body_closes_and_cancels(self):
        for reconnect in (False, True):
            with self.subTest(reconnect=reconnect):
                self.now = 0
                self.requests.clear()
                self.actions = (
                    [[response_event("response.created", 0)]] if reconnect else []
                ) + [
                    self.body_timeout,
                    lambda request: httpx2.Response(200, json={"id": "resp_test"}),
                ]
                result = self.run_model(3)
                self.assertEqual(result["stop"]["budget_exhausted"], "wall_clock")
                self.assertIsNone(result["stop"]["error"])
                self.assertEqual(len(self.requests), 3 if reconnect else 2)
                self.assertEqual(
                    self.requests[-1].url.path, "/v1/responses/resp_test/cancel"
                )

    def test_timeout_before_deadline_can_reconnect(self):
        def early_timeout(request):
            self.advance(0.25)
            raise httpx2.ReadTimeout("timed out", request=request)

        self.actions = [
            [response_event("response.created", 0)],
            early_timeout,
            [response_event("response.created", 0), response_event("response.completed", 1)],
        ]
        self.assertEqual(len(list(self.events(10))), 2)
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(self.requests[-1].extensions["timeout"]["read"], 6.75)

    def test_retry_limit_reports_stream_lost_without_final_sleep(self):
        self.actions = [[response_event("response.created", 0)], [], []]
        with patch("model_calls.STREAM_RESUMES", 2):
            with self.assertRaises(model_calls.StreamLost):
                list(self.events(100))
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(self.sleep.call_count, 2)
        self.assertEqual(self.now, 3)

    def test_completed_response_does_not_cancel(self):
        self.actions = [
            [response_event("response.created", 0), response_event("response.completed", 1)]
        ]
        result = self.run_model(3)
        self.assertEqual(result["stop"]["reason"], "completed")
        self.assertEqual(len(self.requests), 1)

    def test_closing_events_closes_active_stream_without_reconnecting(self):
        self.actions = [[response_event("response.created", 0)]]
        events = self.events(3)
        next(events)
        self.assertFalse(self.streams[0].is_closed)
        events.close()
        self.assertTrue(self.streams[0].is_closed)
        self.assertEqual(len(self.requests), 1)
        self.sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
