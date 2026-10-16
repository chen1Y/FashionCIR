import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shoes_online_pilot import (BASE_URL, Budget, Client, NoRedirect, checked_endpoint,
                                corrected_critic, optimized_prompt, request_body, select_queries)


class OnlinePilotTests(unittest.TestCase):
    def test_only_verified_https_api_hosts_accept_credentials(self):
        self.assertTrue(checked_endpoint(BASE_URL).endswith("/chat/completions"))
        for url in ("http://dashscope.aliyuncs.com/compatible-mode/v1",
                    "https://dashscope.aliyuncs.com.evil.invalid/compatible-mode/v1",
                    "https://user@dashscope.aliyuncs.com/compatible-mode/v1",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1?key=test"):
            with self.assertRaises(ValueError):
                checked_endpoint(url)
        with self.assertRaises(RuntimeError):
            NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://elsewhere.invalid")

    def test_cap_reserves_failed_or_unknown_calls_and_never_sends_over_limit(self):
        budget = Budget(.5, 1)
        budget.reserve(768)
        self.assertEqual(budget.calls, 1)
        with self.assertRaises(RuntimeError):
            budget.reserve(256)
        cheap = Budget(.001, 10)
        with self.assertRaises(RuntimeError):
            cheap.reserve(768)
        self.assertEqual(cheap.calls, 0)

    def test_reported_usage_is_metered_and_bound_violation_stops(self):
        budget = Budget()
        budget.account(dict(prompt_tokens=1000, completion_tokens=500))
        self.assertAlmostEqual(budget.estimated_cny, .0006)
        budget.account(None)
        self.assertEqual(budget.usage_missing_calls, 1)
        with self.assertRaises(RuntimeError):
            budget.account(dict(prompt_tokens=17000, completion_tokens=1))

    def test_request_does_not_contain_target_identity_or_secret(self):
        body = request_body("data:image/jpeg;base64,REFERENCE", "make the shoe blue", 768)
        text = json.dumps(body)
        self.assertNotIn("Authorization", text)
        self.assertNotIn("api_key", text)
        self.assertEqual(len(body["messages"][0]["content"]), 2)
        self.assertFalse(body["enable_thinking"])
        self.assertEqual(body["response_format"], {"type": "json_object"})

    def test_credentials_only_in_header_and_redacted_from_response(self):
        client = Client("test-secret-value", BASE_URL, Budget())
        answer = {"choices": [{"message": {"content": "test-secret-value"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 100, "completion_tokens": 20}, "model": "qwen3.7-flash"}
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps(answer).encode()
        client.opener = Mock()
        client.opener.open.return_value = response
        result = client.ask("data:image/jpeg;base64,REFERENCE", "Return JSON only", 256)
        request = client.opener.open.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret-value")
        self.assertNotIn(b"test-secret-value", request.data)
        self.assertEqual(result["text"], "[REDACTED]")
        self.assertNotIn("test-secret-value", json.dumps(result))

    def test_sampling_is_deterministic_balanced_and_preserves_rejections(self):
        rows = [dict(source_index=i, candidate=f"r{i}", target=f"t{i}", modification_text="lighter") for i in range(8)]
        local = {"samples": [dict(r, query_index=r["source_index"], status="ok" if r["source_index"]%2 else "failed") for r in rows]}
        selected = select_queries(rows, local, 4, 42)
        self.assertEqual(selected, select_queries(rows, local, 4, 42))
        self.assertEqual(sum(r["local_sample"]["status"]=="ok" for r in selected), 2)
        self.assertEqual(len({r["source_index"] for r in selected}), 4)

    def test_optimization_preserves_comparative_direction_and_edit_critic(self):
        prompt = optimized_prompt("is lighter")
        self.assertIn("lighter THAN THE REFERENCE", prompt)
        self.assertIn("MUST be exactly reference_image OR modification_text", prompt)
        critic = corrected_critic("fewer straps", {})
        self.assertIn("reference is NOT the target", critic)
        self.assertIn("does not", critic)


if __name__ == "__main__":
    unittest.main()
