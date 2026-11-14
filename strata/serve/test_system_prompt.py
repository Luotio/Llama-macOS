"""Configured system prompts and aliases, tested without an inference engine."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.request
from unittest import mock

from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages
from serve.server import ByteTokenizer, MockEngine, Service, serve


TEMPLATE = Path(__file__).with_name("chat_template.jinja")
IDENTITY = 'You are Michael, a "coding agent".\nUse {{ tools }} when appropriate.'


class ConfiguredSystemPrompt(unittest.TestCase):
    def setUp(self):
        self.template = ChatTemplate(TEMPLATE, system_prompt=IDENTITY)
        self.tokenizer = ByteTokenizer()
        self.service = Service(MockEngine(self.tokenizer, "</think>\n\nok", max_context=4096),
                               self.tokenizer, self.template, model_name="qwen")
        self.service.set_aliases(["Michael"])

    def test_existing_system_developer_tools_and_messages_are_preserved(self):
        messages = [{"role": "system", "content": "Client system instructions."},
                    {"role": "developer", "content": "Client developer instructions."},
                    {"role": "user", "content": "Fix the code."}]
        tools = [{"type": "function", "function": {"name": "inspect", "parameters": {"type": "object"}}}]
        original = copy.deepcopy(messages)
        prompt = self.service.render_prompt(messages, tools, {})
        self.assertEqual(prompt.count(IDENTITY), 1)
        self.assertIn("Client system instructions.", prompt)
        self.assertIn("Client developer instructions.", prompt)
        self.assertIn("inspect", prompt)
        self.assertIn("Fix the code.", prompt)
        self.assertEqual(messages, original)

    def test_plain_and_anthropic_requests_get_the_same_identity(self):
        for convert, req in ((openai_to_messages, {"messages": [{"role": "user", "content": "hello"}]}),
                             (anthropic_to_messages, {"system": "Client instructions.",
                                                      "messages": [{"role": "user", "content": "hello"}]})):
            messages, tools, kwargs = convert(req)
            prompt = self.service.render_prompt(messages, tools, kwargs)
            self.assertEqual(prompt.count(IDENTITY), 1)
            self.assertEqual(self.service.encode_prompt(messages, tools, kwargs),
                             self.tokenizer.encode(prompt, parse_special=True))

    def test_template_source_contains_the_configured_prefix(self):
        plain = ChatTemplate(TEMPLATE)
        self.assertNotEqual(self.template.source, plain.source)
        self.assertIn("You are Michael", self.template.source)
        self.assertNotIn("You are Michael", plain.render([{"role": "user", "content": "hello"}]))
        self.assertEqual(self.template.caps, plain.caps)

    def test_multimodal_message_keeps_image_placeholder(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "Describe this"},
                                                {"type": "image", "image": "placeholder"}]}]
        prompt = self.template.render(messages)
        self.assertIn(IDENTITY, prompt)
        self.assertIn("Describe this", prompt)
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>", prompt)

    def test_http_alias_and_apply_template(self):
        with mock.patch.object(self.service, "start_telemetry"):
            httpd = serve(self.service, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/v1/models", timeout=10) as response:
                self.assertEqual([x["id"] for x in json.load(response)["data"]], ["qwen", "Michael"])
            with urllib.request.urlopen(base + "/props", timeout=10) as response:
                source = json.load(response)["chat_template"]
            body = {"model": "Michael", "messages": [{"role": "system", "content": "Client instructions."},
                                                      {"role": "user", "content": "hello"}]}
            def post(path, payload=body):
                request = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                                 headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=10) as response:
                    return json.load(response)
            prompt = post("/apply-template")["prompt"]
            self.assertEqual(prompt.count(IDENTITY), 1)
            self.assertIn("Client instructions.", prompt)
            exposed = self.template.template.environment.from_string(source)
            self.assertEqual(exposed.render(messages=body["messages"], tools=None, add_generation_prompt=True),
                             self.template.render(body["messages"]))
            self.assertEqual(post("/v1/chat/completions")["model"], "Michael")
            anthropic = {"model": "Michael", "system": "Client instructions.",
                         "messages": [{"role": "user", "content": "hello"}]}
            messages, tools, kwargs = anthropic_to_messages(anthropic, self.service.anthropic_think_unasked)
            expected = len(self.service.encode_prompt(messages, tools, kwargs, store=False))
            self.assertEqual(post("/v1/messages/count_tokens", anthropic)["input_tokens"], expected)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_main_wires_configured_prompt_and_alias_before_serving(self):
        import serve.server as server

        class ReachedServe(Exception):
            pass

        def capture(service, **_):
            self.assertEqual(service.aliases, ["Michael"])
            self.assertIn(IDENTITY, service.render_prompt([{"role": "user", "content": "hello"}], None, {}))
            raise ReachedServe

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"system_prompt": IDENTITY, "aliases": ["Michael"]}), encoding="utf-8")
            with mock.patch.object(server, "serve", capture):
                with self.assertRaises(ReachedServe):
                    server.main(["--engine", "mock", "--port", "0", "--config", str(config)])
            config.write_text(json.dumps({"system_prompt": ["not a string"]}), encoding="utf-8")
            with mock.patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as exc:
                server.main(["--engine", "mock", "--port", "0", "--config", str(config)])
            self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()