import unittest

from oneharness.types import Message, ToolCall, Usage, dumps


class ToolCallTests(unittest.TestCase):
    def test_parses_json_arguments(self):
        call = ToolCall.from_raw(call_id="c1", name="add", arguments='{"a": 1, "b": 2}')
        self.assertEqual(call.arguments, {"a": 1, "b": 2})
        self.assertIsNone(call.arguments_error)

    def test_accepts_dict_arguments(self):
        call = ToolCall.from_raw(call_id="c1", name="add", arguments={"a": 1})
        self.assertEqual(call.arguments, {"a": 1})
        self.assertEqual(call.arguments_text(), '{"a":1}')

    def test_records_invalid_json(self):
        call = ToolCall.from_raw(call_id="c1", name="add", arguments="{oops")
        self.assertEqual(call.arguments, {})
        self.assertIn("not valid JSON", call.arguments_error)
        self.assertEqual(call.arguments_text(), "{oops")

    def test_rejects_non_object_arguments(self):
        call = ToolCall.from_raw(call_id="c1", name="add", arguments="[1, 2]")
        self.assertIn("must be a JSON object", call.arguments_error)

    def test_empty_arguments_become_empty_dict(self):
        call = ToolCall.from_raw(call_id="c1", name="ping", arguments="")
        self.assertEqual(call.arguments, {})
        self.assertIsNone(call.arguments_error)


class MessageTests(unittest.TestCase):
    def test_assistant_with_tool_calls(self):
        message = Message.assistant("", [ToolCall(id="c1", name="add")])
        data = message.as_dict()
        self.assertEqual(data["tool_calls"], [{"id": "c1", "name": "add", "arguments": {}}])

    def test_tool_message_carries_call_id(self):
        data = Message.tool("c1", "42", "add").as_dict()
        self.assertEqual(data["tool_call_id"], "c1")
        self.assertEqual(data["name"], "add")


class UsageTests(unittest.TestCase):
    def test_add_and_totals(self):
        total = Usage(1, 2) + Usage(3, 4)
        self.assertEqual(total.input_tokens, 4)
        self.assertEqual(total.total_tokens, 10)


class DumpsTests(unittest.TestCase):
    def test_compact_and_non_ascii(self):
        self.assertEqual(dumps({"a": "中文"}), '{"a":"中文"}')


if __name__ == "__main__":
    unittest.main()
