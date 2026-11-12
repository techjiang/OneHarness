import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oneharness.config import Config, find_config_file, load_config
from oneharness.errors import ConfigError


class McpConfigTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, text):
        path = self.root / "oneharness.toml"
        path.write_text(text)
        return path

    def test_single_server_table(self):
        path = self.write(
            """
            [[mcp]]
            command = ["npx", "-y", "server-filesystem", "/tmp"]
            prefix = "fs."
            timeout = 12
            """
        )
        config = load_config(path)
        self.assertEqual(len(config.mcp), 1)
        server = config.mcp[0]
        self.assertEqual(server.command, ["npx", "-y", "server-filesystem", "/tmp"])
        self.assertEqual(server.prefix, "fs.")
        self.assertEqual(server.timeout, 12)
        self.assertEqual(config.as_dict()["mcp"][0]["prefix"], "fs.")

    def test_command_may_be_a_string(self):
        path = self.write(
            """
            [[mcp]]
            command = "python3 -m my_server"
            """
        )
        config = load_config(path)
        self.assertEqual(config.mcp[0].command, ["python3", "-m", "my_server"])

    def test_unknown_keys_rejected(self):
        path = self.write(
            """
            [[mcp]]
            command = ["x"]
            nope = 1
            """
        )
        with self.assertRaises(ConfigError) as ctx:
            load_config(path)
        self.assertIn("unknown mcp[0] keys", str(ctx.exception))

    def test_missing_command_rejected(self):
        path = self.write(
            """
            [[mcp]]
            prefix = "fs."
            """
        )
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_mcp_must_be_a_table_array(self):
        path = self.write('mcp = "nope"\n')
        with self.assertRaises(ConfigError):
            load_config(path)


class DelegateAndParallelConfigTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, text):
        path = self.root / "oneharness.toml"
        path.write_text(text)
        return path

    def test_delegate_table(self):
        config = load_config(self.write('[delegate]\nmax_steps = 3\nsystem = "be brief"\n'))
        self.assertTrue(config.delegate.enabled)
        self.assertEqual(config.delegate.max_steps, 3)
        self.assertEqual(config.delegate.system, "be brief")
        self.assertTrue(config.as_dict()["delegate"]["enabled"])

    def test_delegate_is_off_by_default(self):
        self.assertFalse(load_config(self.write('[provider]\nmodel = "m"\n')).delegate.enabled)

    def test_delegate_unknown_key_rejected(self):
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.write("[delegate]\nnope = 1\n"))
        self.assertIn("unknown [delegate] keys", str(ctx.exception))

    def test_max_parallel_clamps_to_one(self):
        self.assertEqual(load_config(self.write("max_parallel = 0\n")).max_parallel, 1)
        self.assertEqual(load_config(self.write("max_parallel = 4\n")).max_parallel, 4)

    def test_delegate_max_parallel_defaults_to_inheritance(self):
        config = load_config(self.write("[delegate]\n"))
        self.assertEqual(config.delegate.max_parallel, 0)  # 0 = use the top-level value

    def test_delegate_max_parallel_from_file(self):
        config = load_config(self.write("[delegate]\nmax_parallel = 4\n"))
        self.assertEqual(config.delegate.max_parallel, 4)

    def test_max_steps_zero_in_the_file_is_an_error_not_a_default(self):
        # Before this, `max_steps = 0` in the file fell through to the default 8
        # and the only sign was an 8-step run somebody did not ask for.
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.write("max_steps = 0\n"))
        self.assertIn("at least 1", str(ctx.exception))
        with self.assertRaises(ConfigError):
            load_config(self.write("[delegate]\nmax_steps = 0\n"))

    def test_context_budget_from_file(self):
        self.assertEqual(load_config(self.write("context_budget = 32000\n")).context_budget, 32000)
        self.assertEqual(load_config(self.write("context_budget = 0\n")).context_budget, 0)

    def test_context_budget_must_be_zero_or_positive(self):
        with self.assertRaises(ConfigError):
            load_config(self.write("context_budget = -1\n"))
        with self.assertRaises(ConfigError):
            load_config(self.write('context_budget = "lots"\n'))

    def test_context_budget_env_override(self):
        path = self.write("context_budget = 1000\n")
        with mock.patch.dict(os.environ, {"ONEHARNESS_CONTEXT_BUDGET": "8000"}):
            self.assertEqual(load_config(path).context_budget, 8000)

    def test_max_steps_env_below_one_is_an_error(self):
        with (
            mock.patch.dict("os.environ", {"ONEHARNESS_MAX_STEPS": "0"}),
            self.assertRaises(ConfigError),
        ):
            load_config(self.write("max_steps = 2\n"))

    def test_max_parallel_env_override(self):
        path = self.write("max_parallel = 2\n")
        with mock.patch.dict(os.environ, {"ONEHARNESS_MAX_PARALLEL": "5"}):
            self.assertEqual(load_config(path).max_parallel, 5)
        with (
            mock.patch.dict(os.environ, {"ONEHARNESS_MAX_PARALLEL": "x"}),
            self.assertRaises(ConfigError),
        ):
            load_config(path)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._saved_env = {
            k: os.environ.pop(k) for k in list(os.environ) if k.startswith("ONEHARNESS_")
        }

    def tearDown(self):
        self._tmp.cleanup()
        for key, value in self._saved_env.items():
            os.environ[key] = value
        for key in [k for k in os.environ if k.startswith("ONEHARNESS_")]:
            os.environ.pop(key)

    def write(self, text, name="oneharness.toml"):
        path = self.root / name
        path.write_text(text)
        return path

    def test_defaults(self):
        config = Config()
        self.assertEqual(config.provider.preset, "openai")
        self.assertEqual(config.max_steps, 8)

    def test_loads_toml(self):
        path = self.write(
            """
            [provider]
            preset = "anthropic"
            model = "claude-sonnet-4-5"
            temperature = 0.3

            [agent]
            system = "be terse"
            max_steps = 3
            """
        )
        config = load_config(path)
        self.assertEqual(config.provider.preset, "anthropic")
        self.assertEqual(config.provider.temperature, 0.3)
        self.assertEqual(config.system, "be terse")
        self.assertEqual(config.max_steps, 3)

    def test_top_level_shorthand(self):
        config = load_config(self.write('system = "top"\nmax_steps = 2\n'))
        self.assertEqual(config.system, "top")
        self.assertEqual(config.max_steps, 2)

    def test_missing_explicit_path(self):
        with self.assertRaises(ConfigError):
            load_config(self.root / "nope.toml")

    def test_invalid_toml(self):
        with self.assertRaises(ConfigError):
            load_config(self.write("this is not toml ["))

    def test_unknown_key_rejected(self):
        with self.assertRaises(ConfigError) as ctx:
            load_config(self.write("[provider]\nmodle = 'typo'\n"))
        self.assertIn("modle", str(ctx.exception))

    def test_unknown_section_rejected(self):
        with self.assertRaises(ConfigError):
            load_config(self.write("[providr]\npreset = 'openai'\n"))

    def test_env_overrides_file(self):
        path = self.write("[provider]\npreset = 'openai'\nmodel = 'from-file'\n")
        os.environ["ONEHARNESS_MODEL"] = "from-env"
        os.environ["ONEHARNESS_MAX_STEPS"] = "5"
        config = load_config(path)
        self.assertEqual(config.provider.model, "from-env")
        self.assertEqual(config.max_steps, 5)

    def test_env_invalid_int(self):
        os.environ["ONEHARNESS_MAX_STEPS"] = "many"
        with self.assertRaises(ConfigError):
            load_config(self.write(""))

    def test_find_walks_upwards(self):
        nested = self.root / "a" / "b"
        nested.mkdir(parents=True)
        self.write("[provider]\npreset = 'ollama'\n")
        self.assertEqual(find_config_file(nested), self.root / "oneharness.toml")

    def test_options_drops_unset(self):
        options = Config().provider.options()
        self.assertNotIn("model", options)
        self.assertNotIn("api_key", options)
        self.assertEqual(options["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
