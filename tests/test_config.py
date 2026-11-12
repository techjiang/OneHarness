import os
import tempfile
import unittest
from pathlib import Path

from oneharness.config import Config, find_config_file, load_config
from oneharness.errors import ConfigError


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._saved_env = {
            k: os.environ.pop(k)
            for k in list(os.environ)
            if k.startswith("ONEHARNESS_")
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
