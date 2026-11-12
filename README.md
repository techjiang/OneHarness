# OneHarness

给工具调用型 LLM agent 用的一层薄 harness：**一套接口，接所有 provider**。

不引入任何第三方依赖（纯标准库），也不试图做成框架。它只做三件事：

1. 把 OpenAI / Anthropic 两种线格式抹平成同一个 `Completion`；
2. 跑「模型要调工具 → 执行 → 结果回灌 → 再问」这个循环，且不让工具的错误炸掉循环；
3. 用 TOML + 环境变量管配置，用一条命令跑起来。

```console
$ oneharness run "看看 examples/app.conf 里的 timeout，太小了就改大"
  → read_file({"path": "examples/app.conf"})
  ✓ read_file: timeout=5
  → write_file({"path": "examples/app.conf", "content": "timeout=30\n"})
  ✓ write_file: wrote 11 chars to examples/app.conf
timeout 原来是 5，我改成了 30。
```

## 为什么需要它

每个 provider 的差异都琐碎但要命：工具调用参数是 JSON 字符串还是对象、tool 结果挂在
哪个字段、system prompt 是消息还是独立字段、usage 字段叫什么、429 该不该重试。
OneHarness 把这些差异收敛在 `providers/` 里，上层只有一个 `Provider.complete()`。

```python
from oneharness import Agent, Registry, create_provider, tool

@tool
def add(a: int, b: int) -> int:
    """把两个整数相加。"""
    return a + b

agent = Agent(create_provider("openai", model="gpt-4o-mini"), Registry([add]))
result = agent.run("37 加 58 等于多少？")
print(result.text)
```

`@tool` 直接读函数签名和类型注解生成 JSON Schema，不用手写 schema 字符串：

```python
@tool
def search(query: str, mode: Literal["fast", "deep"] = "fast", tags: list[str] = []) -> dict:
    """搜索文档。"""
```

## 安装

```console
$ pip install -e .          # 需要 Python 3.11+
$ oneharness --help
```

## 命令行

| 命令 | 作用 |
|---|---|
| `oneharness run "<prompt>"` | 跑一个 prompt，打印最终回答 |
| `oneharness chat` | 多轮交互会话，保留历史 |
| `oneharness tools` | 列出内置工具及其参数 |
| `oneharness config` | 打印解析后的配置（排查配置问题用） |

常用参数：

```console
$ oneharness run "修复测试" --workspace . --shell --max-steps 12
$ oneharness run "ping" --provider deepseek --model deepseek-chat --json
$ oneharness run "ping" --no-tools --quiet
```

`--json` 输出稳定的机器可读结构，适合挂到脚本里：

```json
{
  "text": "pong",
  "stopped_reason": "done",
  "model": "deepseek-chat",
  "usage": {"input_tokens": 24, "output_tokens": 2, "total_tokens": 26},
  "steps": [{"text": "pong", "tool_calls": [], "tool_results": [], "usage": {...}}]
}
```

退出码：`0` 正常收尾，`1` 配置/网络/工具链出错或步数耗尽，`2` 参数错误（argparse 惯例）。

## 配置

就近找 `oneharness.toml`（从当前目录逐级向上，最后看 `~/.config/`）：

```toml
[provider]
preset = "deepseek"          # openai | anthropic | deepseek | moonshot | qwen | ollama
model = "deepseek-chat"
temperature = 0
max_tokens = 4096

[agent]
system = "你是简洁的工程师，先看代码再改。"
max_steps = 8
```

环境变量优先于文件：`ONEHARNESS_PROVIDER`、`ONEHARNESS_MODEL`、`ONEHARNESS_API_KEY`、
`ONEHARNESS_BASE_URL`、`ONEHARNESS_SYSTEM`、`ONEHARNESS_MAX_STEPS`。

API Key 走各 provider 的常规变量名：`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、
`DEEPSEEK_API_KEY`、`MOONSHOT_API_KEY`、`DASHSCOPE_API_KEY`；`ollama` 不需要。

配 `base_url` 就能接任何 OpenAI 兼容服务（vLLM、llama.cpp、LM Studio、one-api 中转）：

```toml
[provider]
preset = "openai"
base_url = "http://localhost:8000/v1"
model = "Qwen2.5-72B-Instruct"
```

## 内置工具

| 工具 | 说明 |
|---|---|
| `read_file(path)` | 读工作区内的 UTF-8 文本，超过 200 KB 拒绝 |
| `write_file(path, content)` | 写文件，自动建父目录 |
| `list_dir(path=".")` | 列目录，子目录带 `/` 后缀 |
| `run_shell(command)` | 跑一条 shell 命令，默认 60s 超时，需显式 `--shell` 开启 |

文件工具都锁在 workspace 里，`../` 和指向外部的绝对路径直接拒绝。
`run_shell` 默认关闭——它不是沙箱，开了就等于把 shell 交给模型。

## 设计取舍

**工具失败不抛异常。** 参数写错、工具内部报错、调了不存在的工具，都会被转成
`ToolResult(is_error=True)` 回灌给模型，让它自己纠正。错误信息本身是给模型看的上下文，
不是给程序看的日志。真正抛异常的只有 provider 级故障（网络、鉴权、响应结构不对）。

**步数有上限。** `max_steps` 是硬闸，防止模型在工具调用里打转。耗尽时返回
`stopped_reason="max_steps"` 而不是假装成功。

**usage 累加。** `result.usage` 是整个 run 的总和，不是最后一轮。

**没有内置重试模型决策。** provider 层的 429/5xx 会自动退避重试（`transport.py`），
但「模型答得不对要不要重问」属于业务判断，交给调用方。

**同步 API。** 没有 asyncio。工具多半是文件 IO 和短 shell，并发收益不足以抵消复杂度；
需要并发就跑多个进程。

## 项目结构

```
oneharness/
  types.py        Message / ToolCall / Completion / Usage / ToolResult
  transport.py    urllib 封装的 JSON POST，带 429/5xx 退避重试
  providers/      base.py + openai.py + anthropic.py
  tools.py        @tool 装饰器、schema 生成、Registry
  agent.py        工具调用循环，on_event 可观测
  builtin.py      文件/目录/shell 工具
  config.py       TOML + 环境变量
  cli.py          run / chat / tools / config
tests/            73 个单元测试，含本地 HTTP mock
examples/         离线 e2e demo，不需要 API Key
```

## 开发

```console
$ python -m unittest discover -s tests -v    # 全部测试
$ python examples/e2e_demo.py                # 端到端，起本地假 provider
```

## License

MIT
