# OneHarness

给工具调用型 LLM agent 用的一层薄 harness：**一套接口，接所有 provider**。

不引入任何第三方依赖（纯标准库），也不试图做成框架。它只做八件事：

1. 把 OpenAI / Anthropic / Gemini / Bedrock / Azure 五种线格式抹平成同一个 `Completion`——
   包括 Bedrock 的请求签名和二进制流；
2. 跑「模型要调工具 → 执行 → 结果回灌 → 再问」这个循环，且不让工具的错误炸掉循环；
3. 支持流式输出：token 边生成边出，工具调用在**参数还在到达时**就可见，连接断了也留住
   已经收到的半句；
4. 一轮里的多个工具调用可以并发跑，也可以把整个任务丢给带全新上下文的子 agent ——
   包括一次派出去好几个；
5. 用 TOML + 环境变量管配置，用一条命令跑起来，还能顺手接管 MCP server 的工具；
6. 能把一个越跑越长的工具循环塞进模型的窗口：超了就在发请求前裁，裁完还是超就
   按 provider 的原话降预算重试，实在没得裁就停手说清楚；
7. 能验证自己：`doctor` 用真 provider 各发一次最小请求并**判断是谁的错**，
   请求层可 `--record` / `--replay` 成 tape，回放不需要 key 也不需要网络；
8. 能说清慢在哪：`--timing` 把每一步拆成「等模型」和「等工具」，并给出首 token 时间。

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
| `oneharness presets` | 列出 provider 预设、默认模型和端点 |
| `oneharness doctor` | 对真 provider 各发一次最小请求，判断配置/密钥/模型哪一层出问题 |
| `oneharness config` | 打印解析后的配置（排查配置问题用） |

常用参数：

```console
$ oneharness run "修复测试" --workspace . --shell --max-steps 12
$ oneharness run "ping" --provider deepseek --model deepseek-chat --json
$ oneharness run "ping" --no-tools --quiet
$ oneharness run "看看这个仓库" --stream
$ oneharness run "整理这次改动" --delegate --parallel 4
$ oneharness run "查一下今天的文件" --mcp "npx -y @modelcontextprotocol/server-filesystem ."
$ oneharness run "读一遍整个仓库" --context-budget 32000   # 单个请求的 token 上限
$ oneharness run "ping" --record run.json     # 把这次的所有请求存成 tape
$ oneharness run "ping" --replay run.json     # 不联网、不需要 key，重放同一次
$ oneharness run "整理这次改动" --timing       # 每步拆成「等模型」和「等工具」
```

`--json` 输出稳定的机器可读结构，适合挂到脚本里：

```json
{
  "text": "pong",
  "stopped_reason": "done",
  "model": "deepseek-chat",
  "usage": {"input_tokens": 24, "output_tokens": 2, "total_tokens": 26},
  "steps": [
    {"text": "pong", "tool_calls": [], "tool_results": [], "usage": {...},
     "seconds": 0.61, "model_seconds": 0.61, "tool_seconds": 0.0}
  ],
  "timing": {"total_seconds": 0.62, "model_seconds": 0.61, "tool_seconds": 0.0, "...": "..."}
}
```

`--json` 与 `--delegate` 可以一起用：子 agent 的 token 单独打到 stderr，stdout 保持是纯 JSON，
并且多出一个 `delegation` 字段（`seconds` 是子 agent 花掉的墙钟，会话总时间对得上）：

```json
{
  "text": "...",
  "delegation": {
    "runs": 3,
    "seconds": 1.84,
    "usage": {"input_tokens": 1260, "output_tokens": 174, "total_tokens": 1434},
    "by_label": {"sub-1": {"input_tokens": 420, "output_tokens": 58, "total_tokens": 478}}
  }
}
```

父 agent 没有委派时这个字段不出现——空对象比缺字段更容易让人以为坏掉了。

退出码：`0` 正常收尾，`1` 配置/网络/工具链出错或步数耗尽，`2` 参数错误（argparse 惯例）。

## 流式输出

`--stream` 让 token 一边生成一边打印，但结果结构和非流式完全一样：工具循环仍在跑，
`--json` 也照常可用（`--json` 与 `--stream` 同时给时以 JSON 为准）。

```console
$ oneharness run "把两个配置文件都改一下" --stream
  … write_file#0({"path":…
  … write_file#1(…
  … write_file#0( "a.conf",…
  … write_file#1({"path":…
  … write_file#0( "content": "timeout=30\n"}…
  … write_file#1( "b.conf", "content": "retries=3\n"}…
  ✓ write_file: wrote 11 chars to a.conf
  ✓ write_file: wrote 10 chars to b.conf
Two files to fix. Both written. a.conf gets timeout=30, b.conf gets retries=3.
```

三个 provider 都实现了真正的 SSE 解析。细节上各有各的坑，都收在 provider 里：

- OpenAI 的工具调用参数是**跨多个 chunk 的 JSON 片段**，按 `index` 归并后一次性解析；
- Anthropic 的工具输入走 `input_json_delta`，同样要拼；`input_tokens` 在
  `message_start` 里、`output_tokens` 在 `message_delta` 里，分两处取；
- Gemini 走 `streamGenerateContent?alt=sse`，函数调用以完整对象出现，不需要拼。

如果某个 provider 没实现 `stream()`，基类会退化成「一次性发完」，`--stream` 依然能用。

### 工具调用在写完之前就露面

上面的 `… write_file#0(...)` 是**参数还在到达时**就画出来的那一行，不是等工具跑完才补的。
`#N` 是 provider 自己的 `index`：同一轮里两个调用会交错着拼参数，行首的编号是唯一能分辨
「这个逗号属于谁」的东西。

以前做不到这件事：完整参数只存在于 `done` 事件里，于是「有个 `write_file` 要来了」这条信息
总是在模型把两 KB 正文写完**之后**才出现——正好是它失去价值的那一刻。现在 `StreamEvent`
多了一种 `tool`（`text` / `tool` / `done`），CLI 相应多一个 `tool_call_delta` 事件：

```json
{"type": "tool_call_delta", "step": 1, "index": 0, "id": "call_a", "name": "write_file", "fragment": "{\"path\":"}
```

`fragment` 为空表示「这个调用存在，参数还没有」。把同 `index` 的 `fragment` 依次接起来
就是最终的参数 JSON——所以它既是给人看的进度，也是一份能自己拼的增量流。

三个适配器各有各的时机，因为各家的帧结构不同：OpenAI 的名和参数挤在同一个 `tool_calls`
delta 里，Anthropic 在 `content_block_start` 就给了 id 和 name（参数一个字节都还没有），
Gemini 则是一个完整对象、没有增量可言。

### 流断了，半句话还在

连接中途断掉（超时、代理掐了、服务器崩了）时，已经收到的文本不会被丢掉：

```console
$ oneharness run "..." --stream
  ! stream ended without a final completion -- kept 38 chars it had already said
Two files to fix. Both written. a.conf gets timeout=30,
[the answer above is incomplete]
```

`stopped_reason="interrupted"`，`--json` 里多一个 `interrupted` 字段写原因，退出码 1。
Ctrl-C 同理，只是原因写作 `interrupted by user (Ctrl-C)`、退出码 130——和 shell 对 Ctrl-C
的约定一致，`oneharness run ... && ...` 才不会把「人主动停了」当成成功。

这条不对称是刻意的：**非流式没有半句话可言**，body 没解析出来就是没有，仍然抛异常。
流式之所以能留住，是因为那些字节已经真的到了。

判断「断了」的依据是 provider 自己的收尾标记，不是「读完了」：OpenAI 的 `finish_reason`、
Anthropic 的 `stop_reason`、Bedrock 的 `messageStop`、Gemini 的 `finishReason`。少一个都算截断——
适配器不再替服务器编一个「正常结束」，因为它不知道那是不是正常结束。

## MCP

MCP server 本质是「一个说 JSON-RPC 的子进程」，用标准库就能说完，所以没有引依赖。
`oneharness` 只实现 tools 这一层（`initialize` / `tools/list` / `tools/call`）——resources 和
prompts 目前对 agent 循环没有用，写了也是死代码。

```console
$ oneharness run "读一下 /tmp/notes.md" \
    --mcp "npx -y @modelcontextprotocol/server-filesystem /tmp"
  + mcp npx: read_file
  + mcp npx: list_directory
  → read_file({"path": "/tmp/notes.md"})
  ✓ read_file: # 备忘 …
```

也可以写在配置里，长期生效：

```toml
[[mcp]]
command = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
prefix = "fs."     # 可选；两个 server 有同名工具时用它区分
```

两条硬规矩：

- **MCP 工具报错不炸循环。** 服务器返回 `isError` 时转成 `ToolResult(is_error=True)`，
  让模型看到原因自己换路数；只有连接断了、握手超时这种才抛。
- **工具名撞车在启动时就报错**，不会留到运行中途才 ambiguous。服务器在 run 结束时一定被关掉，
  异常路径也一样。

## 子 agent

`--delegate` 把 `delegate(task)` 交给模型。子 agent 有同样的工具、自己的 system prompt、
自己的消息列表——**父 agent 只看到最后一句结论**，中间那二十次读文件永远不会进父的上下文。

```console
$ oneharness run "看看这个仓库的超时配置合不合理" --delegate
  → delegate({"task": "读取 app.conf 和 notes.md，告诉我超时配了多少"})
  [sub-1] ⇢ 读取 app.conf 和 notes.md，告诉我超时配了多少
      [sub-1] → read_file({"path": "app.conf"})
      [sub-1] ✓ read_file: timeout=5
      [sub-1] → read_file({"path": "notes.md"})
      [sub-1] ✓ read_file: - 5 秒对真实请求太短 - 目标是 30 秒
  [sub-1] ⇠ done: 3 step(s), 660 tokens
  ✓ delegate: app.conf: timeout=5；notes.md 里写着 5 秒对真实请求太短。结论：需要调大。
[delegated 1 task(s): 580 in / 80 out tokens]
```

三件事值得说明：

- **子 agent 不能再委派。** `delegate` / `delegate_many` 在建子 agent 时就被剔除了，
  递归深度靠结构保证，不靠一个没人检查的计数器。
- **花费单独记账。** 上面那行 `[delegated ...]` 是子 agent 自己的 token 数，不并进父 agent 的
  `result.usage`。「父花了 2 分、子花了 3 毛」是个应该能看见的事实。
- **子 agent 没答案不算成功。** 它烧完 `max_steps` 还是没给结论时，父 agent 收到的是一条
  `is_error=True` 的工具结果（不是异常），可以自己决定换路数。

也可以只用一个工具，不接管 CLI：

```python
from oneharness import Agent, Delegator, Registry, create_provider, default_tools

provider = create_provider("openai")
delegator = Delegator(provider, default_tools("."), max_steps=6, max_parallel=4)
agent = Agent(provider, Registry([*default_tools("."), delegator.tool(), delegator.batch_tool()]))
print(agent.run("把 README 的错别字都找出来改成 PR").text)
print(delegator.as_dict())   # {"runs": 3, "usage": {...}, "by_label": {...}}
```

## 一次派好几个子 agent

`--delegate` 同时给出 `delegate_many(tasks)`：一个工具调用，N 个子 agent，**一次 spawn 完**。
它和在一个回合里连调三次 `delegate` 的区别不是写法好看，而是那三次调用能不能重叠——
`--parallel` 管的是同一轮的工具调用，所以 `delegate` 攒一批才会有意义。

```console
$ oneharness run "同时查这三件事，最后汇总" --shell --delegate --parallel 3
  → delegate_many({"tasks": ["app.conf 里的 timeout 是多少？", ...]})
  [sub-1] ⇢ app.conf 里的 timeout 是多少？
  [sub-2] ⇢ notes.md 里的结论是什么？
  [sub-3] ⇢ README 有没有提到 0.4？
      [sub-3] → run_shell({"command": "sleep 0.7 && cat README.md"})
      [sub-2] → run_shell({"command": "sleep 0.7 && cat notes.md"})
      [sub-1] → run_shell({"command": "sleep 0.7 && cat app.conf"})
      [sub-3] ✓ run_shell: exit_code=0 # demo  版本 0.3.0
      [sub-1] ✓ run_shell: exit_code=0 timeout=5
      [sub-2] ✓ run_shell: exit_code=0 - 5 秒对真实请求太短
  [sub-3] ⇠ done: 2 step(s), 478 tokens
  [sub-1] ⇠ done: 2 step(s), 478 tokens
  [sub-2] ⇠ done: 2 step(s), 478 tokens
  ✓ delegate_many: [sub-1] alpha: 读完了 app.conf [sub-2] beta: ... [sub-3] gamma: ...
[delegated 3 task(s): 1260 in / 174 out tokens]
  sub-1: 420 in / 58 out
  sub-2: 420 in / 58 out
  sub-3: 420 in / 58 out
```

`--parallel 1` 跑 2.19s，`--parallel 3` 跑 0.78s（`examples/fanout_demo.py` 实测）。
三个子 agent 的输出交错在一起时，**行首的 `[sub-N]` 是唯一能分辨谁是谁的东西**，
所以每个 run 都有标签，`--json` 里也有 `delegation.by_label` 的单独账目。

三条规矩：

- **返回顺序 = 请求顺序。** 谁先跑完谁先说会让父模型的上下文每次都不一样，这种噪声没有价值。
- **一个子 agent 失败不沉掉整批。** 那一行写 `[sub-2] failed: ...`，其余答案照常回来。
  父模型能看见谁没成，自己决定要不要补。
- **标签是并发的报警器。** 上一版为了并行委派临时加了锁才发现：`usage = usage + x`
  是读-改-写，40 个并发 run 会静默丢 1000 个 input token。记账加锁不是洁癖，是它真的会掉。

## 并发工具

默认串行——工具没被假定线程安全。要并发就显式给数：

```console
$ oneharness run "把三条命令都跑一遍" --shell --parallel 3
  → run_shell({"command": "sleep 0.9 && echo alpha"})
  → run_shell({"command": "sleep 0.5 && echo beta"})
  → run_shell({"command": "sleep 0.2 && echo gamma"})
  ✓ run_shell: exit_code=0 gamma
  ✓ run_shell: exit_code=0 beta
  ✓ run_shell: exit_code=0 alpha
```

三条刻意错开的 sleep：顺序跑 1.61s，`--parallel 3` 跑 0.91s，而且完成顺序反过来——
这才是「真的同时跑了」而不是「排着队跑完了」的证据（`examples/delegate_demo.py` 实测）。

一个刻意的不对称：**事件按真实完成顺序发出，回灌给模型的结果按模型请求的顺序排列**。
前者是给看日志的人，后者是给模型的可预测性。单个工具失败仍然只影响自己那一格，
不影响同批的其他工具，也不炸循环。

## Bedrock

Bedrock 是三种线格式之外的第一种「麻烦不在线格式」的 provider。两件事跟别人都不一样：

**每个请求都要签名，而签名覆盖的是真正发出去的字节。** 用 `payload` 重新序列化一遍去签、
发的时候再序列化一遍，两份 JSON 只要差一个空格就是 `SignatureDoesNotMatch`，
而 AWS 不会告诉你是哪一字节错了。所以签名的输入是 `transport` 手里那份 `bytes`，
并且每次重试都重签——签名里带着时间戳，AWS 只接受时间窗内的它；
把第一次失败的签名原样重放，第二次失败的原因就跟这次网络抖动无关了。

**流不是 SSE，是二进制帧。** `application/vnd.amazon.eventstream` 每帧自带长度和两个
CRC32（一个盖住 prelude，一个盖住整帧）。所以它没法按行切，而且必须先校验「长度这个数字
本身」再去读那么多字节——prelude 的 CRC 就是为这件事存在的。

签名和帧编解码各自是一个模块（`sigv4.py` / `eventstream.py`），都能脱离 Bedrock 单独测：

- `sigv4.py` 对 AWS 官方文档里的测试向量逐字节吻合：S3 的 GET object、GET bucket lifecycle，
  通用服务的空 POST、query 参数排序，以及签名密钥的推导
- Bedrock 的假服务器**从它真正收到的字节重算签名**再比对——这是不接真 AWS 能做的最强检查：
  签的和发的只要不是同一份序列化结果，这个测试就红
- `eventstream.py` 用手工按布局拼的帧校准（不是用被测的那个编码器生成的），写的时候还用
  `botocore` 的 `EventStreamBuffer` 交叉解过一遍——自己编码自己解码不算验证，
  同一个误解会同时骗过两遍。`botocore` 没进依赖，也没进测试：只用来校对一次那串测试字节

```console
$ export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-west-2
$ oneharness run "看看 app.conf" --provider bedrock --model anthropic.claude-3-5-sonnet-20241022-v2:0
$ oneharness run "看看 app.conf" --provider bedrock --stream
```

模型 id 会进 URL 路径，所以要 percent-encode：cross-region inference profile 的 id 是
`arn:aws:bedrock:...` 这种形状，冒号原样留在被签名的路径里就是签名不匹配。

凭证只认标准环境变量（`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`），
**没有**实现 profile、实例角色、SSO 那条完整链。那是一条很长的代码和一堆启动时缓慢失败的方式，
而 CI 和容器用的就是环境变量。

## Azure OpenAI

线格式是 OpenAI 的，差异全在「怎么找到它」：端点里嵌的是**部署名**而不是模型名，
`api-version` 是个必填的 query 参数，鉴权头是 `api-key` 而不是 `Bearer`。
三条都是一行代码的差别，搞错了就是一个 404 或 401。

```console
$ export AZURE_OPENAI_API_KEY=... AZURE_OPENAI_ENDPOINT=https://my-res.openai.azure.com
$ oneharness run "ping" --provider azure --model my-deployment
```

`--model` 传的是部署名，`--provider azure` 的 `base_url` 省略时读 `AZURE_OPENAI_ENDPOINT`。
两个都缺的时候报错会直接告诉你缺哪个，而不是在发请求时报 400。

## doctor：把失败归到正确的层

配置出问题时，报错都是一个红字，但「key 错了」「模型名错了」「URL 错了」「我们发错了」
是完全不同的四件事，处理方法相反。`doctor` 发一个最小的真请求，然后读 provider 的原文判断：

```console
$ oneharness doctor openai anthropic gemini
✗ openai     auth         Incorrect API key provided: sk-inva*****c123. ...
             the request shape was accepted and the key was not: rotate the key
✗ anthropic  auth         invalid x-api-key
             the request shape was accepted and the key was not: rotate the key
- deepseek   unconfigured no API key for provider 'deepseek'; set DEEPSEEK_API_KEY ...
             not configured on this machine; set the key to test it
```

七个判定：

| 判定 | 含义 | 该动哪里 |
|---|---|---|
| `ok` | 真答回来了 | —— |
| `auth` | 请求形状被接受，凭证被拒 | 换 key |
| `model` | key 有效，模型/部署名不对 | 改 model |
| `endpoint` | 那个 URL 没人应答 | 改 base_url / region |
| `quota` | 限流或余额 | 等，或换 key |
| `protocol` | provider 说**我们的字段**不对 | OneHarness 的 bug |
| `context` | 请求被看懂了，只是超过窗口 | 设 `context_budget`，或发少一点 |
| `network` | 根本没答：DNS / TLS / 代理 / 超时 | 查网络 |
| `unconfigured` | 本机没配这个 key | 配了再测 |

关键是 `auth` 和 `protocol` 不能混。401 说「key 无效」证明请求本身没问题；
400 说「unknown name messages」意味着这个 harness 发错了体，换对 key 也修不好。
两者在日志里一样红，含义正好相反。

`context` 是同一类区分的第二个例子，而且它和 `protocol` 撞在同一个 400 上：
「prompt is too long: 208310 tokens > 200000 maximum」说明请求被理解了、只是装不下；
去 provider 适配层找 bug 是白费一天。

`unconfigured` 不算失败，退出码仍是 0——一台机器通常只有一两个 key，缺的那些不该把 CI 染红。

```console
$ oneharness doctor --all --json     # 脚本里用
```

## 上下文预算

工具循环只会变长。每步 append 一条带调用的 assistant 消息和一条带结果的 tool 消息，
什么都不出——于是第 2 步刚好塞下的 run，第 9 步就超了窗口，而 provider 回的是一个 400，
读起来像「请求格式不对」而不是「预算超了」。`doctor` 以前会把它判成 `protocol`，
也就是「这是 OneHarness 的 bug」，把人送进错误的文件里。

```console
$ oneharness run "看看这份 40KB 的日志" --context-budget 1200
  → read_file({"path": "big.log"})
  ✓ read_file: 2026-11-12 INFO request done ...
  ~ context 9438→36 tokens (budget 1200, elided 1, dropped 0)
big.log 一共 40000 字节，比这个窗口大。上一次的结果被裁掉了，我手上只剩占位符。
```

两条规矩：

- **裁的是请求，不是记录。** 只有发到线上的那份列表会被裁，`AgentResult.messages`
  一条不少——一份既被截断又被当成记录的历史，比没有记录更糟。
- **先丢最便宜的东西。** 工具结果可以再调一次拿回来，决定调用它的那段推理不行。
  所以结果先被抹成占位符，占位符里写着发生了什么，模型能自己决定要不要重调，
  而不是以为文件是空的。

三个动作，按发生顺序：

1. **发之前裁。** 老的工具结果先被 elide；还不够就从最早的整轮开始丢，但**永远不拆开
   调用和它的结果**（留下一条孤儿 tool 消息，OpenAI 会直接拒绝），也不丢 system、
   第一条 user 和最新的那条 user（前两个是「我是谁 / 任务是什么」，最后一个既是任务，
   也保证裁剪后第一个非 system 消息仍是 user——Anthropic 和 Bedrock 都要求这点）。
2. **被拒了降预算重试。** 400/413 的原文里提到 context / too long，就是尺寸问题而不是
   格式问题。这时候用「刚发出去的那份有多大」当测量值，按 0.6 缩一轮再发。
3. **没得裁就停。** 如果缩完预算一条消息都删不掉（比如 prompt 本身就超窗口），
   再发一次只会拿到同一个 400。于是直接报出 provider 的原话并说明原因——为了塞进窗口
   而丢掉用户的问题，那是在回答另一个问题。

估算刻意做得很粗：字符数除以 4。它不是 tokenizer，但和 provider 无关、不会随某个
包升级而漂移，而且对代码和 JSON（也就是工具循环的主要成分）是偏高而不是偏低。
`--json` 里有 `trim` 字段，只在真的裁过时才出现：

```json
"trim": {"budget": 1200, "tokens_before": 9438, "tokens_after": 36, "elided": 1, "dropped": 0}
```

`context_budget` 也能写进配置（`context_budget = 32000` 或
`ONEHARNESS_CONTEXT_BUDGET=32000`），默认 `0` = 不设上限——之前能跑通的 run
行为完全不变。`--delegate` 的子 agent 继承同一个上限，而不是多一个会和它对不上的旋钮。

`doctor` 也多了一个判定 `context`，和 `protocol` 共用同一个 400：一个说「请求被看懂了，
只是太大」，一个说「请求本身不合法」。同样是红的，修的地方相反。

## tape：录一次真流量，之后离线重放

`doctor` 能问真 provider，但测试不能每次都联网。`--record` 把一次运行的所有请求连同
**原始响应字节**存进一个 JSON，`--replay` 之后原样喂回同一段解码代码：

```console
$ oneharness run "分析下 app.conf" --record session.json
$ oneharness run "分析下 app.conf" --replay session.json    # 无 key、无网络
```

两条刻意的规矩：

**存字节，不存解析结果。** 回放走的是同一个 SSE 行解析器、同一个二进制帧切分器、
同一个 decoder——那才是可能写错的地方。存解析后的对象只能拿 decoder 验证它自己。

**严格匹配。** 回放的请求必须和录下的逐字段一致，且按顺序消费。宽松匹配（同 URL 就用上一条）
会把每一次线格式回归都变成绿灯，而红灯正是 fixture 存在的理由。不匹配时报出**哪个字段**变了：

```console
$ oneharness run "换个 prompt" --replay session.json
oneharness: tape session.json exchange #1: request differs at request.messages[0].content
(recorded 'ping', sent 'a different prompt'). The wire format changed; re-record the tape
if the change is intentional.
```

header 一律不录。所以 tape 里不会有 API key，回放也不需要任何凭证——这就是
`tests/fixtures/tapes/` 里那批文件能在 CI 跑的原因。它们不是假服务器的输出，
是 OpenAI / Anthropic / Gemini / DeepSeek / Groq / Bedrock **真实回过的字节**。

## 时间花在哪：`--timing`

一个 run 结束只说「跑了 4 秒」。这 4 秒里可能全在等模型，也可能全在等一条 shell——
两者的修法完全相反：前者要缩上下文或换模型，后者要收窄工具或加并发。`--timing` 把每一步拆开：

```console
$ oneharness run "跑一遍那三条命令" --timing --parallel 3
  → run_shell({"command": "sleep 0.3 && echo hi"})
  → run_shell({"command": "sleep 0.3 && echo hi"})
  → run_shell({"command": "sleep 0.3 && echo hi"})
  ✓ run_shell: exit_code=0 hi
  ✓ run_shell: exit_code=0 hi
  ✓ run_shell: exit_code=0 hi
[timing] 2 step(s), 3 tool call(s), 307ms total
     1     306ms  model      2ms  tools    303ms  slowest call run_shell 302ms
     2       1ms  model      1ms  tools      0ms
     model 3ms (1%)  tools 303ms (99%)  other 0ms (0%)
     calls summed 906ms > wall clock 303ms: they overlapped, and the gap is what --parallel bought
     slowest step 1 at 306ms
```

五件事值得说明：

- **两半分开测，总量照旧。** `model_seconds` 围着 provider 请求测，`tool_seconds` 围着工具
  那一段测，`seconds` 仍是诚实的墙钟——它不等于两者之和，差额是循环自己做的事
  （拼消息、读历史、写终端）。**差额不会消失，也不需要消失。**
- **单个调用的耗时是并发的证据。** 三个 0.3s 的工具并发跑，墙钟 303ms，而三个调用加起来
  906ms。这两个数不等的时候，`--parallel` 才真的买到了东西；相等的时候它什么也没买到。
- **首 token 时间只对流式有意义。** 一次阻塞请求没有「第一个」可测，所以这个字段在非流式
  下**不存在**，而不是填 0——填 0 会被读成「首 token 极快」。
- **被拒之后的重试也算进这一步。** 一个 400 白跑一趟也是真的等了，把它从统计里抹掉，
  统计就会掩盖问题。
- **委派的时间单独说。** `delegate` 本身是工具调用，所以它的等待落在 `tools` 里；但它等的
  是另一个 agent 在思考。摘要会另起一行说「这 302ms 是子 agent」，否则每次 fan-out 都像是
  工具慢——一个笃定的错误结论比没有结论更糟。
- **重试是可见的。** `model_requests` 比 `steps` 多，说明有请求因为超过窗口被拒、降预算又
  发了一次。一次白跑一趟的 400 也是真的等了，它不该藏在「模型花了 3 秒」里。

`--json` 里是同一个东西，结构稳定：

```json
"timing": {
  "total_seconds": 0.207,
  "model_seconds": 0.004,
  "tool_seconds": 0.203,
  "tool_call_seconds": 0.404,
  "unaccounted_seconds": 0.0,
  "first_token_seconds": null,
  "steps": 2,
  "model_requests": 2,
  "tool_calls": 2,
  "slowest_step": {"step": 1, "seconds": 0.204}
}
```

每个 `steps[]` 也带 `model_seconds` / `tool_seconds`，每个 `tool_results[]` 带自己的
`seconds`。`--timing` 与 `--json` 同时给时以 JSON 为准，不打那段摘要。

摘要走 stderr，答案走 stdout。一段诊断信息不该污染管道里的正文——
`--json` 和 `--stream` 下这条规矩本来就在，`--timing` 只是又多了一个乘客。

## 配置

就近找 `oneharness.toml`（从当前目录逐级向上，最后看 `~/.config/`）：

```toml
[provider]
preset = "deepseek"          # 完整列表见 `oneharness presets`
model = "deepseek-chat"
temperature = 0
max_tokens = 4096

[agent]
system = "你是简洁的工程师，先看代码再改。"
max_steps = 8
max_parallel = 3             # 同一轮里的工具并发数，默认 1
context_budget = 32000       # 单个请求的 token 上限；0（默认）= 不设上限

[delegate]                   # 写了这张表就等于打开 --delegate
max_steps = 6
max_parallel = 4             # 同时跑几个子 agent；省略则跟随 [agent] max_parallel
# system = "只回答结果，不要复述任务。"
```

环境变量优先于文件：`ONEHARNESS_PROVIDER`、`ONEHARNESS_MODEL`、`ONEHARNESS_API_KEY`、
`ONEHARNESS_BASE_URL`、`ONEHARNESS_SYSTEM`、`ONEHARNESS_MAX_STEPS`、`ONEHARNESS_MAX_PARALLEL`、
`ONEHARNESS_CONTEXT_BUDGET`。

API Key 走各 provider 的常规变量名：`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、
`GEMINI_API_KEY`（或 `GOOGLE_API_KEY`）、`DEEPSEEK_API_KEY`、`MOONSHOT_API_KEY`、
`DASHSCOPE_API_KEY`、`ZHIPU_API_KEY`、`GROQ_API_KEY`、`MISTRAL_API_KEY`、
`OPENROUTER_API_KEY`、`AZURE_OPENAI_API_KEY`；`ollama` 不需要。
`bedrock` 用 `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION`。

`region` 和 `api_version` 也能写在 `[provider]` 里，跟着模型走而不是散在环境变量里：

```toml
[provider]
preset = "bedrock"
model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
region = "eu-central-1"
```

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

**裁的是请求，不是记录。** 上下文预算只在「发到线上」的那份列表上生效，
`AgentResult.messages` 仍是一份完整的、可以拿去做下一轮 `history` 的记录。
一份既被截断又被当成记录的历史，比没有记录更糟。

**没有内置重试模型决策。** provider 层的 429/5xx 会自动退避重试（`transport.py`），
但「模型答得不对要不要重问」属于业务判断，交给调用方。

**流式靠 `read1`，不靠 `read`。** `HTTPResponse.read(4096)` 会一直阻塞到攒满 4096 字节或
读到 EOF，于是整个流在最后一次性交出来——它不报错，只是把流式悄悄变成批处理。
`read1` 才是「有多少给多少」。SSE 路径没暴露过这个问题，因为按行迭代本来就是逐行阻塞的；
二进制帧路径把它暴露了。

**测试的假服务器是循环论证，得用真字节破它。** 假服务器是我按自己读的规范写的：
我读错了，它也错，两边一起绿。所以 `tests/fixtures/tapes/` 存的是真端点的响应字节，
`doctor` 对着它们跑。判成 `auth` 是有信息量的——它证明请求过了字段关、只卡在凭证上，
这是任何自写假服务器都拿不到的结论。同一条「故意发错的 OpenAI 请求体给 Gemini」
则拿到 `protocol`，证明这个判定真的会红。

**耗时是量出来的，不是估出来的。** 每一步拆成「等模型」和「等工具」两段，各有各的修法；
差额单独列出来而不是摊进任何一段，因为「循环自己花掉的时间」也是一条信息。
一个各部分永远严格相加的分解，等于在替没测到的部分编数字。

**没有 asyncio。** 流式不需要 event loop——SSE 是按行迭代的生成器。并发只用在唯一真正值得
并发的地方：同一轮里的多个工具调用，用线程池，且要显式 `--parallel N` 才开。provider 请求
永远串行，因为一个 agent 的下一步依赖上一步。

**流式不改变结果形状。** token 通过 `on_event` 走旁路，agent 循环仍然只处理「已完成的一轮」。
工具调用的增量也只是事件，不参与决策：`done` 里的那份完整参数仍是唯一被执行的版本。
这样行为只有一套，`--json` 也照旧。

**截断和中断不是失败，但要如实说。** 流断了就返回 `stopped_reason="interrupted"`，退出码 1；
Ctrl-C 退出码 130。把半句话当成完整答案是一种撒谎，把已经到达的半句话丢掉则是另一种。

## 项目结构

```
oneharness/
  types.py        Message / ToolCall / Completion / Usage / ToolResult / StreamEvent
  sigv4.py        AWS Signature V4：一个纯函数，两处签名（请求 + 每次重试）
  eventstream.py  vnd.amazon.eventstream 二进制帧编解码，两个 CRC 都校验
  transport.py    urllib 封装：JSON POST + SSE + 裸字节流，带 429/5xx 退避重试
  providers/      base.py + openai.py + anthropic.py + gemini.py + bedrock.py + azure.py
  tools.py        @tool 装饰器、schema 生成、Registry
  agent.py        工具调用循环，on_event 可观测，stream=True 分流 token，工具可并发，
                  每步拆出 model / tool 两段耗时与首 token 时间
  delegate.py     子 agent：独立上下文、独立记账、标签、批量 fan-out、不能再委派
  builtin.py      文件/目录/shell 工具
  mcp.py          stdio JSON-RPC 的 MCP 客户端，把远端工具变成普通 Tool
  context.py      上下文预算：估算、先 elide 工具结果、再丢整轮、溢出识别与降预算
  tape.py         录/放真流量：存原始字节、严格匹配请求、header 永不落盘
  doctor.py       一个最小真请求 + 把失败归到 auth/model/protocol/network...
  config.py       TOML + 环境变量 + [[mcp]] + [delegate]
  cli.py          run / chat / tools / presets / doctor / config，含 --timing 摘要
tests/            309 个单元测试，含本地 HTTP mock、真实 MCP 子进程、真端点录制的 tape
examples/         十个离线 demo：e2e / stream / mcp / delegate / fanout / bedrock / doctor /
                  context / stream_tools / timing，都不需要 API Key
```

## 开发

```console
$ python -m unittest discover -s tests -v    # 全部测试
$ python examples/e2e_demo.py                # 端到端，起本地假 provider
$ python examples/stream_demo.py             # 流式 + 跨 chunk 的工具调用
$ python examples/mcp_demo.py                # CLI 真拉一个 MCP 子进程
$ python examples/delegate_demo.py           # 串行 vs 并发、子 agent 委派
$ python examples/fanout_demo.py             # 一次派三个子 agent，逐个 vs 同时
$ python examples/bedrock_demo.py            # 签名回验 + 二进制帧流，本地假 Bedrock
$ python examples/doctor_demo.py             # 真端点录下的 tape 上跑 doctor + record/replay 闭环
$ python examples/context_demo.py            # 40KB 工具结果：先裁 / 被拒后重试 / 没得裁就停
$ python examples/stream_tools_demo.py       # 工具调用在参数写完前露面；流断了半句话还在
$ python examples/timing_demo.py             # 同样的墙钟、不同的元凶；并发把调用之和与墙钟拉开
$ ruff check oneharness tests examples       # 风格（不格式化 Markdown 里的示例）
```

## License

MIT
