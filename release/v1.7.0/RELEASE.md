给工具调用型 LLM agent 用的一层薄 harness：**一套接口，接所有 provider**。纯标准库，零第三方依赖。

```console
$ pip install oneharness        # 需要 Python 3.11+
$ oneharness run "看看 examples/app.conf 里的 timeout，太小了就改大"
```

## 这一版有什么

1. **五种线格式抹平成同一个 `Completion`** —— OpenAI / Anthropic / Gemini / Bedrock / Azure，含 Bedrock 的 SigV4 签名与二进制事件流。13 个 preset（含 ollama 本地免 key）。
2. **工具循环跑得稳** —— 工具失败转成 `is_error` 结果回灌让模型自己纠正，不炸循环；`max_steps` 是硬闸。
3. **真流式** —— token 边生成边出，工具调用在参数还在到达时就可见，连接断了留住已收到的半句。
4. **并发与委派** —— 一轮内多工具并发；子 agent 上下文隔离，`delegate_many` 一次派 N 个，记账分开算。
5. **上下文预算** —— 超窗口就裁，被拒就按 provider 原话降预算重试，裁不动就停下说清楚差多少。
6. **能验证自己** —— `doctor` 用真 provider 各发一次最小请求并判断是谁的错；请求层 `--record` / `--replay` 成 tape，回放不要 key 也不要网络。
7. **说得清慢在哪** —— `--timing` 把每一步拆成「等模型」和「等工具」，含首 token 时间。
8. **不把网络故障丢给上层** —— 四种「没答」四种修法：证书被拒、代理要密码、域名解析不了、端口没人听。
9. **token 数得清** —— 按字符类别估算而不是 `chars / 4`；工具 schema 也算进去；五种缓存字段归一成一个 `Usage`。
10. **不替 shell 撒谎** —— 超时留输出、真杀进程组、混流、二进制不当异常、边读边裁。
11. **不把「服务器死了」报成「超时了」** —— MCP server 崩了立刻说崩了，带退出码与 stderr，回它发来的 ping。
12. **不把「模型话没说完」报成「答完了」** —— 五家用四个词说同一件事（`length` / `max_tokens` / `MAX_TOKENS`），读完那个词再决定这次 run 算不算成功。

## 版本线

本版把从 MVP 到 v1.7 的十四个版本一次性收拢发布：

| 版本 | 主题 |
| --- | --- |
| v0.1 | MVP：多 provider 抽象 + 工具循环 + CLI |
| v0.2 | 流式、Gemini、MCP tools |
| v0.3 | 子 agent 委派 + 并发工具 |
| v0.4 | 并行委派 fan-out + 并发记账加锁 |
| v0.5 | Bedrock（SigV4 + 二进制事件流）、Azure OpenAI |
| v0.6 | `doctor` 自检、tape 录制/回放 |
| v0.7 | 上下文预算：裁、降、停 |
| v0.8 | 流式不骗人：工具调用边写边出、断流留半句 |
| v0.9 | 耗时归因：模型 vs 工具 vs 首 token |
| v1.0 | 网络故障不再冒充崩溃 |
| v1.1 | 四种「没答」不再混成一句话 |
| v1.2 | token 估算不再靠 `chars / 4` |
| v1.3 | 缓存字段归一与 prompt 缓存断点 |
| v1.4 | shell 不再撒谎 |
| v1.5 | MCP server 死了不报成超时 |
| v1.6 | 裁剪不把请求裁成不合法的 |
| v1.7 | 撞上输出上限不报成答完了 |

## 验证情况

- **439 个单元测试全绿**（`python -m unittest discover -s tests`）
- `ruff check` + `ruff format --check` 全干净，已进 CI
- **17 个离线 demo** 全部可在无 API Key、无网络的情况下跑通，全部进 CI
- 发布前在干净工作区重跑了一遍测试与 lint

## 已知取舍

- Bedrock 凭证只认标准环境变量，未实现 profile / 实例角色 / SSO 链（写进 README 了，不是忘了）。
- MCP 只做 tools 层，resources / prompts 暂未实现。
- `run_shell` 默认关闭——它不是沙箱。
