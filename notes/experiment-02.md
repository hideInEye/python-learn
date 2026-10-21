# 实验 02：售后信息抽取输出模式对照

本文件是待填写模板，尚未记录真实实验结果。由操作者逐条手动运行，不自动执行批量 API 请求。

## 对照条件

- 实验日期 / 操作者：待填
- Python、langchain-deepseek、langchain-core、Pydantic 版本：待填
- 请求模型：从日志的 `requested_model` 填写，不在这里记录 API Key。
- Schema 版本：从 `schema_version` 填写。
- Prompt 哈希：逐次记录日志中的 `prompt_sha256`；它对应实际发送的完整系统提示词，包含自动生成的 Schema 和完整格式示例。
- 固定参数：温度默认 0.2，`max_tokens` 默认 1024，非流式、非思考模式，timeout=30，max_retries=0。以各次日志的实际参数为准；对照时保持一致。
- 每组使用相同 `case_id`、输入原文、模型与输出预算，只改变 `--mode`。保留失败记录，不挑选成功结果冒充首次结果。

| CLI 模式 | 日志 `mode` | 日志 `output_mode` | 调用与格式校验 |
| --- | --- | --- | --- |
| prompt | prompt | invoke | 普通 `model.invoke()`；用 `model_validate_json()` 校验原始可见正文 |
| json | json | json_mode | `with_structured_output(..., method="json_mode", include_raw=True)`；检查解析状态和结果，并严格校验原始 JSON 正文 |
| structured | structured | function_calling | 保留工具调用路径；检查唯一工具调用、解析状态及结果类型；正文为空也可能有效 |

`structured` 不代表供应商的 `json_schema`。三种模式使用相同业务指令、自动生成的 Schema 和格式示例；输出方式指令和供应商请求机制不同，因此实际输入 Token 数可能不同。

## 样本计划

先填写输入及人工预期，再逐条运行。以下仅是样本类别，没有填写成绩。

| case_id | 输入原文 | 计划检查点 | 人工预期的关键字段 / 应否追问 |
| --- | --- | --- | --- |
| 待填 | 待填 | 信息完整的单个请求 | 待填 |
| 待填 | 待填 | 缺少订单号、问题或诉求 | 待填 |
| 待填 | 待填 | 存在歧义或明确的其他诉求 | 待填 |
| 待填 | 待填 | 相对时间与原文证据 | 待填 |

## 逐次运行记录

每次运行一行。同一输入的三种模式使用相同 case_id，填写各自的 run_id。

| case_id | 输入 | 模式 | run_id | 格式校验结果 | 关键字段是否正确 | 是否正确追问 | 证据问题 | 耗时 latency_ms | Token：input / output / total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 待填 | 待填 | prompt | 待运行 | 待记录 | 待核对 | 待核对 | 待核对 | 待记录 | 待记录 |
| 待填 | 待填 | json | 待运行 | 待记录 | 待核对 | 待核对 | 待核对 | 待记录 | 待记录 |
| 待填 | 待填 | structured | 待运行 | 待记录 | 待核对 | 待核对 | 待核对 | 待记录 | 待记录 |

从 `data/runs/extraction.jsonl` 按 run_id 查找对应记录。不要把 pytest 的模拟响应填进真实模型实验表。

- 格式校验结果：通过 / 失败 / 未执行。请求失败时填“未执行”；`validation_error` 表示校验失败。`evidence_error` 或 `needs_clarification` 不代表 Schema 格式失败。
- 关键字段是否正确：人工对照输入判断，可写“是 / 否 / 部分 / 不适用”，并注明错在哪里。合法 JSON 不保证语义正确。
- 是否正确追问：根据输入是否缺必要信息、有歧义或诉求为 other 判断；比较是否正确识别问题，不要求模型措辞逐字一致。
- 证据问题：记录空白引用、原文找不到、缺少字段对应证据、缺失字段却附带证据、订单号未出现在原文等情况。“原文中存在证据”只是机械检查，不等于已经证明语义正确。
- 耗时：复制日志的 `latency_ms`，单位毫秒；统计包含输入处理、一次模型调用和校验，不包含 JSONL 落盘时间。
- Token：复制实际返回用量。未知项照填 `null`，不得按文本长度计算，也不能把未知值改成 0。
- 实际模型标识：只能记录日志中的 `returned_model`；`null` 表示无法确认，不用请求模型名补齐。
- `finish_reason=length` 需注明。格式能否通过仍按实际结果判断；若截断导致 JSON 不完整则校验失败，不自动补全。
- 证据失败时 `checks.missing_fields=null` 表示完整性检查未执行。`draft_ready` 仅表示草稿可供复核，不是退款批准。

## 单次运行补充信息

需要分析某次运行时复制此表，仅填写已验证的信息，不复制请求头、完整异常、环境变量或额外推理内容。

| 字段 | 值 |
| --- | --- |
| run_id | 待填 |
| 最终 status | 待填 |
| requested_model / returned_model | 待填 |
| schema_version / prompt_sha256 | 待填 |
| 实际 parameters | 待填 |
| finish_reason | 待填 |
| 脱敏后的 error_type / error_message | 待填 |
| 人工发现的问题及备注 | 待填 |

## 手动运行

在项目根目录执行，下面每条命令都会单独请求真实模型；按需逐条运行。

```bash
python extract.py --mode prompt --max-tokens 1024 --text "订单 A10086 的耳机左边没声音，昨天收到，我想换货。"
python extract.py --mode json --max-tokens 1024 --text "订单 A10086 的耳机左边没声音，昨天收到，我想换货。"
python extract.py --mode structured --max-tokens 1024 --text "订单 A10086 的耳机左边没声音，昨天收到，我想换货。"
```

当前工作区使用 pyenv 的 Python；若已建立 `.venv` 并安装依赖，可把 `python` 替换为 `.venv/bin/python`。

## 结论

待完成实际运行和人工核对后填写。不要根据离线测试通过率推断真实模型的正确率。
