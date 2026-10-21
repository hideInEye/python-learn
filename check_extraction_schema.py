# 这个小脚本仅验证数据结构，不加载配置，也不创建 LLM 客户端。
from pydantic import ValidationError  # 校验不通过时会抛出的异常类型。

from extraction_models import AfterSalesExtraction

# None 会变成 JSON null；所有键仍然必须存在，空列表明确表示没有条目。
payload = {
    "order_id": None,
    "product_name": None,
    "issue_description": None,
    "requested_action": None,
    "received_time_text": None,
    "ambiguities": [],
    "evidence": [],
}

# model_validate 接受 Python 字典；校验通过后返回 AfterSalesExtraction 对象。
result = AfterSalesExtraction.model_validate(payload)

print("测试一：所有字段存在、值未知，Schema 校验通过。")
print(result.model_dump_json(indent=2))  # 转成带两空格缩进的 JSON，方便肉眼检查。

# **payload 展开并复制原字典；后写的同名键覆盖旧值，不修改原 payload。
invalid_payload = {
    **payload,
    "requested_action": "delete_order",
}

try:  # 故意提供非法枚举值，确认 Schema 会拒绝它。
    AfterSalesExtraction.model_validate(invalid_payload)
except ValidationError:  # 出现预期异常，说明这项校验生效。
    print("\n测试二：非法的 requested_action 被拒绝。")
else:  # try 没有抛异常才会进入 else；这里意味着本应拒绝的数据被接受了。
    raise AssertionError("测试失败：非法枚举值没有被拒绝。")
