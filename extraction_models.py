# Literal 把取值限制为列出的几个字符串，类似 TypeScript 的字面量联合类型。
from typing import Literal

# BaseModel 是 Pydantic 数据校验基类；这里的“模型”是数据结构，不是 LLM。
# ConfigDict 配置校验规则；Field 为字段附加约束和描述。
from pydantic import BaseModel, ConfigDict, Field

# 手工维护的 Schema 版本写入运行记录；以后改变字段含义时应考虑更新版本。
SCHEMA_VERSION = "after-sales-v1"

# 证据只能对应下面五个业务字段，不能把任意字符串当作字段名。
EvidenceField = Literal[
    "order_id",
    "product_name",
    "issue_description",
    "requested_action",
    "received_time_text",
]


class Evidence(BaseModel):
    """一个字段对应的原文证据。"""

    # extra="forbid" 拒绝 Schema 外的字段；strict=True 限制隐式类型转换。
    # 这是本地 Pydantic 规则，不会开启供应商 strict/Beta 模式。
    model_config = ConfigDict(extra="forbid", strict=True)

    field: EvidenceField  # 类型别名用于限制此字段对应哪个业务属性。
    # min_length=1 拒绝空字符串；纯空白字符串仍由后续业务规则检查。
    quote: str = Field(
        min_length=1,
        description="支持该字段的用户原文连续片段，不改写。",
    )


class AfterSalesExtraction(BaseModel):
    """单个售后请求的信息抽取结果，不是审批结论。"""

    # 拒绝模型自行添加“已批准退款”等字段，并使用严格的字段类型校验。
    model_config = ConfigDict(extra="forbid", strict=True)

    # str | None 表示字符串或空值，序列化时 None 对应 JSON null。
    # 这里没有设置默认值，所以键必须存在；“允许 null”不等于“允许省略键”。
    order_id: str | None = Field(
        description="唯一明确的订单号；保留原文；缺失或无法确定时为 null。",
    )

    # 商品名允许未知；是否是业务必填项由 extraction_rules.py 单独决定。
    product_name: str | None = Field(
        description="用户明确提到的商品；没有则为 null。",
    )

    # Field 的 description 会进入自动生成的 JSON Schema，供模型理解字段。
    issue_description: str | None = Field(
        description="用户描述的问题或申请原因；没有则为 null。",
    )

    # 业务诉求用有限枚举表达；None 表示没有明确诉求，other 表示明确的其他诉求。
    requested_action: (
        Literal[
            "refund",
            "exchange",
            "repair",
            "other",
        ]
        | None
    ) = Field(
        description=(
            "退款 refund；换货 exchange；维修 repair；"
            "明确的其他诉求 other；未说明或无法确定为 null。"
        ),
    )

    # 时间保存原话，避免代码或模型擅自把相对日期转换成具体日期。
    received_time_text: str | None = Field(
        description=("原样保留收货时间说法，例如昨天收到；不推算日期；没有则为 null。"),
    )

    # list[str] 是字符串列表；没有歧义时应显式提供 []，不能遗漏这个键。
    ambiguities: list[str] = Field(
        description="尚未解决的歧义；没有则为空列表。",
    )

    # 嵌套校验：列表中的每项也必须符合上面 Evidence 的字段与约束。
    evidence: list[Evidence] = Field(
        description="为每个非 null 的业务字段提供原文证据；不能编造。",
    )
