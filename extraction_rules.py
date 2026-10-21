"""确定性的售后检查和固定追问，不调用模型，也不作审批。"""

# TypedDict 描述字典的预期键和类型，类似 TypeScript 对象类型；自身不做运行时校验。
from typing import Literal, TypedDict

from extraction_models import AfterSalesExtraction

# 元组保存全部业务字段；它们有值时都需要原文证据。
BUSINESS_FIELDS = (
    "order_id",
    "product_name",
    "issue_description",
    "requested_action",
    "received_time_text",
)
# 必要信息只有这三个字段；商品名、收货时间缺失不会单独触发追问。
REQUIRED_FIELDS = ("order_id", "issue_description", "requested_action")
# 用字段名查固定模板；追问由代码生成，不再请求一个模型来生成问题。
MISSING_QUESTIONS = {
    "order_id": "请提供需要处理的订单号。",
    "issue_description": "请描述商品出现的问题或申请售后的原因。",
    "requested_action": "请说明你的售后诉求：退款、换货、维修，或其他具体处理方式。",
}
# 有歧义时要求确认；不把模型猜测的某个答案自动当成用户选择。
AMBIGUITY_QUESTION = (
    "描述中存在尚未解决的歧义，请逐项确认，并明确本次要处理的唯一售后请求；"
    "不会自动替你选择答案。"
)
# other 的固定追问。
OTHER_QUESTION = "你希望采用其他处理方式，请具体说明需要怎样处理。"


class CheckResult(TypedDict):
    # 业务检查结论只允许下列三个字符串。
    status: Literal["evidence_error", "needs_clarification", "draft_ready"]
    evidence_errors: list[str]  # 具体证据问题；没有问题时为空列表。
    # None 表示完整性检查没运行；[] 表示运行后没有缺项。
    missing_fields: list[str] | None
    clarification_questions: list[str]  # 需要展示给用户的固定追问。


def is_missing(value: str | None) -> bool:
    # or 会短路：value 是 None 时不会再调用它的 strip()，因此不会空值报错。
    # strip() 去掉首尾空白；not "" 为 True，所以纯空白也算缺失。
    return value is None or not value.strip()


def check_evidence(result: AfterSalesExtraction, text: str) -> list[str]:
    """“原文中存在证据”只是机械检查，不等于已经证明语义正确。"""
    errors: list[str] = []  # 收集所有证据问题，便于一次性展示。
    supported_fields: set[str] = set()  # set 是去重集合：一个字段多条证据也只记一次。
    for item in result.evidence:
        quote = item.quote.strip()  # 引用首尾空白不算有效内容。
        if not quote:
            # append 往列表末尾追加一条可读错误说明。
            errors.append(f"{item.field} 的证据引用不能为空或仅包含空白。")
        elif quote not in text:  # 字符串的 in 检查连续子串，不会做语义相似度判断。
            errors.append(f"{item.field} 的证据引用不是原文中的连续片段。")
        else:
            supported_fields.add(item.field)  # add 往集合放入字段名，重复项自动去重。
        # 纯空白业务值与 null 一样按缺失处理，不改写模型的原始字段。
        # getattr(obj, "order_id") 相当于 obj.order_id，允许用变量动态选择属性。
        if is_missing(getattr(result, item.field)):
            errors.append(f"{item.field} 缺失，不应附带声称已确定该值的证据。")

    # 即使每条现有证据都合法，还要检查有没有完全漏掉某个有值字段的证据。
    for field in BUSINESS_FIELDS:
        if not is_missing(getattr(result, field)) and field not in supported_fields:
            errors.append(f"{field} 缺少有效的原文证据。")
    # 订单号额外核对“字段值本身”是否在原文中，而不只看引用片段是否存在。
    if not is_missing(result.order_id) and result.order_id not in text:
        errors.append("order_id 的值未在原文中出现。")
    return errors  # 空列表表示这些机械检查通过，不代表已理解并验证全部语义。


def check_extraction(result: AfterSalesExtraction, text: str) -> CheckResult:
    """先证据后完整性；draft_ready 仅表示草稿可供人工复核，不是退款批准。"""
    errors = check_evidence(result, text)  # 固定顺序：先证据，再完整性。
    if errors:  # 非空列表为真，说明至少有一个证据问题。
        # 提前 return：证据错误时不会继续计算缺失字段或生成追问。
        return {
            "status": "evidence_error",
            "evidence_errors": errors,
            # null 表示未执行完整性检查，避免 [] 被误解为信息完整。
            "missing_fields": None,
            "clarification_questions": [],
        }

    # 列表推导式：遍历必要字段，只收集满足 is_missing 的项，类似 JS filter。
    missing = [field for field in REQUIRED_FIELDS if is_missing(getattr(result, field))]
    # 另一种推导式：把每个缺失字段映射成固定问题，类似 JS map。
    questions = [MISSING_QUESTIONS[field] for field in missing]
    if result.ambiguities:  # 非空歧义列表也需要追问，即使必要字段都有值。
        questions.append(AMBIGUITY_QUESTION)
    if result.requested_action == "other":  # other 不能直接进入 draft_ready。
        questions.append(OTHER_QUESTION)
    # questions 非空就需要澄清；没有问题时仅标为可供人工复核的草稿。
    return {
        "status": "needs_clarification" if questions else "draft_ready",
        "evidence_errors": [],
        "missing_fields": missing,
        "clarification_questions": questions,
    }
