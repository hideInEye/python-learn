"""纯 Python 业务规则测试，不构造模型或访问网络。"""
# 阅读提示：此文件只测业务规则，不创建模型客户端；Schema 格式校验由其他测试覆盖。
# @pytest.fixture 生成基准数据；每个测试通过 complete 参数获取一份独立对象。
# 测试会直接调整对象字段来构造场景；不要把 setattr 误认为重新执行了 Schema 校验。
# assert 比较实际结果和预期；parametrize 让一条规则覆盖多个字段或取值。

import pytest

from extraction_models import AfterSalesExtraction, Evidence
from extraction_rules import (
    AMBIGUITY_QUESTION,
    BUSINESS_FIELDS,
    MISSING_QUESTIONS,
    OTHER_QUESTION,
    REQUIRED_QUESTIONS,
    check_extraction,
    review_extraction,
)

TEXT = "订单 A10086 的耳机左边没声音，昨天收到，我想换货。"


# 准备必要字段和五条证据都完整的 Schema 对象，作为业务规则测试的基准样本。
@pytest.fixture
def complete() -> AfterSalesExtraction:
    return AfterSalesExtraction(
        order_id="A10086",
        product_name="耳机",
        issue_description="左边没声音",
        requested_action="exchange",
        received_time_text="昨天收到",
        ambiguities=[],
        # 列表推导式为每一对字段名和原文引用创建 Evidence 对象。
        evidence=[
            Evidence(field=field, quote=quote)
            # zip 同时遍历两个序列；strict=True 要求长度相同，防止漏配证据。
            for field, quote in zip(
                BUSINESS_FIELDS,
                ("A10086", "耳机", "左边没声音", "我想换货", "昨天收到"),
                strict=True,
            )
        ],
    )


# 完整样本得到 draft_ready，且检查函数不能修改输入对象或追加审批字段。
def test_complete_draft_requires_no_approval_or_extra_fields(
    complete: AfterSalesExtraction,
) -> None:
    # 保存普通字典快照，检查后再比较，验证规则没有修改原对象。
    before = complete.model_dump()
    assert check_extraction(complete, TEXT) == {
        "status": "draft_ready",
        "missing_fields": [],
        "evidence_errors": [],
        "clarification_questions": [],
    }
    assert complete.model_dump() == before


# 依次删除一个必要字段及其证据，核对缺失字段列表与对应的固定追问。
@pytest.mark.parametrize("field", ["order_id", "issue_description", "requested_action"])
def test_required_missing_fields_are_computed_and_questions_are_fixed(
    field: str,
    complete: AfterSalesExtraction,
) -> None:
    # setattr 用字符串字段名赋值，相当于动态执行 complete.order_id = None。
    setattr(complete, field, None)
    # 保留其他字段的证据，删除缺失字段的证据，避免混入无关的证据错误。
    complete.evidence = [item for item in complete.evidence if item.field != field]
    checked = check_extraction(complete, TEXT)
    assert checked == {
        "status": "needs_clarification",
        "missing_fields": [field],
        "evidence_errors": [],
        "clarification_questions": [MISSING_QUESTIONS[field]],
    }


# 空串和纯空白都应算缺失，但规则函数不会擅自改写模型原字段。
@pytest.mark.parametrize("field", ["order_id", "issue_description"])
@pytest.mark.parametrize("blank", ["", " \n\t"])
def test_blank_required_values_are_treated_as_missing(
    field: str, blank: str, complete: AfterSalesExtraction
) -> None:
    setattr(complete, field, blank)
    complete.evidence = [item for item in complete.evidence if item.field != field]
    checked = check_extraction(complete, TEXT)
    assert checked["status"] == "needs_clarification"
    assert checked["missing_fields"] == [field]
    assert checked["clarification_questions"] == [MISSING_QUESTIONS[field]]
    assert getattr(complete, field) == blank


# 三个必要字段都缺失时，追问按固定字段顺序输出，便于稳定展示和测试。
def test_all_missing_fields_have_deterministic_question_order() -> None:
    result = AfterSalesExtraction(
        order_id=None,
        product_name=None,
        issue_description=None,
        requested_action=None,
        received_time_text=None,
        ambiguities=[],
        evidence=[],
    )
    checked = check_extraction(result, "帮我处理售后。")
    assert checked["status"] == "needs_clarification"
    assert checked["missing_fields"] == [
        "order_id",
        "issue_description",
        "requested_action",
    ]
    # values() 按插入顺序取字典值，list() 将其变成可直接比较的列表。
    assert checked["clarification_questions"] == list(MISSING_QUESTIONS.values())


# 只缺商品名和收货时间时，必要信息仍然完整，不应因此追加追问。
def test_optional_fields_do_not_trigger_clarification(
    complete: AfterSalesExtraction,
) -> None:
    complete.product_name = None
    complete.received_time_text = None
    complete.evidence = [
        item
        for item in complete.evidence
        if item.field not in {"product_name", "received_time_text"}
    ]
    assert check_extraction(complete, TEXT)["status"] == "draft_ready"


# 有歧义或 other 诉求时触发固定追问，不能替用户选择一个答案。
def test_ambiguities_and_other_have_fixed_questions_without_picking_an_answer(
    complete: AfterSalesExtraction,
) -> None:
    complete.ambiguities = ["可能是 A10086，也可能是 A10087。"]
    complete.requested_action = "other"
    checked = check_extraction(complete, TEXT)
    assert checked["status"] == "needs_clarification"
    assert checked["missing_fields"] == []
    assert checked["clarification_questions"] == [AMBIGUITY_QUESTION, OTHER_QUESTION]
    assert complete.ambiguities == ["可能是 A10086，也可能是 A10087。"]


# 依次移除每个业务字段的证据，确认全部五个字段都受证据规则约束。
@pytest.mark.parametrize("field", BUSINESS_FIELDS)
def test_every_present_business_field_requires_evidence(
    field: str,
    complete: AfterSalesExtraction,
) -> None:
    complete.evidence = [item for item in complete.evidence if item.field != field]
    checked = check_extraction(complete, TEXT)
    assert checked["status"] == "evidence_error"
    # None 表示完整性检查没有运行，不能写成表示没有缺项的空列表。
    assert checked["missing_fields"] is None
    assert checked["clarification_questions"] == []
    assert checked["evidence_errors"] == [f"{field} 缺少有效的原文证据。"]


# 空白、虚构或拼接的引用应失败，且证据错误先于信息完整性检查。
@pytest.mark.parametrize("quote", [" \n\t", "不存在的原文", "耳机没声音"])
def test_empty_absent_or_non_contiguous_quotes_fail(
    quote: str,
    complete: AfterSalesExtraction,
) -> None:
    complete.evidence[1].quote = quote
    # 即使同时缺字段、有歧义、为 other，也必须先报证据错误。
    complete.order_id = None
    complete.evidence = [item for item in complete.evidence if item.field != "order_id"]
    complete.ambiguities = ["待确认"]
    complete.requested_action = "other"
    checked = check_extraction(complete, TEXT)
    assert checked["status"] == "evidence_error"
    assert checked["missing_fields"] is None
    assert checked["clarification_questions"] == []
    assert checked["evidence_errors"]


# 值为 None 或空白时，不应附带声称该字段已确定的证据。
@pytest.mark.parametrize("value", [None, "", " \t"])
def test_missing_value_must_not_claim_evidence(
    value: str | None,
    complete: AfterSalesExtraction,
) -> None:
    complete.product_name = value
    checked = check_extraction(complete, TEXT)
    assert checked["status"] == "evidence_error"
    assert "product_name 缺失" in checked["evidence_errors"][0]


# 即使引用片段存在，订单号字段本身不在原文中也必须报证据错误。
def test_order_id_itself_must_appear_in_source(complete: AfterSalesExtraction) -> None:
    complete.order_id = "A99999"
    checked = check_extraction(complete, TEXT)
    assert checked["status"] == "evidence_error"
    assert checked["evidence_errors"] == ["order_id 的值未在原文中出现。"]


# 引用可以去掉首尾空白，同一字段也允许提供多条真实的连续片段。
def test_quote_trimming_and_multiple_valid_quotes(
    complete: AfterSalesExtraction,
) -> None:
    complete.evidence[0].quote = " \tA10086\n"
    # append 新增第二条订单号证据；同一字段允许有多条有效引用。
    complete.evidence.append(Evidence(field="order_id", quote="订单 A10086"))
    assert check_extraction(complete, TEXT)["status"] == "draft_ready"


# 刻意保留一个语义错误但引用存在的结果，说明机械检查无法证明语义正确。
def test_present_quote_is_not_a_proof_of_semantic_correctness(
    complete: AfterSalesExtraction,
) -> None:
    complete.product_name = "机械检查无法判断这个商品名是否正确"
    # 引用本身真实存在，却没有证明 product_name 的语义正确。
    assert check_extraction(complete, TEXT)["status"] == "draft_ready"


# 保留原有三个验收样本：正常、缺订单号、虚构订单号；仅补注释与类型提示。
@pytest.mark.parametrize(
    "order_id,source,expected_status",
    [
        (
            "A10086",
            "订单 A10086 的耳机左边没声音，我想换货。",
            "draft_ready",
        ),
        (
            None,
            "耳机左边没声音，我想换货。",
            "needs_clarification",
        ),
        (
            "A99999",
            "耳机左边没声音，我想换货。",
            "evidence_error",
        ),
    ],
)
def test_review(order_id: str | None, source: str, expected_status: str) -> None:
    # 固定测试数据只在本地构造，不调用模型；不缺失的字段都附上原文证据。
    evidence = [
        Evidence(field="product_name", quote="耳机"),
        Evidence(field="issue_description", quote="左边没声音"),
        Evidence(field="requested_action", quote="我想换货"),
    ]

    if order_id is not None:
        evidence.append(Evidence(field="order_id", quote=order_id))

    extracted = AfterSalesExtraction(
        order_id=order_id,
        product_name="耳机",
        issue_description="左边没声音",
        requested_action="exchange",
        received_time_text=None,
        ambiguities=[],
        evidence=evidence,
    )

    result = review_extraction(source, extracted)

    # 不只比较状态，还验证缺项、证据问题和追问，防止状态正确但内容错误。
    assert result.status == expected_status

    if expected_status == "needs_clarification":
        assert result.missing_fields == ["order_id"]
        assert result.questions == ["请提供订单号。"]

    elif expected_status == "evidence_error":
        assert result.evidence_errors
        assert result.questions == []

    else:
        assert result.missing_fields == []
        assert result.evidence_errors == []
        assert result.questions == []


# 空白仍是 Schema 允许的字符串，但业务上等同缺失；不能顺手修改原值。
@pytest.mark.parametrize("field", ["order_id", "issue_description"])
@pytest.mark.parametrize("value", [None, "", " \t\n"])
def test_review_missing_required_fields_use_fixed_questions(
    field: str, value: str | None, complete: AfterSalesExtraction
) -> None:
    setattr(complete, field, value)
    complete.evidence = [item for item in complete.evidence if item.field != field]
    before = complete.model_dump(mode="json")

    result = review_extraction(TEXT, complete)

    assert result.status == "needs_clarification"
    assert result.missing_fields == [field]
    assert result.questions == [REQUIRED_QUESTIONS[field]]
    assert result.evidence_errors == []
    assert complete.model_dump(mode="json") == before


# 每次构造一种证据错误；已有三个样本未覆盖的“订单号与其引用不匹配”也要拒绝。
@pytest.mark.parametrize(
    "case",
    ["missing", "blank", "absent", "order-mismatch", "null-with-evidence"],
)
def test_review_rejects_bad_evidence_without_changing_input(
    case: str, complete: AfterSalesExtraction
) -> None:
    if case == "missing":
        complete.evidence = complete.evidence[1:]
    elif case == "blank":
        complete.evidence[0].quote = " \t"
    elif case == "absent":
        complete.evidence[0].quote = "不存在于原文"
    elif case == "order-mismatch":
        # 订单号和引用分别都在原文中，但这条引用不能证明该订单号。
        complete.evidence[0].quote = "耳机"
    else:
        complete.order_id = None
    before = complete.model_dump(mode="json")

    result = review_extraction(TEXT, complete)

    assert result.status == "evidence_error"
    assert result.evidence_errors
    assert result.questions == []  # 结果错误不能伪装成用户信息不足。
    if case == "order-mismatch":
        assert "order_id 与引用的证据不对应。" in result.evidence_errors
    assert complete.model_dump(mode="json") == before


# 歧义和 other 分别测试；追问来自固定模板，并保留原始歧义而不选择答案。
@pytest.mark.parametrize("case", ["ambiguity", "blank-ambiguity", "other"])
def test_review_clarifies_ambiguity_or_other_with_fixed_templates(
    case: str, complete: AfterSalesExtraction
) -> None:
    if case == "other":
        complete.requested_action = "other"
        expected = "请说明具体希望如何处理，我们再确认后续流程。"
    elif case == "ambiguity":
        complete.ambiguities = ["换货还是维修？"]
        expected = "请澄清：换货还是维修？"
    else:
        complete.ambiguities = [" \t"]
        expected = "请补充说明存在歧义的信息。"
    before = complete.model_dump(mode="json")

    result = review_extraction(TEXT, complete)

    assert result.status == "needs_clarification"
    assert result.missing_fields == []
    assert result.evidence_errors == []
    assert result.questions == [expected]
    assert result.ambiguities == complete.ambiguities
    assert complete.model_dump(mode="json") == before
