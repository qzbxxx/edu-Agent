# tests/test_eval_metrics.py
# 评测脚本自身的单元测试：防止“禁止词”误判，也防止标注集被改坏。
import os

from scripts.eval_rag import contains_banned, is_refusal_answer, load_cases

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARD_SET = os.path.join(BASE_DIR, "data", "eval_set_hard.jsonl")


def test_banned_phrase_ignores_negation():
    # 拒答里的“无法保证就业”不是承诺，不能算幻觉
    assert not contains_banned("我们无法保证就业，课程不承诺包分配", "保证就业")
    assert not contains_banned("不是当天到账，具体以支付平台为准", "当天到账")
    # 否定词和禁止词之间隔着引号/“没有任何”时，也要能识别
    assert not contains_banned("参考文档中没有任何“保证就业”的承诺", "保证就业")


def test_banned_phrase_catches_positive_claim():
    assert contains_banned("报名后我们保证就业", "保证就业")
    assert contains_banned("退款一般 1-3 个工作日到账", "1-3个工作日")


def test_refusal_detection_accepts_uncertain_wording():
    # 模型不一定说“未提及”，也可能说“无法确认 / 未包含”，这些都应算合规拒答
    assert is_refusal_answer("参考文档中未包含发票说明，无法确认。")
    assert is_refusal_answer("文档没有关于到账时效的说明，已转接人工顾问。")
    assert not is_refusal_answer("退款会原路退回。")


def test_hard_set_is_valid():
    cases = load_cases(HARD_SET)
    assert len(cases) == 18

    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))

    for case in cases:
        assert case["expect"] in {"answer", "refuse"}
        if case["expect"] == "answer":
            assert case.get("expect_source")
            assert case.get("expect_page") is not None
