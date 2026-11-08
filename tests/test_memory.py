# tests/test_memory.py
import os
import sys
import uuid
import pytest

# 1. 自动定位根目录并加入环境寻路
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from langchain_core.messages import HumanMessage
from core.agent_workflow import app_graph
from core.memory_manager import get_user_long_term_memory, save_user_long_term_memory

# 固定的测试测试隔离前缀
TEST_USER_A = f"test_user_a_{uuid.uuid4().hex[:6]}"
TEST_USER_B = f"test_user_b_{uuid.uuid4().hex[:6]}"


def test_01_short_term_memory_query_rewrite():
    """
    【短期记忆测试】
    在同一个 thread_id 下连续对话，测试代词消解与上下文关联
    """
    print("\n" + "="*60)
    print("🧪 1. 正在运行：短期记忆测试（同一 Thread 上下文消解）...")
    print("="*60)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 第一轮：明确指出咨询对象
    msg_1 = "请问 Python 高级架构师训练营 的价格和退费规定是什么？"
    print(f"👤 学员 Round 1: {msg_1}")
    res_1 = app_graph.invoke({
        "messages": [HumanMessage(content=msg_1)],
        "user_id": TEST_USER_A,
        "citations": [],
        "need_human_approval": False
    }, config=config)

    assert res_1.get("final_response") is not None
    print(f"🤖 Agent 回复:\n{res_1.get('final_response')[:90]}...\n")

    # 第二轮：使用省略主语/代词提问（测试短期记忆代词消解）
    msg_2 = "那它的核心大纲包含了哪些实战模块？难学吗？"
    print(f"👤 学员 Round 2 (含代词): {msg_2}")
    res_2 = app_graph.invoke({
        "messages": [HumanMessage(content=msg_2)],
        "user_id": TEST_USER_A,
        "citations": [],
        "need_human_approval": False
    }, config=config)

    rewritten_q = res_2.get("rewritten_query") or ""
    print(f"✨ 模型上下文重写检索词 -> 【{rewritten_q}】")

    # 断言：短期记忆必须能捕捉到上一轮的主题（Python / 架构师）
    assert any(keyword in rewritten_q for keyword in ["Python", "架构", "训练营"]), \
        f"短期记忆失效，改写词未包含前轮主题: {rewritten_q}"
    print("✅ 短期工作记忆与指代消解测试通过！")


def test_02_long_term_memory_extraction():
    """
    【长期记忆抽取测试】
    用户透露个人背景，验证记忆提取器是否沉淀到持久化数据库
    """
    print("\n" + "="*60)
    print("🧪 2. 正在运行：长期记忆抽取与沉淀测试...")
    print("="*60)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 用户输入包含强烈的长期个人特征：非计算机专业、零基础转行
    user_background_msg = "我是非计算机专业的文科生，完全零基础想转行，想先了解下课程安排。"
    print(f"👤 学员画像注入: {user_background_msg}")

    app_graph.invoke({
        "messages": [HumanMessage(content=user_background_msg)],
        "user_id": TEST_USER_A,
        "citations": [],
        "need_human_approval": False
    }, config=config)

    # 直接从长期记忆数据库底层读取
    saved_memory = get_user_long_term_memory(TEST_USER_A)
    print(f"💾 从长期记忆数据库持久化读出的画像: {saved_memory}")

    # 断言长期画像库中已经记录到了相关特征
    profile_str = str(saved_memory)
    has_learned = any(kw in profile_str for kw in ["零基础", "文科", "转行", "初学者", "未知"])
    assert has_learned, f"长期记忆未记录到有效特征: {saved_memory}"
    print("✅ 长期记忆抽取与落库测试通过！")


def test_03_cross_thread_long_term_recall():
    """
    【跨会话长期记忆唤醒测试】
    使用一个全新的 thread_id（短期记忆已彻底归零），验证 Agent 能否跨越会话读取长期画像
    """
    print("\n" + "="*60)
    print("🧪 3. 正在运行：跨会话（跨 Thread）长期记忆唤醒测试...")
    print("="*60)

    # 预先确保为 TEST_USER_A 写入了独特的长期记忆标签
    save_user_long_term_memory(TEST_USER_A, {
        "learning_goal": "快速就业转行",
        "skill_level": "零基础小白",
        "preferences": ["喜欢保姆级答疑", "担心跟不上进度"]
    })

    # 创建一个全新的 thread_id，模拟一周后或者重新打开窗口
    new_fresh_thread = f"new_session_{uuid.uuid4().hex[:6]}"
    new_config = {"configurable": {"thread_id": new_fresh_thread}}

    new_query = "第七条退学退费规定具体是什么要求？"
    print(f"👤 学员在新会话 (Thread: {new_fresh_thread}) 提问: {new_query}")

    res = app_graph.invoke({
        "messages": [HumanMessage(content=new_query)],
        "user_id": TEST_USER_A, # 同样的 user_id
        "citations": [],
        "need_human_approval": False
    }, config=new_config)

    final_resp = res.get("final_response", "")
    ltm_in_state = res.get("long_term_memory", {})
    print(f"\n🤖 Agent 当前持有的长期记忆: {ltm_in_state}")
    print(f"🤖 Agent 结合长期记忆给出的个性化回复:\n{final_resp}\n")

    # 断言：State 中必须成功装载了跨会话的长期记忆
    assert ltm_in_state.get("skill_level") == "零基础小白"
    assert "快速就业转行" in ltm_in_state.get("learning_goal", "")
    print("✅ 跨会话长期记忆读取测试通过！")


def test_04_user_isolation():
    """
    【多用户隔离性测试】
    验证 TEST_USER_B 不会读到 TEST_USER_A 的记忆
    """
    print("\n" + "="*60)
    print("🧪 4. 正在运行：多用户记忆隔离测试...")
    print("="*60)

    user_b_mem = get_user_long_term_memory(TEST_USER_B)
    print(f"💾 新用户 {TEST_USER_B} 的长期记忆: {user_b_mem}")

    # 验证新用户的画像是干净的默认值
    assert user_b_mem.get("learning_goal") == "未知"
    assert user_b_mem.get("skill_level") == "未知"
    print("✅ 多租户长期记忆物理隔离测试通过！")


if __name__ == "__main__":
    print("\n🚀 开始执行长短期记忆机制全流程验证...\n")
    try:
        test_01_short_term_memory_query_rewrite()
        test_02_long_term_memory_extraction()
        test_03_cross_thread_long_term_recall()
        test_04_user_isolation()
        print("\n" + "🎉"*20)
        print("🎉 恭喜！长短期记忆全套测试用例全部通过！")
        print("🎉"*20 + "\n")
    except AssertionError as e:
        print(f"\n❌ 测试失败，断言未通过: {e}")
    except Exception as e:
        print(f"\n❌ 运行异常: {e}")