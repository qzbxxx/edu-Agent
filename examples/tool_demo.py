# examples/tool_demo.py
# 演示 @tool 到底把函数变成了什么，以及不加装饰器会怎样。
# 运行：python examples/tool_demo.py
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool


def add_plain(a: int, b: int) -> str:
    """把两个整数相加"""
    return str(a + b)


@tool
def add(a: int, b: int) -> str:
    """把两个整数相加"""
    return str(a + b)


def show(title, value):
    print(f"  {title:<22}{value}")


def main():
    print("=" * 72)
    print("1. 普通函数")
    print("=" * 72)
    show("type()", type(add_plain))
    show("__name__", add_plain.__name__)
    show("直接调用 add(1, 2)", add_plain(1, 2))
    show("有 .name 吗", hasattr(add_plain, "name"))
    show("有 .invoke 吗", hasattr(add_plain, "invoke"))

    print("\n  如果强行当工具用：")
    try:
        add_plain.invoke({"a": 1, "b": 2})
    except AttributeError as error:
        print(f"    AttributeError: {error}")

    print()
    print("=" * 72)
    print("2. 加了 @tool 之后")
    print("=" * 72)
    show("type()", type(add))
    show(".name", add.name)
    show(".description", add.description)
    show(".args", add.args)
    show(".invoke({...})", add.invoke({"a": 1, "b": 2}))
    show("原始函数 .func(1, 2)", add.func(1, 2))

    print("\n  参数校验（故意把 b 传成字符串）：")
    try:
        add.invoke({"a": 1, "b": "二"})
    except Exception as error:
        first_line = str(error).splitlines()[0]
        print(f"    校验失败：{type(error).__name__}: {first_line}")

    print()
    print("=" * 72)
    print("3. 一次工具调用的消息长什么样（不请求网络）")
    print("=" * 72)
    conversation = [
        HumanMessage(content="1 + 2 等于几？"),
        AIMessage(content="", tool_calls=[
            {"name": "add", "args": {"a": 1, "b": 2}, "id": "call_001"}
        ]),
    ]
    result = add.invoke({"a": 1, "b": 2})
    conversation.append(ToolMessage(content=result, tool_call_id="call_001"))

    for message in conversation:
        line = f"  {message.type:<8} content={message.content!r}"
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            line += f"\n           tool_calls={tool_calls}"
        print(line)


if __name__ == "__main__":
    main()
