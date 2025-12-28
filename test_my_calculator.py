import enum
from dotenv import load_dotenv
from my_calculator_tool import create_calculator_registry

load_dotenv()

def test_calculator_tool():
    """custom calculator tool test"""

    # create a registry containing the calculator
    registry = create_calculator_registry()

    print("--- 测试自定义计算器工具 ---")

    # test case
    test_cases = [
        "2 + 3",           # 基本加法
        "10 - 4",          # 基本减法
        "5 * 6",           # 基本乘法
        "15 / 3",          # 基本除法
        "sqrt(16)",        # 平方根
    ]

    for i, expression in enumerate(test_cases, 1):
        print(f"测试 {i}: {expression}")
        result = registry.execute_tool("my_calculator", expression)
        print(f"结果: {result}")

def test_with_simple_agent():
    """test integration with SimpleAgent"""
    from hello_agents import HelloAgentsLLM
    
    # create llm client
    llm = HelloAgentsLLM()
 
    # create a registry containing the calculator
    registry = create_calculator_registry()

    print("🤖 与SimpleAgent集成测试:")

    user_question = "请帮我计算 sqrt(16) + 2 * 3"

    print(f"用户问题：{user_question}")

    # 使用工具计算
    calc_result = registry.execute_tool("my_calculator", "sqrt(16) + 2 * 3")
    print(f"计算结果: {calc_result}")

    # 构建最终回答
    final_messages = [
        {"role": "user", "content": f"计算结果是 {calc_result}，请用自然语言回答用户的问题:{user_question}"}
    ]

    print("\n🎯 SimpleAgent的回答:")
    response = llm.think(final_messages)
    for chunk in response:
        print(chunk, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    test_calculator_tool()
    test_with_simple_agent()