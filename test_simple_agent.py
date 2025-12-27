from dotenv import load_dotenv
from hello_agents import CalculatorTool, HelloAgentsLLM, ToolRegistry

from my_simple_agent import MySimpleAgent

load_dotenv()

# 创建LLM
llm = HelloAgentsLLM()

# 测试基础Agent
basic_agent = MySimpleAgent(
    name="basic",
    llm=llm,
    system_prompt="你是一个友好的AI助手，请用简洁明了的方式回答问题。"
)

response1 = basic_agent.run("你好，请介绍一下自己")
print(f"基础对话响应: {response1}\n")

# 测试带工具的Agent
tool_registry = ToolRegistry()
calculator = CalculatorTool()
tool_registry.register_tool(calculator)

enhanced_agent = MySimpleAgent(
    name="enhanced",
    llm=llm,
    system_prompt="你是一个智能助手，可以使用工具来帮助用户",
    tool_registry=tool_registry,
    enable_tool_calling=True
)
response2 = enhanced_agent.run("请帮我计算 15 * 8 + 32")
print(f"工具增强响应: {response2}\n")

# 测试流式响应
print("===流式响应测试===\n")
print("流式响应：", end="")
for chunk in basic_agent.stream_run("请解释什么是人工智能"):
    pass

# 测试动态加载工具
# 测试4:动态添加工具
print("\n=== 测试4:动态工具管理 ===")
print(f"添加工具前: {basic_agent.has_tools()}")
basic_agent.add_tool(calculator)
print(f"添加工具后: {basic_agent.has_tools()}")
print(f"可用工具: {basic_agent.list_tools()}")

# 查看对话历史
print(f"\n对话历史: {len(basic_agent.get_history())} 条消息")
