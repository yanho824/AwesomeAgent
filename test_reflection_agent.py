# test_reflection_agent.py
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM
from my_reflection_agent import MyReflectionAgent

load_dotenv()
llm = HelloAgentsLLM()

# 使用默认通用提示词
general_agent = MyReflectionAgent(name="我的反思助手", llm=llm)

# # 使用自定义代码生成提示词（类似第四章）
# code_prompts = {
#     "initial": "你是Python专家，请编写函数:{task}",
#     "reflect": "请审查代码的算法效率:\n任务:{task}\n代码:{content}",
#     "refine": "请根据反馈优化代码:\n任务:{task}\n反馈:{feedback}"
# }
code_agent = MyReflectionAgent(
    name="我的代码生成助手",
    llm=llm,
)

input_text = "编写一个名为 generate_fibonacci(n) 的 Python 函数，它接受一个非负整数 $n$ 作为输入，并返回斐波那契数列（Fibonacci Sequence）中第 $n$ 个数的值。"

# 测试使用
result = general_agent.run(input_text=input_text, temperature=0.7)
print(f"最终结果: {result}")