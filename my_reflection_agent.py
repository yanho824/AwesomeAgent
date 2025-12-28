from typing import List, Optional, Dict
from hello_agents import Config, ReflectionAgent, HelloAgentsLLM, Message, ToolRegistry

DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务:

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的**严重问题**或**必要的改进**：

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量。注意：
1. 如果回答**已经正确完成了任务要求**，即使有小的优化空间，也请回答"无需改进"
2. 只有存在**功能错误、逻辑问题、或明显不符合要求**时，才需要改进
3. 不要为了追求完美而提出次要的优化建议

你的评价：
""",
    "refine": """
请根据反馈意见改进你的回答:

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}

class MyReflectionAgent(ReflectionAgent):
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max_iterations
        self.custom_prompts = custom_prompts
    
    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """调用LLM并获取完整响应"""
        messages = [{"role": "user", "content": prompt}]
        # 使用 invoke 而不是 stream_invoke，因为需要完整的字符串
        return self.llm.invoke(messages, **kwargs) or ""
    
    def run(self, input_text: str, **kwargs) -> str:
        print(f"\n--- 开始处理任务 ---\n任务：{input_text}")

        # memory
        memory = []

        # llm invoke (inital)
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = DEFAULT_PROMPTS["initial"].format(task=input_text)
        initial_response = self._get_llm_response(initial_prompt, **kwargs)
        print(f"\n首次响应：{initial_response}\n")
        memory.append(input_text)
        memory.append(initial_response)

        # for loop
        for i in range(self.max_iterations):
            # llm invoke (reflect)
            last_response = memory[-1]
            reflect_prompt = DEFAULT_PROMPTS["reflect"].format(task=input_text, content=last_response)
            reflect_response = self._get_llm_response(reflect_prompt, **kwargs)
            print(f"\n反思：{reflect_response}\n")
           
            # 检查是否应该停止迭代
            stop_signals = [
                "无需改进",
                "已经很好",
                "质量很高",
                "整体质量较高",
                "满足要求",
                "no need for improvement",
                "looks good",
                "sufficiently good"
            ]
            
            if any(signal in reflect_response.lower() for signal in [s.lower() for s in stop_signals]):
                print(f"\n✅ 反思认为结果已足够好，停止迭代")
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(last_response, "assistant"))
                
                return last_response
            else:
                # llm invoke (refine)
                refine_prompt = DEFAULT_PROMPTS["refine"].format(
                    task=input_text,
                    last_attempt=last_response,
                    feedback=reflect_response
                )
                refined_response = self._get_llm_response(refine_prompt, **kwargs)
                print(f"\n第 {i+1} 次修改后代码：{refined_response}\n")
                memory.append(refined_response)
        
        print("已达到最大迭代次数")
        final_answer = "抱歉，我无法在限定步数内完成这个任务。"
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer
