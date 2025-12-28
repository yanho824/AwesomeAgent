# 默认规划器提示词模板
import ast
from typing import List, Optional, Dict
from urllib import response
from hello_agents import HelloAgentsLLM, Message, PlanAndSolveAgent, Config


DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。

问题: {question}

**重要：你必须严格按照以下格式输出，包括代码块标记：**

```python
["步骤1", "步骤2", "步骤3", ...]
```

注意：必须包含开头的 ```python 和结尾的 ```，不要省略！
"""

# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""

class Planner:
    """规划器"""
    
    def __init__(
        self,
        llm: HelloAgentsLLM,
        prompt_template: Optional[str] = None
    ):
        self.llm = llm
        self.prompt_template = prompt_template if prompt_template else DEFAULT_PLANNER_PROMPT
    
    def plan(self, input_text: str, **kwargs) -> List[str]:
        """
        生成执行计划

        Args:
            question: 要解决的问题
            **kwargs: LLM调用参数

        Returns:
            步骤列表
        """
        prompt = self.prompt_template.format(question=input_text)
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---\n")
        response_text = self.llm.invoke(messages, **kwargs) or ""
        print(f"计划已生成：\n{response_text}")

        try:
            plan_str = None
            
            # 方法1：尝试提取 ```python 代码块
            if "```python" in response_text:
                parts = response_text.split("```python")
                if len(parts) > 1:
                    plan_str = parts[1].split("```")[0].strip()
            
            # 方法2：尝试提取普通代码块 ```
            elif "```" in response_text:
                parts = response_text.split("```")
                if len(parts) > 1:
                    plan_str = parts[1].strip()
            
            # 方法3：直接查找列表 (LLM可能直接返回列表而不用代码块)
            if not plan_str and "[" in response_text and "]" in response_text:
                start = response_text.find("[")
                end = response_text.rfind("]") + 1
                plan_str = response_text[start:end]
            
            # 尝试解析
            if plan_str:
                plan = ast.literal_eval(plan_str)
                return plan if isinstance(plan, list) else []
            else:
                print(f"❌ 无法在响应中找到列表格式")
                return []
                
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错：{e}")
            print(f"原始响应：{response_text}")
            return []
        except Exception as e:  
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []

class Executor:
    """执行器 - 负责按计划逐步执行"""
    def __init__(
        self,
        llm: HelloAgentsLLM,
        prompt_template: Optional[str] = None
    ):
        self.llm = llm
        self.prompt_template = prompt_template if prompt_template else DEFAULT_EXECUTOR_PROMPT
    
    def execute(self, input_text: str, plan: List[str], **kwargs) -> str:
        """
        按计划执行任务

        Args:
            question: 原始问题
            plan: 执行计划
            **kwargs: LLM调用参数

        Returns:
            最终答案
        """
        history = ""
        final_answer = ""

        print("\n--- 正在执行计划 ---")

        for i, step in enumerate(plan, 1):
            print(f"\n-> 正在执行步骤 {i} / len(plan): {step}")
            prompt = self.prompt_template.format(
                question=input_text,
                plan=plan,
                history=history if history else "无",
                current_step=step
            )    
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.invoke(messages, **kwargs)

            history += f"步骤{i}: {step}\n结果：{response_text}"
            final_answer = response_text
            print(f"步骤 {i} 已完成，结果: {final_answer}")

        return final_answer

    
class MyPlanAndSolveAgent(PlanAndSolveAgent):
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        super().__init__(name, llm, system_prompt, config)
        
        # 设置提示词模板，用户自定义优先，否则使用默认模板
        planner_prompt = custom_prompts.get("planner") if custom_prompts else DEFAULT_PLANNER_PROMPT
        executor_prompt = custom_prompts.get("executor") if custom_prompts else DEFAULT_EXECUTOR_PROMPT

        self.planner = Planner(self.llm, planner_prompt)
        self.executor = Executor(self.llm, executor_prompt)

    def run(self, question: str, **kwargs) -> str:
        """
        运行Plan and solve agent
        """
        print(f"\n {self.name} 开始处理问题：{question}")

        # 1. 生成计划
        plan = self.planner.plan(question, **kwargs)
        if not plan:
            final_answer = "无法生成有效的行动计划，任务终止。"
            print(f"\n--- 任务终止 ---\n{final_answer}")

            # 保存到历史记录
            self.add_message(Message(question, "user"))
            self.add_message(Message(final_answer, "assistant"))
            
            return final_answer

        # 2. 按照计划执行
        final_answer = self.executor.execute(question, plan, **kwargs)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")
        
        # 保存到历史记录
        self.add_message(Message(question, "user"))
        self.add_message(Message(final_answer, "assistant"))
        
        return final_answer

