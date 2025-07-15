# /root/autodl-tmp/AetherMind-Agent/src/agent_core.py

import os
# 从项目根目录导入模块
# 注意：确保这些导入路径在您实际运行环境 (例如通过 `python -m src.agent_core`) 时是可访问的
from src.llm_interface import LLMInterface # 导入 LLM 接口
from src.knowledge_manager import KnowledgeManager # 导入知识库管理模块
from src.rag_system import RAGSystem # 导入 RAG 系统模块

class AgentCore:
    def __init__(self):
        """
        初始化智能体的核心组件。
        """
        print("正在初始化智能体核心组件...")
        # 定义聊天模型和嵌入模型的名称
        # 这些名称与您通过 Ollama 拉取的模型名称相对应
        self.chat_model_name = "qwen3:8b" 
        self.embedding_model_name = "Qwen3-Embedding-8B:latest" 

        # 实例化 LLMInterface，用于与大模型交互
        self.llm_interface = LLMInterface(
            chat_model_name=self.chat_model_name,
            embedding_model_name=self.embedding_model_name
        )
        # 实例化 KnowledgeManager，用于管理原始知识库文件
        self.knowledge_manager = KnowledgeManager()
        # 实例化 RAGSystem，用于处理知识检索
        self.rag_system = RAGSystem(
            embedding_model_name=self.embedding_model_name
        )
        # 智能体启动时，自动初始化知识库和向量数据库
        self._initialize_knowledge_base_and_vector_db()
        print("智能体核心组件初始化完成。")

    def _initialize_knowledge_base_and_vector_db(self):
        """
        加载知识库文件，分割文档，并添加到向量数据库。
        这个方法在 AgentCore 初始化时被调用，确保智能体拥有最新的知识库。
        """
        print("正在初始化知识库和向量数据库...")
        documents = self.knowledge_manager.load_knowledge_base()
        
        # 检查知识库是否为空，如果为空则添加一些默认内容
        if not documents or not documents[0].page_content.strip(): 
            print("知识库文件为空或加载失败，正在添加一些默认内容用于测试...")
            default_content = "# 个人知识库\n\n## 初始知识\n\n欢迎来到您的智能知识管理助手！您可以通过总结或提问来管理和回顾知识。例如：今天我学习了Ollama，它是一个用于运行本地大型语言模型的平台。"
            self.knowledge_manager.append_to_knowledge_base(default_content)
            documents = self.knowledge_manager.load_knowledge_base() # 重新加载
            print("默认知识已添加并重新加载。")

        # 将知识库文档分割成小块，便于 RAG 检索
        chunks = self.knowledge_manager.split_documents(documents)
        if not chunks:
            print("文档分割后没有生成块，请检查知识库内容。")
            return

        # 将文档块添加到向量数据库
        self.rag_system.add_documents_to_vector_db(chunks)
        print("知识库和向量数据库初始化完成。")

    def _identify_task(self, user_input: str) -> str:
        """
        根据用户输入识别其意图（任务类型）。
        目前使用基于关键词的简单逻辑进行判断。
        """
        user_input_lower = user_input.lower()
        # 识别知识查询任务的关键词
        if any(keyword in user_input_lower for keyword in ["查询", "回顾", "查找", "什么", "怎么", "告诉我"]):
            return "query_knowledge"
        # 识别总结整理任务的关键词
        elif any(keyword in user_input_lower for keyword in ["总结", "整理", "记录", "今日总结", "汇总"]):
            return "summarize_and_add"
        # 识别计划制定任务的关键词
        elif any(keyword in user_input_lower for keyword in ["计划", "规划", "安排", "目标", "制定"]):
            return "formulate_plan"
        # 如果关键词不匹配，则认为是无法识别的任务
        else:
            return "unknown_task" 

    def execute_task(self, task_type: str, *args, **kwargs) -> str:
        """
        根据识别出的任务类型，分发并执行相应的处理函数。
        这是外部接口（如 main.py）调用智能体功能的入口。
        """
        if task_type == "query_knowledge":
            query = kwargs.get("query", "")
            return self._handle_query_task(query)
        elif task_type == "summarize_and_add":
            content_to_summarize = kwargs.get("content", "")
            return self._handle_summarize_and_add_task(content_to_summarize)
        elif task_type == "formulate_plan":
            plan_goal = kwargs.get("goal", "")
            return self._handle_formulate_plan_task(plan_goal)
        else: # unknown_task
            return "抱歉，我当前只支持知识查询、总结整理和计划制定任务，不支持通用聊天。"

    def _handle_query_task(self, query: str) -> str:
        """
        具体处理“知识查询”任务的逻辑。
        它会调用 RAG 系统检索相关文档，然后结合查询和文档生成回答。
        """
        if not query:
            return "请提供您要查询的具体内容。"

        print(f"\n执行任务: 知识查询 - '{query}'")
        # 从向量数据库检索最相关的3个文档
        retrieved_docs = self.rag_system.retrieve_documents(query, k=3) 

        if not retrieved_docs:
            print("未能检索到相关文档，直接调用LLM生成回答。")
            # 如果没有检索到，仍然尝试让 LLM 回答，但会提示信息不足
            response_prompt = f"无法在知识库中找到与 '{query}' 相关的信息。但请尝试直接回答用户的问题：{query}"
            response = self.llm_interface.generate_chat_response(response_prompt)
            return response
        
        # 将检索到的文档内容拼接成上下文
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 构建发送给 LLM 的提示词，指导它基于知识库信息进行回答
        prompt = f"""根据以下知识库信息，简洁地回答用户的问题。如果信息不足，请如实说明或尝试给出通用性回答。请用中文回答。

        知识库信息:
        {context}

        用户问题:
        {query}
        """
        print("正在调用LLM生成回答...")
        response = self.llm_interface.generate_chat_response(prompt)
        print("LLM回答生成完成。")
        return response

    def _handle_summarize_and_add_task(self, content: str) -> str:
        """
        具体处理“总结整理并添加知识”任务的逻辑。
        它会调用 LLM 对内容进行总结，然后将总结添加到知识库并更新向量数据库。
        """
        if not content:
            return "请提供需要总结和添加的内容。"

        print(f"\n执行任务: 总结整理并添加知识 - 内容长度 {len(content)}")
        
        # 构建提示词，让 LLM 对提供的原始内容进行总结
        summary_prompt = f"""请对以下内容进行简洁、清晰的总结，并以Markdown格式（如使用标题、列表等）整理好，方便未来回顾。重点突出关键信息和核心概念。
        
        原始内容:
        {content}
        """
        print("正在调用LLM生成总结...")
        generated_summary = self.llm_interface.generate_chat_response(summary_prompt)

        # 将 LLM 生成的总结追加到知识库文件
        self.knowledge_manager.append_to_knowledge_base(generated_summary)
        
        # 重新加载整个知识库并更新向量数据库，以包含新的总结
        # 注意：对于大型知识库，这可能不是最高效的方法，但在当前阶段足够用
        print("正在重新加载知识库并更新向量数据库...")
        documents = self.knowledge_manager.load_knowledge_base()
        chunks = self.knowledge_manager.split_documents(documents)
        self.rag_system.add_documents_to_vector_db(chunks) # Chroma 会处理文档的去重或更新
        print("知识库和向量数据库更新完成。")
        
        return f"已成功将总结整理并添加到您的知识库中。\n总结内容预览:\n{generated_summary[:200]}..."

    def _handle_formulate_plan_task(self, goal: str) -> str:
        """
        具体处理“制定计划”任务的逻辑。
        它会根据目标调用 LLM 制定详细计划，并可选地从知识库中检索相关参考信息。
        """
        if not goal:
            return "请提供您要制定计划的具体目标。"

        print(f"\n执行任务: 制定计划 - 目标: '{goal}'")
        
        # 尝试从知识库中检索与计划制定或目标相关的参考信息
        retrieved_plan_docs = self.rag_system.retrieve_documents(f"如何制定计划？关于 {goal} 的学习方法或步骤。", k=2)
        plan_context = ""
        if retrieved_plan_docs:
            plan_context = "\n\n以下知识库中的规划相关信息可供参考：\n" + \
                           "\n\n".join([doc.page_content for doc in retrieved_plan_docs])

        # 构建提示词，让 LLM 根据目标和参考信息制定计划
        plan_prompt = f"""根据以下目标，请您制定一个详细、可行、分步骤的学习或行动计划。计划应包含时间、资源、具体任务和衡量标准。请用中文回答。{plan_context}

        目标:
        {goal}
        """
        print("正在调用LLM制定计划...")
        generated_plan = self.llm_interface.generate_chat_response(plan_prompt)
        
        # 您可以选择是否将生成的计划也添加到知识库中
        # 如果需要，取消注释以下两行：
        # self.knowledge_manager.append_to_knowledge_base(f"### 计划: {goal}\n\n{generated_plan}")
        # self.rag_system.add_documents_to_vector_db(self.knowledge_manager.split_documents(self.knowledge_manager.load_knowledge_base()))
        
        return f"已为您制定计划：\n{generated_plan}"

# # --- 测试 AgentCore 功能 (仅在直接运行此文件时执行) ---
# if __name__ == "__main__":
#     print("--- 测试 AgentCore ---")
    
#     # 确保 ollama serve 正在运行：
#     # 在您的服务器终端执行 `ollama serve`
#     # 确保 qwen3:8b 和 Qwen3-Embedding-8B:latest 模型已拉取：
#     # 在您的服务器终端执行 `ollama pull qwen3:8b`
#     # 在您的服务器终端执行 `ollama pull Qwen3-Embedding-8B:latest`
    
#     # 初始化 AgentCore 实例
#     agent = AgentCore()

#     # --- 开始各项任务的测试 ---

#     # 1. 测试知识查询任务
#     print("\n" + "="*30 + "\n--- 1. 测试知识查询任务 ---\n" + "="*30)
#     query_input_1_raw = "查询 Ollama 模型和 RAG 系统有什么进展？"
#     task_type_1 = agent._identify_task(query_input_1_raw)
#     response_1 = agent.execute_task(task_type_1, query="Ollama 模型和 RAG 系统有什么进展？")
#     print(f"\nAgent 回答 1 (查询):\n{response_1}")

#     # 2. 测试总结整理并添加知识任务
#     print("\n" + "="*30 + "\n--- 2. 测试总结整理并添加知识任务 ---\n" + "="*30)
#     content_to_summarize_2 = """
#     今天参加了公司关于敏捷开发的培训，主要学习了Scrum框架。Scrum强调迭代开发，通过短周期的Sprint（冲刺）来交付可工作的产品增量。
#     核心角色包括产品负责人（Product Owner）、Scrum Master和开发团队。
#     重要的会议有Sprint计划会议、每日站会、Sprint评审会议和Sprint回顾会议。
#     Scrum的优势在于灵活性高、响应变化快、团队协作紧密。
#     我需要将这些内容整理进我的知识库，方便以后复习。
#     """
#     task_type_2 = agent._identify_task("总结 " + content_to_summarize_2[:50] + "...") # 识别任务时传入部分内容
#     response_2 = agent.execute_task(task_type_2, content=content_to_summarize_2)
#     print(f"\nAgent 回答 2 (总结):\n{response_2}")
    
#     # 再次查询新添加的知识，看是否能检索到
#     print("\n" + "="*30 + "\n--- 2.1 再次查询新添加的知识 ---\n" + "="*30)
#     query_input_2_1_raw = "查询 Scrum 敏捷开发的核心要素是什么？"
#     task_type_2_1 = agent._identify_task(query_input_2_1_raw)
#     response_2_1 = agent.execute_task(task_type_2_1, query="Scrum 敏捷开发的核心要素是什么？")
#     print(f"\nAgent 回答 2.1 (新知识查询):\n{response_2_1}")


#     # 3. 测试计划制定任务
#     print("\n" + "="*30 + "\n--- 3. 测试计划制定任务 ---\n" + "="*30)
#     plan_input_3_raw = "制定一个学习 Go 语言基础的计划。"
#     task_type_3 = agent._identify_task(plan_input_3_raw)
#     response_3 = agent.execute_task(task_type_3, goal="学习 Go 语言基础")
#     print(f"\nAgent 回答 3 (计划):\n{response_3}")

#     # 4. 测试无法识别的任务 (通用聊天)
#     print("\n" + "="*30 + "\n--- 4. 测试无法识别的任务 (通用聊天) ---\n" + "="*30)
#     chat_input_4_raw = "你好，你今天过得怎么样？"
#     task_type_4 = agent._identify_task(chat_input_4_raw)
#     response_4 = agent.execute_task(task_type_4) # 这种情况下不需要传递额外的kwargs
#     print(f"\nAgent 回答 4 (通用聊天):\n{response_4}")

#     print("\n" + "="*30 + "\nAgentCore 所有测试完成！\n" + "="*30)