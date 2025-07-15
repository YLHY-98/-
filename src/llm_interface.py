# /root/autodl-tmp/AetherMind-Agent/src/llm_interface.py

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

class LLMInterface:
    def __init__(self, chat_model_name: str = "qwen3:8b", embedding_model_name: str = "Qwen3-Embedding-8B:latest"):
        """
        初始化 LLM 接口，包括聊天模型和嵌入模型。
        Args:
            chat_model_name: 用于聊天的 Ollama 模型名称。默认为 "qwen3:8b"。
            embedding_model_name: 用于生成嵌入的 Ollama 模型名称。默认为 "Qwen3-Embedding-8B:latest"。
        """
        print(f"正在初始化聊天模型: {chat_model_name}")
        # 注意：LangChainDeprecationWarning 是正常的，因为它提示您未来的版本更新
        self.chat_model = Ollama(model=chat_model_name)
        print(f"聊天模型 {chat_model_name} 初始化完成。")

        print(f"正在初始化嵌入模型: {embedding_model_name}")
        self.embedding_model = OllamaEmbeddings(model=embedding_model_name)
        print(f"嵌入模型 {embedding_model_name} 初始化完成。")

    def generate_chat_response(self, prompt: str, history: list = None) -> str:
        """
        使用聊天模型生成对话响应。
        Args:
            prompt: 用户的当前输入或构建的提示。
            history: 之前的对话历史，格式为 LangChain 的消息列表（可选）。
                     例如：[HumanMessage(content="你好"), AIMessage(content="你好！")]
        Returns:
            模型的回答字符串。
        """
        if history is None:
            history = []

        # 构建发送给 LLM 的消息列表
        # 添加系统消息以设定模型角色和语言偏好
        messages = [SystemMessage(content="你是一个有用的助手，请用中文回答。")] + history + [HumanMessage(content=prompt)]
        
        try:
            # 调用 Ollama 模型，直接传递消息列表
            response = self.chat_model.invoke(messages)
            return response
        except Exception as e:
            print(f"调用 Ollama 聊天模型时发生错误: {e}")
            return "抱歉，LLM 服务或模型调用出现问题，无法生成回答。请检查 Ollama 服务是否运行，以及模型是否已拉取。"

    def get_embeddings(self, text: str) -> list[float]:
        """
        使用嵌入模型生成文本的嵌入向量。
        Args:
            text: 需要生成嵌入的文本。
        Returns:
            文本的嵌入向量列表。
        """
        try:
            embeddings = self.embedding_model.embed_query(text)
            return embeddings
        except Exception as e:
            print(f"调用 Ollama 嵌入模型时发生错误: {e}")
            return []

# # --- 测试 LLMInterface 功能 (仅在直接运行此文件时执行) ---
# if __name__ == "__main__":
#     print("--- 测试 LLMInterface ---")
    
#     # 确保 ollama serve 正在运行：
#     # 在您的服务器终端执行 `ollama serve`
#     # 确保 qwen3:8b 和 Qwen3-Embedding-8B:latest 模型已拉取：
#     # 在您的服务器终端执行 `ollama pull qwen3:8b`
#     # 在您的服务器终端执行 `ollama pull Qwen3-Embedding-8B:latest`

#     # 初始化 LLMInterface，使用默认或指定的模型
#     llm_interface = LLMInterface(chat_model_name="qwen3:8b", embedding_model_name="Qwen3-Embedding-8B:latest")

#     # 测试聊天模型
#     print("\n--- 1. 测试聊天模型 ---")
#     chat_response = llm_interface.generate_chat_response("你好，请做个简单的自我介绍。")
#     print(f"聊天模型回答: {chat_response}")

#     # 测试嵌入模型
#     print("\n--- 2. 测试嵌入模型 ---")
#     test_text = "这是一个用于生成嵌入的测试文本，希望能够成功。"
#     embeddings = llm_interface.get_embeddings(test_text)
#     if embeddings:
#         print(f"文本嵌入向量的前5个维度: {embeddings[:5]}...")
#         print(f"嵌入向量的长度: {len(embeddings)}")
#     else:
#         print("未能获取嵌入向量。")

#     print("\nLLMInterface 测试完成。")