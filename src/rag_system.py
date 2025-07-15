# /root/autodl-tmp/AetherMind-Agent/src/rag_system.py

import os
from langchain_community.vectorstores import Chroma # 导入Chroma向量数据库
from langchain_community.embeddings import OllamaEmbeddings # 导入Ollama嵌入模型

# 导入KnowledgeManager，用于获取分割后的文档
from .knowledge_manager import KnowledgeManager # 注意：这里使用相对导入，因为两者都在src目录下

class RAGSystem:
    def __init__(self, 
                 embedding_model_name: str = "Qwen3-Embedding-8B:latest", # 嵌入模型名称
                 persist_directory: str = "../data/vector_store"): # 向量数据库持久化路径
        """
        初始化 RAG 系统。
        Args:
            embedding_model_name: 用于生成嵌入的 Ollama 模型名称。
            persist_directory: 向量数据库持久化存储的目录。
        """
        self.embedding_model = OllamaEmbeddings(model=embedding_model_name)
        # 向量数据库持久化路径相对于当前脚本文件 (src/rag_system.py)
        self.persist_directory = os.path.join(os.path.dirname(__file__), persist_directory)
        os.makedirs(self.persist_directory, exist_ok=True) # 确保目录存在

        # 初始化 Chroma 向量数据库
        # 如果目录已存在，则加载；否则创建新的数据库
        print(f"正在初始化向量数据库，持久化路径: {self.persist_directory}")
        self.vector_db = Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=self.embedding_model
        )
        print("向量数据库初始化完成。")

    def add_documents_to_vector_db(self, documents):
        """
        将文本块及其嵌入添加到向量数据库。
        Args:
            documents: 待添加的文本块列表 (通常来自KnowledgeManager的split_documents方法)。
        Returns:
            添加的文档ID列表。
        """
        if not documents:
            print("没有文档可供添加。")
            return []
        
        print(f"正在将 {len(documents)} 个文档添加到向量数据库...")
        # Chroma 的 .add_documents() 方法会自动处理嵌入生成
        # 如果Chroma实例是空的，它会创建一个新的Collection；否则会添加到现有Collection
        # 注意：这里假设您的documents是Langchain的Document对象列表
        ids = self.vector_db.add_documents(documents)
        print(f"成功添加 {len(ids)} 个文档到向量数据库。")
        # 添加文档后需要调用persist()来保存到磁盘
        self.vector_db.persist()
        print("向量数据库已保存到磁盘。")
        return ids

    def retrieve_documents(self, query: str, k: int = 4):
        """
        根据用户查询，从向量数据库中检索最相关的文档块。
        Args:
            query: 用户查询字符串。
            k: 检索返回的最相关文档块的数量。
        Returns:
            检索到的文档块列表。
        """
        if not query:
            print("查询为空，无法检索。")
            return []
            
        print(f"正在检索与查询 '{query}' 相关的文档 (k={k})...")
        # .as_retriever() 方法会返回一个Retriever对象，可以直接调用其get_relevant_documents
        # 或者直接使用 .similarity_search() 方法
        retrieved_docs = self.vector_db.similarity_search(query, k=k)
        print(f"检索到 {len(retrieved_docs)} 个文档。")
        return retrieved_docs

# # --- 测试 RAGSystem 功能 (仅在直接运行此文件时执行) ---
# if __name__ == "__main__":
#     print("--- 测试 RAGSystem ---")
    
#     # 1. 初始化 KnowledgeManager 并加载/分割文档
#     print("\n--- 1. 初始化 KnowledgeManager 并加载/分割文档 ---")
#     # 知识库文件路径相对于 src/knowledge_manager.py
#     # 但在这里我们从 src/rag_system.py 导入 KnowledgeManager，所以路径结构保持一致
#     km = KnowledgeManager() 
#     initial_docs = km.load_knowledge_base() # 加载所有文档
    
#     # 确保知识库有足够的内容用于测试
#     if not initial_docs or "学习总结" not in initial_docs[0].page_content: # 简单检查一下之前追加的内容
#         print("知识库内容不足，添加一些测试内容...")
#         test_content_1 = "### 2025年7月15日学习总结\n\n今天学习了如何设置Ollama本地大模型，并成功解决了Ollama Chat和Embeddings模型的调用问题。这是一个重要的里程碑，为后续的RAG系统打下了基础。"
#         test_content_2 = "### 2025年7月16日项目进展\n\n完成了KnowledgeManager和RAGSystem模块的初步开发，并成功进行了本地测试。下一步将整合这些模块到核心智能体逻辑中。"
#         test_content_3 = "### Python编程技巧\n\nPython中的虚拟环境（venv）是管理项目依赖的最佳实践，可以避免不同项目间的库版本冲突。"
#         km.append_to_knowledge_base(test_content_1)
#         km.append_to_knowledge_base(test_content_2)
#         km.append_to_knowledge_base(test_content_3)
#         initial_docs = km.load_knowledge_base() # 重新加载
        
#     chunks_to_add = km.split_documents(initial_docs) # 分割文档

#     # 2. 初始化 RAGSystem
#     print("\n--- 2. 初始化 RAGSystem ---")
#     # 注意：这里 embedding_model_name 应该与您实际拉取的 Ollama 嵌入模型名称一致
#     # 比如您之前测试通过的是nomi-embed-text，或者您自定义的Qwen3-Embedding-8B:latest
#     rag_system = RAGSystem(embedding_model_name="Qwen3-Embedding-8B:latest") 
    
#     # 3. 将文档添加到向量数据库
#     print("\n--- 3. 将文档添加到向量数据库 ---")
#     rag_system.add_documents_to_vector_db(chunks_to_add)

#     # 4. 测试文档检索
#     print("\n--- 4. 测试文档检索 ---")
#     query = "Ollama模型和RAG系统有什么进展？"
#     retrieved_docs = rag_system.retrieve_documents(query, k=2)
    
#     print(f"\n查询: '{query}' 的检索结果:")
#     for i, doc in enumerate(retrieved_docs):
#         print(f"\n--- 检索到的文档 {i+1} ---")
#         print(f"内容预览: {doc.page_content[:200]}...")
#         print(f"元数据: {doc.metadata}")
    
#     print("\nRAGSystem 测试完成。")