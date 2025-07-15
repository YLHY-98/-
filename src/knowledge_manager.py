# /root/autodl-tmp/AetherMind-Agent/src/knowledge_manager.py

import os
from langchain_community.document_loaders import TextLoader # 用于加载文本文件
from langchain.text_splitter import RecursiveCharacterTextSplitter # 用于文本分割

class KnowledgeManager:
    def __init__(self, knowledge_base_path: str = "../data/knowledge_base.md"):
        """
        初始化知识库管理器。
        Args:
            knowledge_base_path: 知识库文件的路径。
        """
        self.knowledge_base_path = os.path.join(os.path.dirname(__file__), knowledge_base_path)
        # 确保data目录存在
        os.makedirs(os.path.dirname(self.knowledge_base_path), exist_ok=True)
        # 如果知识库文件不存在，则创建它
        if not os.path.exists(self.knowledge_base_path):
            with open(self.knowledge_base_path, 'w', encoding='utf-8') as f:
                f.write("# 个人知识库\n\n") # 写入一个初始标题

    def load_knowledge_base(self):
        """
        加载知识库文件中的所有文本内容。
        Returns:
            加载的文档列表。
        """
        if not os.path.exists(self.knowledge_base_path):
            print(f"知识库文件不存在: {self.knowledge_base_path}")
            return []
        
        print(f"正在加载知识库文件: {self.knowledge_base_path}")
        loader = TextLoader(self.knowledge_base_path, encoding='utf-8')
        documents = loader.load()
        print(f"已加载 {len(documents)} 个文档。")
        return documents

    def split_documents(self, documents, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        将加载的文档分割成更小的块。
        Args:
            documents: 从知识库加载的文档列表。
            chunk_size: 每个文本块的最大长度。
            chunk_overlap: 文本块之间的重叠长度。
        Returns:
            分割后的文本块列表。
        """
        if not documents:
            print("没有文档可供分割。")
            return []
            
        print(f"正在分割文档 (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True, # 添加每个块在原始文档中的起始索引
        )
        texts = text_splitter.split_documents(documents)
        print(f"文档已分割成 {len(texts)} 个块。")
        return texts

    def append_to_knowledge_base(self, content: str):
        """
        将新内容追加到知识库文件。
        Args:
            content: 需要追加到知识库的文本内容。
        """
        with open(self.knowledge_base_path, 'a', encoding='utf-8') as f:
            f.write("\n\n" + content)
        print(f"内容已成功追加到知识库文件: {self.knowledge_base_path}")

# # --- 测试 KnowledgeManager 功能 (仅在直接运行此文件时执行) ---
# if __name__ == "__main__":
#     print("--- 测试 KnowledgeManager ---")
#     # 假设项目根目录是 AetherMind-Agent
#     # 知识库文件路径相对于当前脚本文件 (src/knowledge_manager.py)
#     # 实际路径会是 /root/autodl-tmp/AetherMind-Agent/data/knowledge_base.md
#     km = KnowledgeManager()

#     # 1. 测试初始文件创建和加载
#     print("\n--- 1. 测试初始文件创建和加载 ---")
#     initial_docs = km.load_knowledge_base()
#     if initial_docs:
#         print(f"初始知识库内容预览 (前200字符): {initial_docs[0].page_content[:200]}...")
#     else:
#         print("初始知识库为空或加载失败。")

#     # 2. 测试追加内容
#     print("\n--- 2. 测试追加内容 ---")
#     test_content = "### 2025年7月15日学习总结\n\n今天学习了如何设置Ollama本地大模型，并成功解决了Ollama Chat和Embeddings模型的调用问题。这是一个重要的里程碑，为后续的RAG系统打下了基础。"
#     km.append_to_knowledge_base(test_content)

#     # 3. 重新加载并测试文档分割
#     print("\n--- 3. 重新加载并测试文档分割 ---")
#     updated_docs = km.load_knowledge_base()
#     if updated_docs:
#         chunks = km.split_documents(updated_docs)
#         if chunks:
#             print(f"第一个文本块内容预览 (前200字符): {chunks[0].page_content[:200]}...")
#             print(f"第一个文本块的元数据: {chunks[0].metadata}")
#         else:
#             print("文档分割后没有生成块。")
#     else:
#         print("重新加载知识库失败。")

#     print("\nKnowledgeManager 测试完成。")