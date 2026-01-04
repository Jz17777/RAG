"""
RAG 全流程集成主程序（异步流式版本）
集成 Multi-query、RRF 重排序、qwen3-rerank 二次重排序和流式答案生成

主要功能：
- Multi-query 策略（自动生成多个相关查询）
- RRF (Reciprocal Rank Fusion) 重排序（融合多个查询结果，提高检索质量）
- qwen3-rerank 模型二次重排序（进一步提升检索精度）
- 异步流式生成，支持对话历史管理

使用方法：

1. 命令行使用：
   python main.py "你的问题"
   python main.py  # 交互模式

2. 作为模块使用：
   import asyncio
   from main import RAGPipeline
   
   async def main():
       pipeline = RAGPipeline()
       async for chunk in pipeline.query_stream_async("你的问题"):
           print(chunk, end="", flush=True)
   
   asyncio.run(main())

环境配置：
需要设置环境变量 DASHSCOPE_API_KEY 和 DASHSCOPE_API_URL
"""
import os
import asyncio
from typing import List, Optional
from functools import partial
import dotenv

# 加载环境变量
dotenv.load_dotenv()

# 导入其他模块
from Retrieval.Rerank import VectorRetriever
from Generation.Generate import create_multi_question_rag_generation_stream_async


class RAGPipeline:
    """
    RAG 全流程管道类
    整合查询改写、向量检索和答案生成
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents",
        embedding_type: str = "dashscope",
        embedding_model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: str = "qwen-flash",
        temperature: float = 0.3,
        top_p: float = 0.9,
        n_queries: int = 3,
        retrieval_k: int = 5,
        max_docs: Optional[int] = None,
        rrf_k: int = 60,
        use_rerank: bool = True,
        rerank_model: str = "qwen3-rerank",
        rrf_top_n: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
        use_scores: bool = False
    ):
        """
        初始化 RAG 管道
        
        Args:
            persist_directory: Chroma 数据库持久化目录路径
            collection_name: 集合名称
            embedding_type: 嵌入模型类型，"dashscope" 或 "local"
            embedding_model: 嵌入模型名称（可选）
            api_key: API Key（如果为 None，从环境变量读取）
            api_url: API URL（如果为 None，从环境变量读取）
            model: 生成模型名称
            temperature: 生成温度参数
            top_p: 生成 top_p 参数
            n_queries: 多查询生成的查询数量
            retrieval_k: 每个查询检索的文档数量（用于 RRF 检索中的 k_per_query）
            max_docs: 生成时使用的最大文档数量（如果为 None，使用所有检索到的文档）
            rrf_k: RRF 算法常数 k，默认为 60。值越小排名影响越大
            use_rerank: 是否使用 qwen3-rerank 模型进行二次重排序，默认为 True
            rerank_model: 重排序模型名称，默认为 "qwen3-rerank"
            rrf_top_n: RRF 排序后保留的文档数量，用于 rerank 重排序（如果为 None，保留所有文档）
            rerank_top_n: rerank 模型返回的文档数量（如果为 None，使用 top_n 的值）
            use_scores: 是否在生成答案时使用文档分数，默认为 False
        """
        # 初始化向量检索器
        print("正在初始化向量检索器...")
        self.retriever = VectorRetriever(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_type=embedding_type,
            embedding_model=embedding_model
        )
        print("向量检索器初始化成功")
        
        # 保存配置参数
        self.retrieval_k = retrieval_k
        self.n_queries = n_queries
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_docs = max_docs
        self.rrf_k = rrf_k
        self.use_rerank = use_rerank
        self.rerank_model = rerank_model
        self.rrf_top_n = rrf_top_n
        self.rerank_top_n = rerank_top_n
        self.use_scores = use_scores
        
        # 初始化对话历史管理
        self.chat_history: List[tuple] = []  # 存储 (用户问题, 助手回答) 的元组列表
        self.max_history_length: int = 10  # 最大保存的对话轮数
    
    def clear_history(self):
        """
        清除对话历史
        """
        self.chat_history = []
        print("对话历史已清除")
    
    def get_history(self) -> List[tuple]:
        """
        获取当前对话历史
        
        Returns:
            对话历史列表，每个元素为 (用户问题, 助手回答) 元组
        """
        return self.chat_history.copy()
    
    def set_max_history_length(self, max_length: int):
        """
        设置最大对话历史长度
        
        Args:
            max_length: 最大保存的对话轮数
        """
        self.max_history_length = max_length
        # 如果当前历史超过新长度，截断
        if len(self.chat_history) > max_length:
            self.chat_history = self.chat_history[-max_length:]
    
    async def query_stream_async(self, question: str, use_chat_history: bool = True):
        """
        异步流式执行 RAG 流程
        
        Args:
            question: 用户问题
            use_chat_history: 是否使用对话历史，默认为 True
        
        Yields:
            生成答案的文本块（实时从模型流式返回）
        """
        if not question or not question.strip():
            yield "未提供问题"
            return
        
        # 步骤 1: 使用 multi-query + RRF + rerank 策略进行检索（这部分不是流式的）
        # 异步调用同步的检索方法
        loop = asyncio.get_event_loop()
        retrieve_func = partial(
            self.retriever.retrieve_with_multi_query_rrf_with_scores,
            question=question,
            n_queries=self.n_queries,
            k_per_query=self.retrieval_k,
            top_n=self.max_docs if self.max_docs else 20,
            rrf_k=self.rrf_k,
            use_original_query=True,
            use_rerank=self.use_rerank,
            rerank_model=self.rerank_model,
            rerank_api_key=self.api_key,
            rerank_api_url=None,
            rrf_top_n=self.rrf_top_n,
            rerank_top_n=self.rerank_top_n
        )
        retrieval_results = await loop.run_in_executor(None, retrieve_func)
        
        # 提取文档列表
        all_documents = [doc for doc, score in retrieval_results]
        queries = []  # 查询在检索方法内部生成
        
        if not all_documents:
            yield "未检索到相关文档，不足以回答"
            return
        
        # 步骤 2: 异步流式生成答案
        print("\n正在流式生成答案...")
        
        # 准备对话历史（如果启用）
        chat_history_to_use = self.chat_history if use_chat_history else []
        
        # 创建异步流式生成器（这个函数不是异步的，它返回一个异步生成器函数）
        stream_generator = create_multi_question_rag_generation_stream_async(
            api_key=self.api_key or os.getenv("DASHSCOPE_API_KEY"),
            api_url=self.api_url or os.getenv("DASHSCOPE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_docs=self.max_docs,
            with_scores=self.use_scores,
            enable_chat_history=use_chat_history
        )
        
        # 准备输入数据
        input_data = {
            "original_question": question,
            "queries": queries,
            "chat_history": chat_history_to_use
        }
        
        if self.use_scores:
            input_data["results"] = retrieval_results
        else:
            input_data["documents"] = all_documents
        
        # 流式生成并收集完整答案（用于更新对话历史）
        full_answer = ""
        try:
            async for chunk in stream_generator(input_data):
                full_answer += chunk
                yield chunk
        except Exception as e:
            error_msg = f"流式生成时发生错误：{str(e)[:200]}"
            print(f"\n警告：{error_msg}")
            yield error_msg
            return
        
        # 更新对话历史（使用完整答案）
        if use_chat_history and full_answer:
            self.chat_history.append((question, full_answer))
            # 限制历史长度
            if len(self.chat_history) > self.max_history_length:
                self.chat_history = self.chat_history[-self.max_history_length:]
        
        print("\n答案生成完成\n")


async def main_async():
    """
    主函数：演示 RAG 流程的使用
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG 全流程问答系统')
    parser.add_argument('question', nargs='?', help='用户问题（如果不提供，将进入交互模式）')
    parser.add_argument('--persist-dir', default='./chroma_db', 
                       help='向量数据库存储目录（默认: ./chroma_db）')
    parser.add_argument('--collection', default='documents', 
                       help='集合名称（默认: documents）')
    parser.add_argument('--embedding-type', choices=['dashscope', 'local'], 
                       default='dashscope', help='嵌入模型类型（默认: dashscope）')
    parser.add_argument('--n-queries', type=int, default=4, 
                       help='多查询生成的查询数量（默认: 3）')
    parser.add_argument('--retrieval-k', type=int, default=10, 
                       help='每个查询检索的文档数量（默认: 5，用于 RRF 检索中的 k_per_query）')
    parser.add_argument('--max-docs', type=int, 
                       help='生成时使用的最大文档数量（默认: 使用所有检索到的文档）')
    parser.add_argument('--rrf-k', type=int, default=60,
                       help='RRF 算法常数 k，值越小排名影响越大（默认: 60）')
    parser.add_argument('--no-rerank', action='store_true',
                       help='禁用 qwen3-rerank 二次重排序，仅使用 RRF 排序')
    parser.add_argument('--rerank-model', default='qwen3-rerank',
                       help='重排序模型名称（默认: qwen3-rerank）')
    parser.add_argument('--rrf-top-n', type=int, default=10,
                       help='RRF 排序后保留的文档数量，用于 rerank 重排序（默认: 10）')
    parser.add_argument('--rerank-top-n', type=int,default=3,
                       help='rerank 模型返回的文档数量（默认: 使用 top_n 的值）')
    parser.add_argument('--use-scores', action='store_true',
                       help='在生成答案时使用文档分数')
    
    args = parser.parse_args()
    
    # 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    api_url = os.getenv("DASHSCOPE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    if not api_key:
        print("警告: 请设置环境变量 DASHSCOPE_API_KEY")
        print("Windows 示例: set DASHSCOPE_API_KEY=your-api-key")
        print("Linux/Mac 示例: export DASHSCOPE_API_KEY='your-api-key'")
        return
    
    try:
        # 初始化 RAG 管道
        print("=" * 80)
        print("正在初始化 RAG 管道...")
        print("=" * 80)
        pipeline = RAGPipeline(
            persist_directory=args.persist_dir,
            collection_name=args.collection,
            embedding_type=args.embedding_type,
            n_queries=args.n_queries,
            retrieval_k=args.retrieval_k,
            max_docs=args.max_docs,
            api_key=api_key,
            api_url=api_url,
            rrf_k=args.rrf_k,
            use_rerank=not args.no_rerank,
            rerank_model=args.rerank_model,
            rrf_top_n=args.rrf_top_n,
            rerank_top_n=args.rerank_top_n,
            use_scores=args.use_scores
        )
        print("=" * 80)
        print("RAG 管道初始化完成！")
        print("=" * 80)
        
        # 如果提供了问题，直接回答（使用流式输出）
        if args.question:
            print("\n" + "=" * 80)
            print("问题:")
            print("=" * 80)
            print(args.question)
            print("\n" + "=" * 80)
            print("答案（流式输出）:")
            print("=" * 80)
            
            async for chunk in pipeline.query_stream_async(question=args.question):
                print(chunk, end="", flush=True)
            
            print("\n" + "=" * 80)
        else:
            # 交互模式（使用流式输出）
            print("\n进入交互模式，输入 'quit' 或 'exit' 退出，输入 'clear' 清除对话历史")
            print("当前模式: 流式输出（实时生成）")
            print("-" * 80)
            
            while True:
                try:
                    question = input("\n请输入您的问题: ").strip()
                    
                    if not question:
                        continue
                    
                    if question.lower() in ['quit', 'exit', '退出']:
                        print("再见！")
                        break
                    
                    # 支持清除历史命令
                    if question.lower() in ['clear', '清除', '清空', 'clear history']:
                        pipeline.clear_history()
                        print("对话历史已清除，请继续提问")
                        continue
                    
                    # 流式输出模式
                    print("\n" + "=" * 80)
                    print("答案（流式输出）:")
                    print("=" * 80)
                    
                    async for chunk in pipeline.query_stream_async(question=question):
                        print(chunk, end="", flush=True)
                    
                    print("\n" + "=" * 80)
                    
                except KeyboardInterrupt:
                    print("\n\n再见！")
                    break
                except Exception as e:
                    print(f"\n错误: {e}")
                    import traceback
                    traceback.print_exc()
    
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def main():
    """同步包装函数，用于运行异步主函数"""
    return asyncio.run(main_async())


if __name__ == "__main__":
    import sys
    sys.exit(main())

