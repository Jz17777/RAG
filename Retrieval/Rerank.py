"""
向量检索模块 - 使用 Multi-query 策略、RRF 重排序和 qwen3-rerank 模型

主要功能：
- Multi-query 策略（自动生成多个相关查询）
- RRF 重排序（融合多个查询结果，提高检索质量）
- qwen3-rerank 模型二次重排序（进一步提升检索精度）
- 返回带分数的检索结果

使用方法：
    from Retrieval.Rerank import VectorRetriever
    
    retriever = VectorRetriever(
        persist_directory="./chroma_db",
        embedding_type="dashscope"
    )
    
    results = retriever.retrieve_with_multi_query_rrf_with_scores(
        question="什么是机器学习？",
        n_queries=3,
        k_per_query=5,
        top_n=10,
        rrf_k=60,
        use_rerank=True
    )
"""
from typing import List, Optional, Union, Any
from pathlib import Path
import os
import requests

# 添加项目根目录到 Python 路径，以便导入 Indexing 模块
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
import sys
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入 Chunk_Optimization 中的嵌入模型函数
from Indexing.Chunk_Optimization import load_chroma_vectorstore
from langchain_core.documents import Document

# 导入 Multi-query 模块
import importlib.util
multi_query_path = project_root / "Query_Translation" / "Multi-query.py"
if multi_query_path.exists():
    spec = importlib.util.spec_from_file_location("Multi_query", multi_query_path)
    multi_query_module = importlib.util.module_from_spec(spec)
    sys.modules["Multi_query"] = multi_query_module
    spec.loader.exec_module(multi_query_module)
    create_multi_query_chain = multi_query_module.create_multi_query_chain
else:
    create_multi_query_chain = None


class VectorRetriever:
    """
    向量检索器类 - 使用 Multi-query 策略、RRF 重排序和 qwen3-rerank 模型
    """
    
    def __init__(self,
                 persist_directory: Union[str, Path] = "./chroma_db",
                 collection_name: str = "documents",
                 embedding_type: str = "dashscope",
                 embedding_model: Optional[str] = None,
                 multi_query_chain: Optional[Any] = None):
        """
        初始化向量检索器
        
        Args:
            persist_directory: Chroma 数据库持久化目录路径，默认为 "./chroma_db"
            collection_name: 集合名称，默认为 "documents"
            embedding_type: 嵌入模型类型，"dashscope" 或 "local"，默认为 "dashscope"
            embedding_model: 模型名称（可选，使用默认值）
            multi_query_chain: 多查询生成链（可选，如果不提供则在使用时创建）
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_type = embedding_type
        self.embedding_model = embedding_model
        self._multi_query_chain = multi_query_chain
        
        # 加载向量数据库
        try:
            self.vector_store = load_chroma_vectorstore(
                persist_directory=str(self.persist_directory),
                collection_name=self.collection_name,
                embedding_type=self.embedding_type,
                embedding_model=self.embedding_model
            )
            print(f"向量检索器初始化成功")
        except Exception as e:
            raise Exception(f"初始化向量检索器失败: {str(e)}")
    
    def retrieve(self,
                 query: str,
                 k: int = 5) -> List[Document]:
        """
        执行向量检索（内部方法，用于 RRF 检索）
        
        Args:
            query: 查询文本
            k: 返回的文档数量，默认为 5
            
        Returns:
            检索到的文档列表
        """
        if not query or not query.strip():
            return []
        
        results = self.vector_store.similarity_search(
            query=query,
            k=k
        )
        
        return results
    
    def _get_multi_query_chain(self, n_queries: int = 3):
        """
        获取或创建多查询生成链
        
        Args:
            n_queries: 生成的查询数量
            
        Returns:
            多查询生成链
        """
        if self._multi_query_chain is not None:
            return self._multi_query_chain
        
        if create_multi_query_chain is None:
            raise ImportError("无法导入 Multi-query 模块，请确保 Query_Translation/Multi-query.py 存在")
        
        # 从环境变量获取 API 配置
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        api_url = os.getenv("DASHSCOPE_API_URL") or os.getenv("DEEPSEEK_API_URL")
        # 从环境变量获取模型名称，如果没有则使用默认值
        model = os.getenv("MULTI_QUERY_MODEL", "qwen-flash")
        
        return create_multi_query_chain(
            api_key=api_key,
            api_url=api_url,
            model=model,
            n_queries=n_queries
        )
    
    def _rrf_score(self, doc_content: str, query_results: List[List[Document]], k: int = 60) -> float:
        """
        计算文档的 RRF (Reciprocal Rank Fusion) 分数
        
        Args:
            doc_content: 文档内容（用于匹配）
            query_results: 每个查询的检索结果列表
            k: RRF 常数，默认为 60
            
        Returns:
            RRF 分数
        """
        rrf_score = 0.0
        
        for query_docs in query_results:
            # 在当前查询结果中查找文档的排名
            for rank, doc in enumerate(query_docs, start=1):
                # 使用内容的前100个字符进行匹配（与去重逻辑一致）
                if doc.page_content[:100] == doc_content[:100]:
                    rrf_score += 1.0 / (k + rank)
                    break
        
        return rrf_score
    
    def _rerank_with_qwen3(self, 
                          query: str, 
                          documents: List[Document],
                          api_key: Optional[str] = None,
                          api_url: Optional[str] = None,
                          model: str = "qwen3-rerank",
                          top_n: Optional[int] = None) -> List[tuple]:
        """
        使用 qwen3-rerank 模型对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            api_key: API 密钥（如果为 None，从环境变量读取）
            api_url: API 地址（如果为 None，从环境变量读取）
            model: 重排序模型名称，默认为 "qwen3-rerank"
            top_n: 返回的文档数量（如果为 None，返回所有文档）
            
        Returns:
            (文档, rerank分数) 元组列表，按 rerank 分数降序排列
        """
        if not documents:
            return []
        
        # 从环境变量获取 API 配置
        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        if api_url is None:
            # 使用官方 rerank API 地址
            api_url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        
        if not api_key:
            print("警告: 未设置 DASHSCOPE_API_KEY，跳过 qwen3-rerank 重排序")
            # 返回原始文档，分数设为 0
            return [(doc, 0.0) for doc in documents]
        
        try:
            # 准备请求数据
            # 提取文档内容（限制长度以避免超出 API 限制）
            doc_texts = [doc.page_content[:2000] for doc in documents]  # 限制每个文档长度
            
            # 构建请求（按照官方 API 格式）
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建请求体，按照官方 API 格式
            payload = {
                "model": model,
                "input": {
                    "query": query,
                    "documents": doc_texts
                },
                "parameters": {
                    "return_documents": True,
                    "top_n": top_n if top_n is not None else len(documents)
                }
            }
            
            # 发送请求
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # 解析结果（根据官方 API 返回格式）
            # 官方 API 返回格式: {"output": {"results": [{"index": 0, "relevance_score": 0.95, "document": "..."}, ...]}}
            if "output" in result:
                output = result["output"]
                rerank_results = output.get("results", [])
                
                if not rerank_results:
                    print(f"警告: qwen3-rerank API 返回结果为空")
                    return [(doc, 0.0) for doc in documents]
                
                doc_scores = []
                
                # 遍历 rerank 结果
                for rerank_item in rerank_results:
                    # 官方 API 返回格式包含 index 和 relevance_score
                    index = rerank_item.get("index", -1)
                    # 尝试获取分数（可能是 relevance_score 或 score）
                    score = rerank_item.get("relevance_score", rerank_item.get("score", 0.0))
                    
                    # 如果返回了文档内容，也可以使用（但通常使用 index 更可靠）
                    returned_doc = rerank_item.get("document", None)
                    
                    if 0 <= index < len(documents):
                        # 使用索引匹配文档（最可靠的方式）
                        doc_scores.append((documents[index], float(score)))
                    elif returned_doc:
                        # 如果返回了文档内容但没有索引，尝试通过内容匹配
                        for i, doc in enumerate(documents):
                            if doc.page_content[:100] == returned_doc[:100]:
                                doc_scores.append((doc, float(score)))
                                break
                
                # 按分数降序排序（rerank API 通常已经排序，但为了确保）
                doc_scores.sort(key=lambda x: x[1], reverse=True)
                
                # 返回 TOP N（如果指定了 top_n，API 已经返回了 top_n 个结果，但再次限制以确保）
                if top_n is not None and len(doc_scores) > top_n:
                    return doc_scores[:top_n]
                return doc_scores
            else:
                print(f"警告: qwen3-rerank API 返回格式异常，缺少 'output' 字段: {result}")
                # 返回原始文档，分数设为 0
                return [(doc, 0.0) for doc in documents]
                
        except requests.exceptions.RequestException as e:
            print(f"警告: qwen3-rerank API 调用失败: {e}，跳过重排序")
            # 返回原始文档，分数设为 0
            return [(doc, 0.0) for doc in documents]
        except Exception as e:
            print(f"警告: qwen3-rerank 重排序时出错: {e}，跳过重排序")
            # 返回原始文档，分数设为 0
            return [(doc, 0.0) for doc in documents]
    
    def retrieve_with_multi_query_rrf_with_scores(self,
                                                 question: str,
                                                 n_queries: int = 3,
                                                 k_per_query: int = 5,
                                                 top_n: int = 10,
                                                 rrf_k: int = 60,
                                                 use_original_query: bool = True,
                                                 use_rerank: bool = True,
                                                 rerank_model: str = "qwen3-rerank",
                                                 rerank_api_key: Optional[str] = None,
                                                 rerank_api_url: Optional[str] = None,
                                                 rrf_top_n: Optional[int] = None,
                                                 rerank_top_n: Optional[int] = None) -> List[tuple]:
        """
        使用 multi-query 策略、RRF 重排序和 qwen3-rerank 模型进行检索
        
        Args:
            question: 原始问题
            n_queries: 生成的额外查询数量，默认为 3
            k_per_query: 每个查询返回的文档数量，默认为 5
            top_n: 最终返回的文档数量，默认为 10
            rrf_k: RRF 算法常数 k，默认为 60
            use_original_query: 是否将原始问题也作为查询之一，默认为 True
            use_rerank: 是否使用 qwen3-rerank 模型进行二次重排序，默认为 True
            rerank_model: 重排序模型名称，默认为 "qwen3-rerank"
            rerank_api_key: 重排序 API 密钥（如果为 None，从环境变量读取）
            rerank_api_url: 重排序 API 地址（如果为 None，从环境变量读取）
            rrf_top_n: RRF 排序后保留的文档数量，默认为 None（保留所有文档）
            rerank_top_n: rerank 模型返回的文档数量，默认为 None（使用 top_n 的值）
            
        Returns:
            (文档, 分数) 元组列表，按分数降序排列
        """
        if not question or not question.strip():
            return []
        
        # 步骤 1: 生成多个查询
        try:
            multi_query_chain = self._get_multi_query_chain(n_queries=n_queries)
            # 同步调用
            queries = multi_query_chain.invoke({"question": question})
        except Exception as e:
            print(f"生成多查询时出错: {e}，仅使用原始问题")
            queries = []
        
        # 展示生成的查询语句
        if queries:
            print(f"\n生成了 {len(queries)} 个改写查询:")
            for i, q in enumerate(queries, 1):
                print(f"  查询 {i}: {q}")
        
        # 构建所有查询列表
        all_queries = [question] + queries if use_original_query else queries
        
        if use_original_query and queries:
            print(f"共使用 {len(all_queries)} 个查询进行检索（包含原始问题）")
        elif queries:
            print(f"共使用 {len(all_queries)} 个查询进行检索")
        
        if not all_queries:
            return []
        
        # 步骤 2: 对每个查询进行检索
        query_results = []
        for query in all_queries:
            if not query or not query.strip():
                continue
            docs = self.retrieve(query=query, k=k_per_query)
            query_results.append(docs)
        
        if not query_results:
            return []
        
        # 步骤 3: 收集所有唯一文档
        all_docs_dict = {}  # 使用内容前100字符作为键
        for query_docs in query_results:
            for doc in query_docs:
                content_key = doc.page_content[:100]
                if content_key not in all_docs_dict:
                    all_docs_dict[content_key] = doc
        
        # 步骤 4: 计算每个文档的 RRF 分数
        doc_scores = []
        for content_key, doc in all_docs_dict.items():
            rrf_score = self._rrf_score(doc.page_content, query_results, k=rrf_k)
            doc_scores.append((doc, rrf_score))
        
        # 步骤 5: 按 RRF 分数降序排序
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 步骤 5.5: 如果指定了 rrf_top_n，只保留前 rrf_top_n 个文档用于 rerank
        if rrf_top_n is not None and rrf_top_n > 0:
            rrf_kept_docs = doc_scores[:rrf_top_n]
            print(f"RRF 排序完成，保留前 {len(rrf_kept_docs)} 个文档用于 rerank 重排序")
        else:
            rrf_kept_docs = doc_scores
            if use_rerank:
                print(f"RRF 排序完成，共 {len(rrf_kept_docs)} 个文档将进行 rerank 重排序")
        
        # 步骤 6: 使用 qwen3-rerank 进行二次重排序（如果启用）
        if use_rerank and rrf_kept_docs:
            print(f"\n正在使用 qwen3-rerank 模型进行二次重排序...")
            # 提取文档列表
            rrf_docs = [doc for doc, _ in rrf_kept_docs]
            
            # 确定 rerank 返回的文档数量
            rerank_return_n = rerank_top_n if rerank_top_n is not None else top_n
            
            # 调用 rerank API
            rerank_results = self._rerank_with_qwen3(
                query=question,
                documents=rrf_docs,
                api_key=rerank_api_key,
                api_url=rerank_api_url,
                model=rerank_model,
                top_n=rerank_return_n
            )
            
            # 判断 rerank 是否成功：如果有结果且结果数量与输入文档数量一致（或更多），认为成功
            if rerank_results and len(rerank_results) > 0:
                # 检查是否有有效的分数（至少有一个分数 > 0）
                has_valid_scores = any(score > 0 for _, score in rerank_results)
                if has_valid_scores:
                    # rerank 成功，使用 rerank 结果
                    print(f"qwen3-rerank 重排序完成，返回 {len(rerank_results)} 个文档")
                    # 如果指定了 rerank_top_n 且与 top_n 不同，需要截取前 top_n 个
                    if rerank_top_n is not None and rerank_top_n != top_n:
                        return rerank_results[:top_n]
                    return rerank_results
                else:
                    # rerank 返回了结果但分数都为 0，可能是 API 格式问题，回退到 RRF
                    print(f"警告: qwen3-rerank 返回的分数无效，使用 RRF 排序结果")
            else:
                # rerank 失败，回退到 RRF 结果
                print(f"qwen3-rerank 重排序失败，使用 RRF 排序结果")
        
        # 步骤 7: 返回 TOP N（带分数）
        # 如果不使用 rerank，返回 doc_scores 的前 top_n 个
        # 如果使用 rerank 但失败了，返回 rrf_kept_docs 的前 top_n 个（如果指定了 rrf_top_n）
        # 注意：如果 rerank 成功，已经在步骤 6 中返回了
        if not use_rerank:
            # 不使用 rerank，直接返回 RRF 排序结果
            return doc_scores[:top_n]
        else:
            # 使用 rerank 但失败了，返回保留的文档（如果指定了 rrf_top_n，则只返回保留的）
            return rrf_kept_docs[:top_n]
