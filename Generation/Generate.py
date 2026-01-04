"""
基于检索增强生成（RAG）的答案生成模块
使用 LangChain 和 LCEL 构建，支持基于检索文档生成答案
"""
import os
import asyncio
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import Runnable
from langchain_core.documents import Document
import dotenv

dotenv.load_dotenv()

# 创建 RAG 生成专用的 ChatPromptTemplate（支持对话历史）
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个基于检索增强生成（RAG）的专业问答系统。

请基于【参考文档】回答【用户问题】，并遵守以下规则：

【规则】
1. 仅使用参考文档中的信息
2. 不允许引入外部常识或猜测
3. 若文档信息不足，请直接说明"不足以回答"
4. 给出清晰、逻辑自洽的答案
5. 每一条关键结论都必须能在文档中找到依据
6. 如果提供了【对话历史】，请结合历史上下文理解当前问题

【对话历史】
{chat_history}

【用户问题】
{question}

【参考文档】
{context}

【回答（中文）】"""),
])


def format_context(documents: List[Document], max_docs: Optional[int] = None) -> str:
    """
    格式化文档列表为上下文字符串
    
    Args:
        documents: 文档列表
        max_docs: 最大文档数量（如果为 None，使用所有文档）
    
    Returns:
        格式化后的上下文字符串
    """
    if not documents:
        return "无相关文档"
    
    # 如果指定了最大文档数量，只取前 n 个
    docs_to_use = documents[:max_docs] if max_docs else documents
    
    context_parts = []
    for i, doc in enumerate(docs_to_use, 1):
        content = doc.page_content.strip()
        metadata_str = ""
        if doc.metadata:
            # 格式化元数据
            metadata_items = [f"{k}: {v}" for k, v in doc.metadata.items()]
            metadata_str = f"（元数据: {', '.join(metadata_items)}）"
        context_parts.append(f"文档 {i}{metadata_str}:\n{content}\n")
    
    return "\n".join(context_parts)


def format_context_with_scores(
    results: List[tuple], 
    max_docs: Optional[int] = None
) -> str:
    """
    格式化带分数的文档列表为上下文字符串
    
    Args:
        results: (文档, 分数) 元组列表
        max_docs: 最大文档数量（如果为 None，使用所有文档）
    
    Returns:
        格式化后的上下文字符串
    """
    if not results:
        return "无相关文档"
    
    # 如果指定了最大文档数量，只取前 n 个
    results_to_use = results[:max_docs] if max_docs else results
    
    context_parts = []
    for i, (doc, score) in enumerate(results_to_use, 1):
        content = doc.page_content.strip()
        metadata_str = ""
        if doc.metadata:
            # 格式化元数据
            metadata_items = [f"{k}: {v}" for k, v in doc.metadata.items()]
            metadata_str = f"（元数据: {', '.join(metadata_items)}）"
        context_parts.append(
            f"文档 {i}（相似度分数: {score:.4f}）{metadata_str}:\n{content}\n"
        )
    
    return "\n".join(context_parts)


async def create_multi_question_rag_generation_chain_async(
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model: str = "qwen-flash",
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_docs: Optional[int] = None,
    with_scores: bool = False,
    enable_chat_history: bool = True
) -> Runnable:
    """
    创建支持多问题的异步 RAG 生成链
    将多个问题都放入 prompt 中
    
    Args:
        api_key: API Key（如果为 None，从环境变量读取）
        api_url: API URL（如果为 None，从环境变量读取）
        model: 模型名称，默认为 "qwen-flash"
        temperature: 温度参数，默认为 0.3
        top_p: top_p 参数，默认为 0.9
        max_docs: 最大文档数量（如果为 None，使用所有文档）
        with_scores: 输入是否包含分数，默认为 False
        enable_chat_history: 是否启用对话历史，默认为 True
    
    Returns:
        LCEL 链，支持异步调用，输入为 {"original_question": str, "queries": List[str], 
        "documents": List[Document], "chat_history": List[tuple]} 或 
        {"original_question": str, "queries": List[str], "results": List[tuple],
        "chat_history": List[tuple]}，输出为 str（生成的答案）
    """
    # 从环境变量获取 API Key 和 URL（如果未提供）
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if api_url is None:
        api_url = os.getenv("DASHSCOPE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    # 创建模型实例
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=api_url,
        temperature=temperature,
        top_p=top_p,
    )
    
    async def generate_answer_with_queries_async(input_dict: dict) -> str:
        """
        基于多个查询异步生成答案
        
        Args:
            input_dict: 包含 original_question, queries 和 documents/results 的字典
        
        Returns:
            生成的答案字符串
        """
        original_question = input_dict.get("original_question", "")
        queries = input_dict.get("queries", [])
        
        if not original_question:
            return "未提供问题"
        
        # 获取对话历史
        chat_history = input_dict.get("chat_history", [])
        if enable_chat_history and chat_history:
            # 格式化对话历史
            history_text = ""
            for i, (user_msg, assistant_msg) in enumerate(chat_history[-5:], 1):  # 只保留最近5轮对话
                history_text += f"第{i}轮对话：\n"
                history_text += f"用户：{user_msg}\n"
                history_text += f"助手：{assistant_msg}\n\n"
        else:
            history_text = "无对话历史"
        
        # 格式化问题部分（包含原始问题和生成的查询）
        if queries:
            questions_text = f"原始问题：{original_question}\n\n相关查询：\n"
            for i, query in enumerate(queries, 1):
                questions_text += f"{i}. {query}\n"
            question = questions_text
        else:
            question = original_question
        
        # 格式化上下文
        if with_scores:
            # 输入是带分数的结果
            results = input_dict.get("results", [])
            if not results:
                return "无相关文档，不足以回答"
            context = format_context_with_scores(results, max_docs=max_docs)
        else:
            # 输入是文档列表
            documents = input_dict.get("documents", [])
            if not documents:
                return "无相关文档，不足以回答"
            context = format_context(documents, max_docs=max_docs)
        
        # 异步调用 LLM 生成答案
        try:
            response = await llm.ainvoke(rag_prompt.format(
                question=question, 
                context=context,
                chat_history=history_text
            ))
            
            # 提取答案内容
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        except Exception as e:
            # 处理内容审核错误
            error_str = str(e)
            if 'data_inspection_failed' in error_str or 'inappropriate content' in error_str.lower():
                return "抱歉，输入内容可能包含敏感信息，无法生成回答。请尝试重新表述您的问题，或使用其他查询方式。"
            # 处理其他 API 错误
            elif 'BadRequestError' in str(type(e)) or '400' in error_str:
                return f"API 请求错误：{error_str[:200]}。请检查输入内容或稍后重试。"
            else:
                # 其他未知错误
                return f"生成答案时发生错误：{error_str[:200]}。请稍后重试。"
    
    # 包装为异步函数
    async def async_wrapper(input_dict: dict) -> str:
        return await generate_answer_with_queries_async(input_dict)
    
    # 使用自定义的异步包装器
    class AsyncRunnableWrapper:
        def __init__(self, async_func):
            self.async_func = async_func
        
        async def ainvoke(self, input_dict: dict, config=None):
            return await self.async_func(input_dict)
        
        def invoke(self, input_dict: dict, config=None):
            # 同步调用时使用 asyncio.run
            return asyncio.run(self.async_func(input_dict))
    
    return AsyncRunnableWrapper(async_wrapper)


def create_multi_question_rag_generation_stream_async(
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model: str = "qwen-flash",
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_docs: Optional[int] = None,
    with_scores: bool = False,
    enable_chat_history: bool = True
):
    """
    创建支持多问题的异步流式 RAG 生成器
    返回一个异步生成器函数，该函数可以实时流式输出答案
    
    Args:
        api_key: API Key（如果为 None，从环境变量读取）
        api_url: API URL（如果为 None，从环境变量读取）
        model: 模型名称，默认为 "qwen-flash"
        temperature: 温度参数，默认为 0.3
        top_p: top_p 参数，默认为 0.9
        max_docs: 最大文档数量（如果为 None，使用所有文档）
        with_scores: 输入是否包含分数，默认为 False
        enable_chat_history: 是否启用对话历史，默认为 True
    
    Yields:
        生成的答案文本块（字符串）
    """
    # 从环境变量获取 API Key 和 URL（如果未提供）
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if api_url is None:
        api_url = os.getenv("DASHSCOPE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    # 创建模型实例
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=api_url,
        temperature=temperature,
        top_p=top_p,
        streaming=True,  # 启用流式输出
    )
    
    async def generate_answer_stream_async(input_dict: dict):
        """
        基于多个查询异步流式生成答案
        
        Args:
            input_dict: 包含 original_question, queries 和 documents/results 的字典
        
        Yields:
            生成的答案文本块
        """
        original_question = input_dict.get("original_question", "")
        queries = input_dict.get("queries", [])
        
        if not original_question:
            yield "未提供问题"
            return
        
        # 获取对话历史
        chat_history = input_dict.get("chat_history", [])
        if enable_chat_history and chat_history:
            # 格式化对话历史
            history_text = ""
            for i, (user_msg, assistant_msg) in enumerate(chat_history[-5:], 1):  # 只保留最近5轮对话
                history_text += f"第{i}轮对话：\n"
                history_text += f"用户：{user_msg}\n"
                history_text += f"助手：{assistant_msg}\n\n"
        else:
            history_text = "无对话历史"
        
        # 格式化问题部分（包含原始问题和生成的查询）
        if queries:
            questions_text = f"原始问题：{original_question}\n\n相关查询：\n"
            for i, query in enumerate(queries, 1):
                questions_text += f"{i}. {query}\n"
            question = questions_text
        else:
            question = original_question
        
        # 格式化上下文
        if with_scores:
            # 输入是带分数的结果
            results = input_dict.get("results", [])
            if not results:
                yield "无相关文档，不足以回答"
                return
            context = format_context_with_scores(results, max_docs=max_docs)
        else:
            # 输入是文档列表
            documents = input_dict.get("documents", [])
            if not documents:
                yield "无相关文档，不足以回答"
                return
            context = format_context(documents, max_docs=max_docs)
        
        # 异步流式调用 LLM 生成答案
        try:
            formatted_prompt = rag_prompt.format(
                question=question, 
                context=context,
                chat_history=history_text
            )
            
            # 使用 astream 进行流式生成
            async for chunk in llm.astream(formatted_prompt):
                # 提取每个块的内容
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    if content:
                        yield content
                else:
                    # 如果 chunk 是字符串
                    if chunk:
                        yield str(chunk)
        except Exception as e:
            # 处理内容审核错误
            error_str = str(e)
            if 'data_inspection_failed' in error_str or 'inappropriate content' in error_str.lower():
                yield "抱歉，输入内容可能包含敏感信息，无法生成回答。请尝试重新表述您的问题，或使用其他查询方式。"
            # 处理其他 API 错误
            elif 'BadRequestError' in str(type(e)) or '400' in error_str:
                yield f"API 请求错误：{error_str[:200]}。请检查输入内容或稍后重试。"
            else:
                # 其他未知错误
                yield f"生成答案时发生错误：{error_str[:200]}。请稍后重试。"
    
    return generate_answer_stream_async


def test_rag_generation():
    """
    测试函数：测试 RAG 生成功能
    """
    from langchain_core.documents import Document
    
    # 从环境变量获取 API Key
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    
    if not DASHSCOPE_API_KEY:
        print("警告: 请设置环境变量 DASHSCOPE_API_KEY")
    else:
        # 创建测试文档
        test_docs = [
            Document(
                page_content="机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。",
                metadata={"source": "test1"}
            ),
            Document(
                page_content="深度学习是机器学习的一个子领域，使用神经网络进行学习。",
                metadata={"source": "test2"}
            )
        ]
        
        # 创建异步生成链并测试
        async def test_async():
            chain = await create_multi_question_rag_generation_chain_async(max_docs=2)
            result = await chain.ainvoke({
                "original_question": "什么是机器学习？",
                "queries": ["机器学习是什么", "机器学习的定义"],
                "documents": test_docs,
                "chat_history": []
            })
            return result
        
        # 运行异步测试
        result = asyncio.run(test_async())
        
        print("=" * 50)
        print("RAG 生成测试")
        print("=" * 50)
        print(f"问题: 什么是机器学习？")
        print(f"生成的答案:\n{result}")


# 执行测试
if __name__ == "__main__":
    test_rag_generation()

