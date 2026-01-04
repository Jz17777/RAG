"""
信息检索系统中的查询改写模块
使用 LangChain 和 LCEL 构建，支持生成多个语义相关但表述不同的检索查询
支持普通调用和流式输出
"""
import os
import re
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import dotenv

dotenv.load_dotenv()

# 创建查询改写专用的 ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个信息检索系统中的查询改写模块。

给定一个用户问题，请生成多个语义相关但表述不同的检索查询，
用于提高文档召回率。

要求：
1. 保持原始问题的核心语义不变
2. 使用不同措辞、角度或关键词表达
3. 不要引入原问题中不存在的事实
4. 每个查询应适合直接用于向量检索
5. 不要回答问题本身
6. 输出仅包含查询列表，每行一个查询，不要编号，不要额外说明"""),
    ("user", "用户问题：\n{question}\n\n请生成 {N} 条检索查询。")
])


def parse_queries(response_text: str) -> List[str]:
    """
    解析模型输出，提取查询列表
    
    Args:
        response_text: 模型的原始输出文本
    
    Returns:
        查询列表
    """
    queries = []
    # 按行分割
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        # 移除可能的编号（如 "1. ", "1、", "- " 等）
        line = re.sub(r'^[\d一二三四五六七八九十]+[\.、)\-]\s*', '', line)
        line = re.sub(r'^[-*]\s*', '', line)
        # 移除引号
        line = line.strip('"\'')
        
        if line and len(line) > 0:
            queries.append(line)
    
    return queries


def create_multi_query_chain(
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model: str = "deepseek-chat",
    temperature: float = 0.4,
    top_p: float = 0.9,
    n_queries: int = 3
) -> RunnableLambda:
    """
    创建多查询生成的 LCEL 链
    
    Args:
        api_key: API Key（如果为 None，从环境变量读取）
        api_url: API URL（如果为 None，从环境变量读取）
        model: 模型名称，默认为 "deepseek-chat"
        temperature: 温度参数，默认为 0.4
        top_p: top_p 参数，默认为 0.9
        n_queries: 生成的查询数量，默认为 3
    
    Returns:
        LCEL 链，输入为 {"question": str}，输出为 List[str]
    """
    # 从环境变量获取 API Key 和 URL（如果未提供）
    if api_key is None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if api_url is None:
        api_url = os.getenv("DEEPSEEK_API_URL", "")
    
    # 创建模型实例
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=api_url,
        temperature=temperature,
        top_p=top_p,
    )
    
    # 创建解析函数，将字符串解析为查询列表
    def parse_output(response: str) -> List[str]:
        return parse_queries(response)
    
    # 构建完整的 LCEL 链
    # 输入格式: {"question": "用户问题"}
    # 输出格式: List[str] (查询列表)
    chain = (
        prompt.partial(N=n_queries)  # 固定 N 参数
        | llm
        | StrOutputParser()
        | RunnableLambda(parse_output)
    )
    
    return chain


def test_multi_query():
    """
    测试函数：包含所有查询改写功能的实现和测试
    """
    # 从环境变量获取 API Key
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "")
    
    # 执行测试
    # 检查 API Key
    if not DEEPSEEK_API_KEY:
        print("警告: 请设置环境变量 DEEPSEEK_API_KEY")
        print("Windows 示例: set DEEPSEEK_API_KEY=your-api-key")
        print("Linux/Mac 示例: export DEEPSEEK_API_KEY='your-api-key'")
    else:
        # 创建 LCEL 链
        chain = create_multi_query_chain(n_queries=5)
        
        # 示例问题
        test_question = "什么是机器学习？"
        
        print("=" * 50)
        print("查询改写模块 - LCEL 链调用示例")
        print("=" * 50)
        print(f"原始问题: {test_question}\n")
        
        queries = chain.invoke({"question": test_question})
        print("生成的检索查询列表:")
        for i, query in enumerate(queries, 1):
            print(f"{i}. {query}")
        
        print("\n" + "=" * 50)
        print("查询改写模块 - 流式输出示例")
        print("=" * 50)
        print(f"原始问题: {test_question}\n")
        print("生成的检索查询：")
        print("-" * 50)
        queries_stream = []
        for chunk in chain.stream({"question": test_question}):
            if isinstance(chunk, list):
                queries_stream = chunk
                for i, q in enumerate(chunk, 1):
                    print(f"{i}. {q}")
            else:
                print(chunk, end="", flush=True)
        print("\n" + "-" * 50)
        print(f"\n共生成 {len(queries_stream)} 条查询")


# 执行测试
if __name__ == "__main__":
    test_multi_query()
