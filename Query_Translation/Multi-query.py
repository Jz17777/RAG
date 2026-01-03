"""
信息检索系统中的查询改写模块
使用 LangChain 和 LCEL 构建，支持生成多个语义相关但表述不同的检索查询
支持普通调用和流式输出
"""
from langchain_core.prompts import ChatPromptTemplate

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


def test_multi_query():
    """
    测试函数：包含所有查询改写功能的实现和测试
    """
    import os
    import re
    from typing import List
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    import dotenv

    dotenv.load_dotenv()

    # 从环境变量获取 API Key
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "")

    # 创建模型实例
    # DeepSeek 使用 OpenAI 兼容的 API
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_API_URL,
        temperature=0.4,
        top_p=0.9,
    )

    # 使用 LCEL 构建可执行的链
    chain = prompt | llm | StrOutputParser()

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

    def generate_queries(question: str, n: int = 3, stream: bool = False) -> List[str]:
        """
        生成多个语义相关但表述不同的检索查询
        
        Args:
            question: 用户输入的原始问题
            n: 需要生成的查询数量，默认为 3
            stream: 是否使用流式输出（流式模式下返回空列表，直接打印输出）
        
        Returns:
            查询列表（非流式模式）
        """
        if stream:
            # 流式输出
            print("生成的检索查询：")
            print("-" * 50)
            full_response = ""
            for chunk in chain.stream({"question": question, "N": n}):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n" + "-" * 50)
            # 解析并返回查询列表
            return parse_queries(full_response)
        else:
            # 普通输出
            response = chain.invoke({"question": question, "N": n})
            return parse_queries(response)

    # 执行测试
    # 检查 API Key
    if not DEEPSEEK_API_KEY:
        print("警告: 请设置环境变量 DEEPSEEK_API_KEY")
        print("Windows 示例: set DEEPSEEK_API_KEY=your-api-key")
        print("Linux/Mac 示例: export DEEPSEEK_API_KEY='your-api-key'")
    else:
        # 示例问题
        test_question = "什么是机器学习？"
        num_queries = 5
        
        print("=" * 50)
        print("查询改写模块 - 普通调用示例")
        print("=" * 50)
        print(f"原始问题: {test_question}")
        print(f"生成查询数量: {num_queries}\n")
        
        queries = generate_queries(test_question, n=num_queries, stream=False)
        print("生成的检索查询列表:")
        for i, query in enumerate(queries, 1):
            print(f"{i}. {query}")
        
        print("\n" + "=" * 50)
        print("查询改写模块 - 流式输出示例")
        print("=" * 50)
        print(f"原始问题: {test_question}")
        print(f"生成查询数量: {num_queries}\n")
        
        queries_stream = generate_queries(test_question, n=num_queries, stream=True)
        print(f"\n共生成 {len(queries_stream)} 条查询")


# 执行测试
if __name__ == "__main__":
    test_multi_query()
