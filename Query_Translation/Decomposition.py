"""
信息检索系统中的查询分解模块
使用 LangChain 和 LCEL 构建，支持将复杂问题拆分为多个简单的子问题
支持普通调用和流式输出
"""
from langchain_core.prompts import ChatPromptTemplate

# 创建查询分解专用的 ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个信息检索系统中的 Query Decomposition 模块。

给定一个用户问题，请将其拆分为多个更简单、可独立用于检索的子问题。

要求：
1. 保持用户原始意图不变
2. 每个子问题只关注一个明确的信息点
3. 子问题之间应互补，而不是改写同一句话
4. 不要回答问题
5. 不要生成推理过程或解释
6. 子问题应适合直接用于向量或关键词检索
7. 子问题数量控制在 {N} 条以内"""),
    ("user", """用户问题：
{question}

请输出子问题列表：""")
])


def test_decomposition():
    """
    测试函数：包含所有查询分解功能的实现和测试
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

    def parse_subquestions(response_text: str) -> List[str]:
        """
        解析模型输出，提取子问题列表
        
        Args:
            response_text: 模型的原始输出文本
        
        Returns:
            子问题列表
        """
        subquestions = []
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
                subquestions.append(line)
        
        return subquestions

    def decompose_query(question: str, n: int = 3, stream: bool = False) -> List[str]:
        """
        将复杂问题分解为多个简单的子问题
        
        Args:
            question: 用户输入的原始问题
            n: 子问题的最大数量，默认为 3
            stream: 是否使用流式输出（流式模式下返回空列表，直接打印输出）
        
        Returns:
            子问题列表（非流式模式）
        """
        if stream:
            # 流式输出
            print("生成的子问题列表：")
            print("-" * 50)
            full_response = ""
            for chunk in chain.stream({"question": question, "N": n}):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n" + "-" * 50)
            # 解析并返回子问题列表
            return parse_subquestions(full_response)
        else:
            # 普通输出
            response = chain.invoke({"question": question, "N": n})
            return parse_subquestions(response)

    # 执行测试
    # 检查 API Key
    if not DEEPSEEK_API_KEY:
        print("警告: 请设置环境变量 DEEPSEEK_API_KEY")
        print("Windows 示例: set DEEPSEEK_API_KEY=your-api-key")
        print("Linux/Mac 示例: export DEEPSEEK_API_KEY='your-api-key'")
    else:
        # 示例问题
        test_question = "比较 Qwen3-8B 和 LLaMA3 在电商客服场景下的微调成本和效果"
        num_subquestions = 5
        
        print("=" * 50)
        print("查询分解模块 - 普通调用示例")
        print("=" * 50)
        print(f"原始问题: {test_question}")
        print(f"最大子问题数量: {num_subquestions}\n")
        
        subquestions = decompose_query(test_question, n=num_subquestions, stream=False)
        print("生成的子问题列表:")
        for i, subq in enumerate(subquestions, 1):
            print(f"{i}. {subq}")
        
        print("\n" + "=" * 50)
        print("查询分解模块 - 流式输出示例")
        print("=" * 50)
        print(f"原始问题: {test_question}")
        print(f"最大子问题数量: {num_subquestions}\n")
        
        subquestions_stream = decompose_query(test_question, n=num_subquestions, stream=True)
        print(f"\n共生成 {len(subquestions_stream)} 条子问题")


# 执行测试
if __name__ == "__main__":
    test_decomposition()

