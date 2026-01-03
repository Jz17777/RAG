"""
信息检索系统中的查询抽象（Step-back）模块
使用 LangChain 和 LCEL 构建，支持将具体问题改写为更抽象、更通用的问题
支持普通调用和流式输出
"""
from langchain_core.prompts import ChatPromptTemplate

# 创建查询抽象专用的 ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个信息检索系统中的 Query Abstraction（Step-back）模块。

给定一个用户问题，请将其改写为一个更抽象、更通用的问题，
用于检索背景性或总结性文档。

要求：
1. 保持用户原始意图不变
2. 提升问题的抽象层级（从具体场景退回到通用概念）
3. 不引入新的条件或假设
4. 不拆分为多个问题
5. 不回答问题本身
6. 输出应适合直接用于检索"""),
    ("user", """用户问题：
{question}

请输出一个 Step-back 后的问题：""")
])


def test_step_back():
    """
    测试函数：包含所有查询抽象功能的实现和测试
    """
    import os
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

    def clean_step_back_question(response_text: str) -> str:
        """
        清理模型输出，提取 Step-back 问题
        
        Args:
            response_text: 模型的原始输出文本
        
        Returns:
            清理后的 Step-back 问题
        """
        # 移除首尾空白
        question = response_text.strip()
        
        # 移除可能的引号
        question = question.strip('"\'')
        
        # 如果输出包含多行，通常第一行或最后一行是问题本身
        # 移除可能的说明性文字（如"Step-back 后的问题："等）
        lines = question.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # 跳过明显的说明性文字
            if any(keyword in line for keyword in ['Step-back', 'step-back', '抽象问题', '改写后']):
                if ':' in line or '：' in line:
                    # 提取冒号后的内容
                    parts = line.split(':', 1) if ':' in line else line.split('：', 1)
                    if len(parts) > 1:
                        line = parts[1].strip()
                    else:
                        continue
                else:
                    continue
            if line:
                cleaned_lines.append(line)
        
        # 如果有多行，取最长的非空行作为问题
        if cleaned_lines:
            question = max(cleaned_lines, key=len)
        
        return question.strip()

    def step_back_query(question: str, stream: bool = False) -> str:
        """
        将具体问题改写为更抽象、更通用的问题
        
        Args:
            question: 用户输入的原始问题
            stream: 是否使用流式输出（流式模式下返回空字符串，直接打印输出）
        
        Returns:
            Step-back 后的抽象问题（非流式模式）
        """
        if stream:
            # 流式输出
            print("生成的 Step-back 问题：")
            print("-" * 50)
            full_response = ""
            for chunk in chain.stream({"question": question}):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n" + "-" * 50)
            # 清理并返回问题
            return clean_step_back_question(full_response)
        else:
            # 普通输出
            response = chain.invoke({"question": question})
            return clean_step_back_question(response)

    # 执行测试
    # 检查 API Key
    if not DEEPSEEK_API_KEY:
        print("警告: 请设置环境变量 DEEPSEEK_API_KEY")
        print("Windows 示例: set DEEPSEEK_API_KEY=your-api-key")
        print("Linux/Mac 示例: export DEEPSEEK_API_KEY='your-api-key'")
    else:
        # 示例问题
        test_question = "比较 Qwen3-8B 和 LLaMA3 在电商客服场景下的微调成本和效果"
        
        print("=" * 50)
        print("查询抽象模块 - 普通调用示例")
        print("=" * 50)
        print(f"原始问题: {test_question}\n")
        
        step_back_question = step_back_query(test_question, stream=False)
        print(f"Step-back 后的问题: {step_back_question}")
        
        print("\n" + "=" * 50)
        print("查询抽象模块 - 流式输出示例")
        print("=" * 50)
        print(f"原始问题: {test_question}\n")
        
        step_back_question_stream = step_back_query(test_question, stream=True)
        print(f"\n最终 Step-back 问题: {step_back_question_stream}")


# 执行测试
if __name__ == "__main__":
    test_step_back()

