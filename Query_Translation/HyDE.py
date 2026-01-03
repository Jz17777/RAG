"""
信息检索系统中的 HyDE（Hypothetical Document Embeddings）模块
使用 LangChain 和 LCEL 构建，支持生成假设性文档片段用于向量检索
支持普通调用和流式输出
"""
from langchain_core.prompts import ChatPromptTemplate

# 创建 HyDE 专用的 ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个信息检索系统中的 HyDE 模块。

给定一个用户问题，请生成一段"可能存在于文档中的答案性文本"，
用于后续向量检索。

要求：
1. 不要求内容完全正确
2. 使用陈述性、说明性语气
3. 结构接近真实文档或说明文字
4. 不包含对话形式或提问句
5. 不引用具体来源或链接
6. 不生成结论性总结
7. 不回答为用户服务的语气"""),
    ("user", """用户问题：
{question}

请生成一段假想文档：""")
])


def test_hyde():
    """
    测试函数：包含所有 HyDE 功能的实现和测试
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
        temperature=0.7,  # HyDE 需要一定的创造性，温度稍高
        top_p=0.9,
    )

    # 使用 LCEL 构建可执行的链
    chain = prompt | llm | StrOutputParser()

    def clean_hypothetical_document(response_text: str) -> str:
        """
        清理模型输出，提取假设性文档
        
        Args:
            response_text: 模型的原始输出文本
        
        Returns:
            清理后的假设性文档文本
        """
        # 移除首尾空白
        document = response_text.strip()
        
        # 移除可能的引号
        document = document.strip('"\'')
        
        # 移除可能的说明性前缀（如"假想文档："、"生成的文档："等）
        lines = document.split('\n')
        cleaned_lines = []
        skip_first = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            # 跳过明显的说明性文字（通常在开头）
            if i == 0 and any(keyword in line for keyword in ['假想文档', '生成的文档', '假设性文档', '文档内容']):
                if ':' in line or '：' in line:
                    # 提取冒号后的内容
                    parts = line.split(':', 1) if ':' in line else line.split('：', 1)
                    if len(parts) > 1:
                        line = parts[1].strip()
                        if line:
                            cleaned_lines.append(line)
                    skip_first = True
                    continue
                else:
                    skip_first = True
                    continue
            
            if line:
                cleaned_lines.append(line)
        
        # 重新组合文档
        if cleaned_lines:
            document = '\n'.join(cleaned_lines)
        
        return document.strip()

    def generate_hypothetical_document(question: str, stream: bool = False) -> str:
        """
        根据用户问题生成假设性文档片段
        
        Args:
            question: 用户输入的原始问题
            stream: 是否使用流式输出（流式模式下返回空字符串，直接打印输出）
        
        Returns:
            生成的假设性文档文本（非流式模式）
        """
        if stream:
            # 流式输出
            print("生成的假设性文档：")
            print("-" * 50)
            full_response = ""
            for chunk in chain.stream({"question": question}):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n" + "-" * 50)
            # 清理并返回文档
            return clean_hypothetical_document(full_response)
        else:
            # 普通输出
            response = chain.invoke({"question": question})
            return clean_hypothetical_document(response)

    # 执行测试
    # 检查 API Key
    if not DEEPSEEK_API_KEY:
        print("警告: 请设置环境变量 DEEPSEEK_API_KEY")
        print("Windows 示例: set DEEPSEEK_API_KEY=your-api-key")
        print("Linux/Mac 示例: export DEEPSEEK_API_KEY='your-api-key'")
    else:
        # 示例问题
        test_question = "什么是机器学习？"
        
        print("=" * 50)
        print("HyDE 模块 - 普通调用示例")
        print("=" * 50)
        print(f"用户问题: {test_question}\n")
        
        hypothetical_doc = generate_hypothetical_document(test_question, stream=False)
        print("生成的假设性文档:")
        print(hypothetical_doc)
        
        print("\n" + "=" * 50)
        print("HyDE 模块 - 流式输出示例")
        print("=" * 50)
        print(f"用户问题: {test_question}\n")
        
        hypothetical_doc_stream = generate_hypothetical_document(test_question, stream=True)
        print(f"\n最终生成的假设性文档长度: {len(hypothetical_doc_stream)} 字符")


# 执行测试
if __name__ == "__main__":
    test_hyde()

