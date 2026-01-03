"""
文档切分优化模块
支持多种格式文档加载、智能切分和 Chroma 向量数据库嵌入

支持的文档格式：.txt, .md, .markdown, .pdf, .docx, .doc

使用方法：

1. 命令行使用：
    # 仅处理文档（不嵌入）
    python Chunk_Optimization.py [目录路径]
    
    # 处理并嵌入到向量数据库（推荐）
    python Chunk_Optimization.py [目录路径] --embed
    
    # 完整参数示例
    python Chunk_Optimization.py Data --embed --persist-dir ./chroma_db \
        --chunk-size 500 --chunk-overlap 50 --embedding-type local

2. 处理目录并切分：
    from Indexing.Chunk_Optimization import process_directory
    chunks = process_directory("Data", chunk_size=500, chunk_overlap=50)

3. 处理目录并嵌入到 Chroma（推荐，使用 LCEL 语法）：
    from Indexing.Chunk_Optimization import process_and_embed_directory_lcel
    # 使用 LCEL 语法处理并嵌入（推荐）
    vector_store = process_and_embed_directory_lcel(
        directory="Data",
        persist_directory="./chroma_db",
        embedding_type="dashscope",  # 或 "local" 使用本地模型
        auto_embed=True  # True=嵌入到向量库，False=仅返回Document列表
    )
    
    # 使用向量数据库检索
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    results = retriever.get_relevant_documents("查询问题")
    

4. 加载已存在的 Chroma 数据库：
    from Indexing.Chunk_Optimization import load_chroma_vectorstore
    vector_store = load_chroma_vectorstore(
        persist_directory="./chroma_db",
        collection_name="documents"
    )

注意事项：
- DashScope嵌入：需要在 .env 文件中设置 DASHSCOPE_API_KEY，默认模型 "text-embedding-v1"
- 本地嵌入：使用 sentence-transformers，默认模型 "BAAI/bge-small-zh-v1.5"（中文优化）
- 需要安装：langchain-chroma, chromadb, dashscope, langchain-community, sentence-transformers, python-dotenv（可选）

"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader
)
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import sys
import dotenv
dotenv.load_dotenv()
import os

# 加载 API key 和 API URL（可从环境变量或自定义变量名读取）
api_key = os.getenv("DASHSCOPE_API_KEY")
api_url = os.getenv("DASHSCOPE_API_URL")


# 尝试导入嵌入模型支持
try:
    from langchain_community.embeddings import DashScopeEmbeddings, HuggingFaceEmbeddings
    HAS_DASHSCOPE_EMBEDDINGS = True
    HAS_LOCAL_EMBEDDINGS = True
except ImportError:
    HAS_DASHSCOPE_EMBEDDINGS = False
    HAS_LOCAL_EMBEDDINGS = False


def split_text(text: str, 
               chunk_size: int = 500, 
               chunk_overlap: int = 50) -> List[str]:
    """
    切分文档文本
    
    Args:
        text: 需要切分的文档文本
        chunk_size: 每个块的最大字符数，默认为 500
        chunk_overlap: 块之间的重叠字符数，默认为 50
        
    Returns:
        切分后的文档块列表
    """
    if not text or not text.strip():
        return []
    
    separators = ["\n\n", "\n", "。", "，", " ", ""]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators
    )
    
    return text_splitter.split_text(text)


def load_document(file_path: Union[str, Path]) -> str:
    """
    加载文档函数，支持多种文档格式
    
    支持的格式：
    - TextLoader：文本文档加载（.txt）
    - UnstructuredMarkdownLoader：Markdown加载（.md, .markdown）
    - UnstructuredPDFLoader：PDF加载（.pdf）
    - UnstructuredWordDocumentLoader：Word文档加载（.docx, .doc）
    
    Args:
        file_path: 文档文件路径（字符串或Path对象）
        
    Returns:
        文档文本内容（字符串）
        
    Raises:
        ValueError: 如果文件不存在或格式不支持
        Exception: 如果加载过程中出现错误
    """
    file_path = Path(file_path)
    
    # 检查文件是否存在
    if not file_path.exists():
        raise ValueError(f"文件不存在: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"路径不是文件: {file_path}")
    
    # 获取文件扩展名（小写）
    file_ext = file_path.suffix.lower()
    
    try:
        # 根据文件扩展名选择合适的加载器
        if file_ext == '.txt':
            # 文本文档加载
            loader = TextLoader(str(file_path), encoding='utf-8')
        elif file_ext in ['.md', '.markdown']:
            # Markdown文档加载
            loader = UnstructuredMarkdownLoader(str(file_path))
        elif file_ext == '.pdf':
            # PDF文档加载
            loader = UnstructuredPDFLoader(str(file_path))
        elif file_ext in ['.docx', '.doc']:
            # Word文档加载
            loader = UnstructuredWordDocumentLoader(str(file_path))
        else:
            raise ValueError(
                f"不支持的文件格式: {file_ext}。"
                f"支持的格式: .txt, .md, .markdown, .pdf, .docx, .doc"
            )
        
        # 加载文档
        documents = loader.load()
        
        # 合并所有文档页面的内容
        text_content = "\n\n".join([doc.page_content for doc in documents])
        
        return text_content
        
    except Exception as e:
        raise Exception(f"加载文档时出错 ({file_path}): {str(e)}")


def load_documents(file_paths: List[Union[str, Path]]) -> List[str]:
    """
    批量加载多个文档
    
    Args:
        file_paths: 文档文件路径列表
        
    Returns:
        文档文本内容列表（每个元素对应一个文档）
        
    Raises:
        ValueError: 如果某个文件不存在或格式不支持
    """
    documents = []
    errors = []
    
    for file_path in file_paths:
        try:
            text = load_document(file_path)
            documents.append(text)
            print(f"成功加载文档: {file_path}")
        except Exception as e:
            error_msg = f"加载文档失败 ({file_path}): {str(e)}"
            errors.append(error_msg)
            print(f"警告: {error_msg}")
    
    if errors and not documents:
        raise ValueError(f"所有文档加载失败:\n" + "\n".join(errors))
    
    return documents




def process_directory(directory: Union[str, Path],
                     chunk_size: int = 500,
                     chunk_overlap: int = 50) -> List[str]:
    """
    处理目录中的所有支持格式的文档
    
    Args:
        directory: 要处理的目录路径
        chunk_size: 每个块的最大字符数，默认为 500
        chunk_overlap: 块之间的重叠字符数，默认为 50
        
    Returns:
        切分后的文档块列表（所有文档的块合并在一起）
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise ValueError(f"目录不存在: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"路径不是目录: {directory}")
    
    # 支持的文件扩展名
    supported_extensions = ['.txt', '.md', '.markdown', '.pdf', '.docx', '.doc']
    
    # 查找所有支持的文件
    file_paths = []
    for ext in supported_extensions:
        file_paths.extend(directory.glob(f"*{ext}"))
        file_paths.extend(directory.glob(f"**/*{ext}"))  # 递归查找
    
    # 去重并排序
    file_paths = sorted(set(file_paths))
    
    if not file_paths:
        print(f"在目录 {directory} 中未找到支持的文件格式")
        return []
    
    print(f"找到 {len(file_paths)} 个文档文件")
    for file_path in file_paths:
        print(f"  - {file_path}")
    
    # 加载所有文档
    documents = load_documents(file_paths)
    
    # 切分所有文档
    all_chunks = []
    for i, doc in enumerate(documents):
        if not doc or not doc.strip():
            continue
        chunks = split_text(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        print(f"文档 {i+1} 被切分为 {len(chunks)} 个块")
    
    print(f"\n总共生成 {len(all_chunks)} 个文档块")
    
    return all_chunks


def get_embeddings(embedding_type: str = "dashscope",
                   model_name: Optional[str] = None) -> Embeddings:
    """
    获取嵌入模型实例
    
    Args:
        embedding_type: 嵌入模型类型，"dashscope" 或 "local"
        model_name: 模型名称，默认 "text-embedding-v1"
        
    Returns:
        嵌入模型实例
    """
    if embedding_type.lower() == "dashscope":
        if not HAS_DASHSCOPE_EMBEDDINGS:
            raise ImportError(
                "DashScope 嵌入模型需要安装额外依赖。请运行: pip install dashscope"
            )
        if model_name is None:
            model_name = "text-embedding-v1"
        # 直接使用全局变量 api_key
        return DashScopeEmbeddings(
            model=model_name,
            dashscope_api_key=api_key
        )
    
    elif embedding_type.lower() == "local":
        if not HAS_LOCAL_EMBEDDINGS:
            raise ImportError(
                "本地嵌入模型需要安装额外依赖。请运行: pip install sentence-transformers"
            )
        if model_name is None:
            model_name = "BAAI/bge-small-zh-v1.5"
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        raise ValueError(f"不支持的嵌入模型类型: {embedding_type}，支持的类型: dashscope, local")


def process_and_embed_directory_lcel(
    directory: Union[str, Path],
    persist_directory: Union[str, Path] = "./chroma_db",
    collection_name: str = "documents",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_type: str = "dashscope",
    embedding_model: Optional[str] = None,
    auto_embed: bool = True) -> Union[Chroma, List[Document]]:
    """
    使用 LCEL（LangChain Expression Language）语法处理目录并嵌入到向量库
    
    Args:
        directory: 要处理的文档目录路径
        persist_directory: Chroma 数据库持久化目录路径，默认为 "./chroma_db"
        collection_name: 集合名称，默认为 "documents"
        chunk_size: 每个块的最大字符数，默认为 500
        chunk_overlap: 块之间的重叠字符数，默认为 50
        embedding_type: 嵌入模型类型，"dashscope" 或 "local"，默认为 "dashscope"
        embedding_model: 模型名称（可选，使用默认值）
        auto_embed: 是否自动嵌入到向量库，True 返回 Chroma 对象，False 返回 Document 列表
        
    Returns:
        如果 auto_embed=True，返回 Chroma 向量存储对象
        如果 auto_embed=False，返回 Document 对象列表
        
    Raises:
        ValueError: 如果目录不存在或处理过程中出现错误
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise ValueError(f"目录不存在: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"路径不是目录: {directory}")
    
    # 支持的文件扩展名
    supported_extensions = ['.txt', '.md', '.markdown', '.pdf', '.docx', '.doc']
    
    # 查找所有支持的文件
    file_paths = []
    for ext in supported_extensions:
        file_paths.extend(directory.glob(f"*{ext}"))
        file_paths.extend(directory.glob(f"**/*{ext}"))
    
    file_paths = sorted(set(file_paths))
    
    if not file_paths:
        print(f"在目录 {directory} 中未找到支持的文件格式")
        return [] if not auto_embed else None
    
    print(f"找到 {len(file_paths)} 个文档文件")
    
    # 获取嵌入模型（如果需要嵌入）
    embeddings = None
    if auto_embed:
        embeddings = get_embeddings(embedding_type, embedding_model)
    
    # 使用 LCEL 语法构建处理链
    # 步骤1: 加载文档
    def load_docs(file_paths_list: List[Path]) -> List[str]:
        """加载所有文档内容"""
        texts = []
        for file_path in file_paths_list:
            try:
                text = load_document(file_path)
                texts.append(text)
                print(f"成功加载文档: {file_path}")
            except Exception as e:
                print(f"警告: 加载文档失败 ({file_path}): {str(e)}")
        return texts
    
    # 步骤2: 切分文本
    def split_docs(texts_list: List[str]) -> List[str]:
        """切分所有文档"""
        all_chunks = []
        for i, text in enumerate(texts_list):
            if not text or not text.strip():
                continue
            chunks = split_text(text, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
            print(f"文档 {i+1} 被切分为 {len(chunks)} 个块")
        print(f"\n总共生成 {len(all_chunks)} 个文档块")
        return all_chunks
    
    # 步骤3: 转换为 Document 对象
    def to_documents(chunks_list: List[str]) -> List[Document]:
        """转换为 Document 对象"""
        metadata = {"source_directory": str(directory.absolute())}
        documents = []
        for i, chunk in enumerate(chunks_list):
            doc_metadata = {"chunk_index": i}
            doc_metadata.update(metadata)
            documents.append(Document(page_content=chunk, metadata=doc_metadata))
        return documents
    
    # 步骤4: 嵌入到向量库
    def embed_documents(docs_list: List[Document]) -> Chroma:
        """嵌入文档到 Chroma 向量库"""
        persist_directory_path = Path(persist_directory)
        vector_store = Chroma.from_documents(
            documents=docs_list,
            embedding=embeddings,
            persist_directory=str(persist_directory_path),
            collection_name=collection_name
        )
        print(f"\n成功将 {len(docs_list)} 个文档块嵌入到 Chroma 数据库")
        print(f"数据库路径: {persist_directory_path}")
        print(f"集合名称: {collection_name}")
        return vector_store
    
    # 使用 LCEL 语法构建处理链
    if auto_embed:
        # 完整流程：加载 -> 切分 -> 转换 -> 嵌入
        pipeline = (
            RunnableLambda(load_docs)
            | RunnableLambda(split_docs)
            | RunnableLambda(to_documents)
            | RunnableLambda(embed_documents)
        )
        return pipeline.invoke(file_paths)
    else:
        # 不嵌入流程：加载 -> 切分 -> 转换
        pipeline = (
            RunnableLambda(load_docs)
            | RunnableLambda(split_docs)
            | RunnableLambda(to_documents)
        )
        return pipeline.invoke(file_paths)


def load_chroma_vectorstore(persist_directory: Union[str, Path] = "./chroma_db",
                            collection_name: str = "documents",
                            embedding_type: str = "dashscope",
                            embedding_model: Optional[str] = None) -> Chroma:
    """
    加载已存在的 Chroma 向量数据库
    
    Args:
        persist_directory: Chroma 数据库持久化目录路径，默认为 "./chroma_db"
        collection_name: 集合名称，默认为 "documents"
        embedding_type: 嵌入模型类型，"dashscope" 或 "local"，默认为 "dashscope"
        embedding_model: 模型名称（可选，使用默认值）
        
    Returns:
        Chroma 向量存储对象
        
    Raises:
        Exception: 如果数据库不存在或加载失败
    """
    try:
        embeddings = get_embeddings(embedding_type, embedding_model)
        
        persist_directory = Path(persist_directory)
        vector_store = Chroma(
            persist_directory=str(persist_directory),
            collection_name=collection_name,
            embedding_function=embeddings
        )
        
        print(f"成功加载 Chroma 向量数据库: {persist_directory}")
        return vector_store
        
    except Exception as e:
        raise Exception(f"加载 Chroma 向量数据库时出错: {str(e)}")


def main():
    """主程序入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='文档处理和信息索引工具')
    parser.add_argument('directory', nargs='?', default='Data', help='要处理的文档目录（默认: Data）')
    parser.add_argument('--embed', action='store_true', help='是否嵌入到向量数据库')
    parser.add_argument('--persist-dir', default='./chroma_db', help='向量数据库存储目录（默认: ./chroma_db）')
    parser.add_argument('--collection', default='documents', help='集合名称（默认: documents）')
    parser.add_argument('--chunk-size', type=int, default=500, help='文档块大小（默认: 500）')
    parser.add_argument('--chunk-overlap', type=int, default=50, help='文档块重叠（默认: 50）')
    parser.add_argument('--embedding-type', choices=['dashscope', 'local'], default='dashscope', 
                       help='嵌入模型类型（默认: dashscope）')
    parser.add_argument('--embedding-model', help='嵌入模型名称（可选）')
    
    args = parser.parse_args()
    
    try:
        if args.embed:
            # 使用 LCEL 语法处理并嵌入
            vector_store = process_and_embed_directory_lcel(
                directory=args.directory,
                persist_directory=args.persist_dir,
                collection_name=args.collection,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                embedding_type=args.embedding_type,
                embedding_model=args.embedding_model,
                auto_embed=True
            )
            print(f"\n{'='*50}")
            print(f"处理完成！文档已嵌入到向量数据库")
            print(f"{'='*50}")
        else:
            # 仅处理文档，不嵌入
            chunks = process_directory(
                args.directory,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            print(f"\n{'='*50}")
            print(f"处理完成！共生成 {len(chunks)} 个文档块")
            print(f"{'='*50}")
        return 0
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
