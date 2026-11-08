# scripts/ingest.py
import os
from pathlib import Path

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = Path(DATA_DIR)
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
MD_OUTPUT_PATH = os.path.join(DATA_DIR, "processed_chunks.md")


def load_document(path: Path):
    """按扩展名加载文档：PDF 保留分页，TXT 当作单页资料。"""
    if path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(path)).load()
    text = path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": path.name, "page": 0})]


def build_multi_pdf_knowledge_base():
    # 1. 递归扫描 data/ 下的 PDF 和 TXT，干扰文档放在子目录也能被纳入
    pdf_files = sorted(DATA_PATH.rglob("*.pdf"))
    txt_files = sorted(DATA_PATH.rglob("*.txt"))
    source_files = pdf_files + txt_files

    # 这份 PDF 是另一所化妆学校的招生指南，只作为人工对照，不进入业务知识库。
    # 否则客服会把“怎么选化妆学校老师”当成课程问题回答。
    skip_files = {"teacher_faculty.pdf"}
    source_files = [path for path in source_files if path.name not in skip_files]
    pdf_files = [path for path in pdf_files if path.name not in skip_files]
    txt_files = [path for path in txt_files if path.name not in skip_files]

    if not source_files:
        print(f"❌ 错误: 在 {DATA_DIR} 目录下未找到任何 .pdf 或 .txt 文件！")
        return

    print("=" * 65)
    print(f"📚 开始批量知识库构建，共检测到 {len(source_files)} 个文档"
          f"（PDF {len(pdf_files)} / TXT {len(txt_files)}）：")
    for path in source_files:
        print(f"   -> 发现文档: {path.relative_to(DATA_PATH)}")
    print("=" * 65)

    all_splits = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )

    # 2. 逐一解析并切分
    for path in source_files:
        print(f"🔄 正在解析并语义切片: {path.name} ...")
        try:
            docs = load_document(path)
            splits = text_splitter.split_documents(docs)
            for split in splits:
                # 记录归属的文件名；引用展示用 basename
                split.metadata["source_file"] = path.name
            all_splits.extend(splits)
            print(f"   -> 成功提取 {len(splits)} 个切片。")
        except Exception as error:
            print(f"   ❌ 解析 {path.name} 失败: {error}")

    print(f"\n 全量文档切分完毕！总计生成 {len(all_splits)} 个知识切片。")

    # 3. 统一导出 Markdown 总切分台账，方便人工审计
    print(f"📝 正在导出全局切片清单至: {MD_OUTPUT_PATH} ...")
    with open(MD_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("# 智学在线教育 - 知识库全量离线切分台账\n\n")
        for idx, doc in enumerate(all_splits):
            src = doc.metadata.get("source_file", "未知文档")
            page = doc.metadata.get("page", 0) + 1
            f.write(f"## 切片 #{idx+1} [来源: 《{src}》 第 {page} 页]\n")
            f.write(f"```text\n{doc.page_content}\n```\n\n")

    # 4. 重建并持久化至 Chroma 向量库
    print(f"💾 正在同步更新持久化向量数据库: {PERSIST_DIR} ...")
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    
    try:
        client.delete_collection("edu_pdf_knowledge")
    except:
        pass
        
    collection = client.create_collection(name="edu_pdf_knowledge")

    documents = []
    metadatas = []
    ids = []

    for idx, doc in enumerate(all_splits):
        documents.append(doc.page_content)
        metadatas.append({
            "source": doc.metadata.get("source_file", "未知文档"),
            "page": doc.metadata.get("page", 0) + 1
        })
        ids.append(f"doc_chunk_{idx}")

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print("🎉 多文档知识库构建完毕！向量索引已刷新。\n")

if __name__ == "__main__":
    build_multi_pdf_knowledge_base()