# scripts/ingest.py
import os
import glob
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
MD_OUTPUT_PATH = os.path.join(DATA_DIR, "processed_chunks.md")

def build_multi_pdf_knowledge_base():
    # 1. 扫描 data/ 目录下的所有 PDF 文件
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_files:
        print(f"❌ 错误: 在 {DATA_DIR} 目录下未找到任何 .pdf 文件！")
        return

    print("="*65)
    print(f"📚 开始批量知识库构建，共检测到 {len(pdf_files)} 个 PDF 规章文件：")
    for p in pdf_files:
        print(f"   -> 发现文档: {os.path.basename(p)}")
    print("="*65)

    all_splits = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )

    # 2. 逐一解析并切分
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"🔄 正在解析并语义切片: {filename} ...")
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            splits = text_splitter.split_documents(docs)
            for s in splits:
                # 记录归属的文件名
                s.metadata["source_file"] = filename
            all_splits.extend(splits)
            print(f"   -> 成功提取 {len(splits)} 个切片。")
        except Exception as e:
            print(f"   ❌ 解析 {filename} 失败: {e}")

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
    print("🎉 多 PDF 知识库构建完毕！向量索引已刷新。\n")

if __name__ == "__main__":
    build_multi_pdf_knowledge_base()