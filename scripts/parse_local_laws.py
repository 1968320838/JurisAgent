# Parse Local Laws Script
"""
解析本地法律文件并导入向量数据库
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import uuid
from pathlib import Path
from typing import List, Dict


def parse_law_file(file_path: Path) -> List[Dict]:
    """
    解析单个法律文件，提取法条

    Args:
        file_path: 文件路径

    Returns:
        List[Dict]: 法条列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"读取失败 {file_path}: {e}")
        return []

    # 提取法律名称（通常在第一行标题中）
    lines = content.strip().split('\n')
    law_name = ""
    category = ""

    for line in lines[:10]:
        if line.startswith('# '):
            name = line[2:].strip()
            if not law_name:
                law_name = name
            elif not category:
                category = name
        if '<!-- INFO END -->' in line:
            break

    # 按条款分割 - 支持中文数字和阿拉伯数字
    # 匹配: 第四百六十三条 或 第463条
    article_pattern = r'第([一二三四五六七八九十百零\d]+)条[ \s]*([^\n]+(?:\n(?!第[一二三四五六七八九十百零\d]+条)[^\n]+)*)'

    articles = []
    article_counter = 0  # 用于生成唯一ID
    for match in re.finditer(article_pattern, content):
        article_num = match.group(1)
        article_text = match.group(2).strip()

        # 清理条款文本
        article_text = re.sub(r'\n+', ' ', article_text)
        article_text = re.sub(r'\s+', ' ', article_text)

        if len(article_text) > 15:  # 过滤太短的内容
            article_counter += 1
            # 使用UUID生成唯一ID
            unique_id = str(uuid.uuid4())

            articles.append({
                "id": unique_id,
                "text": article_text,
                "metadata": {
                    "law_name": law_name,
                    "article_number": f"第{article_num}条",
                    "category": category,
                    "source_file": str(file_path.relative_to(file_path.parents[2])),
                    "doc_type": "law"
                }
            })

    return articles


def chinese_to_arabic(chinese_num: str) -> str:
    """将中文数字转换为阿拉伯数字"""
    chinese_map = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '十': 10, '百': 100
    }

    # 如果已经是阿拉伯数字，直接返回
    if chinese_num.isdigit():
        return chinese_num

    # 简单的中文数字转换
    result = 0
    temp = 0

    for char in chinese_num:
        if char in chinese_map:
            val = chinese_map[char]
            if val >= 10:
                if temp == 0:
                    temp = 1
                result += temp * val
                temp = 0
            else:
                temp = val

    result += temp
    return str(result) if result > 0 else chinese_num


def parse_all_laws(data_dir: str, max_files: int = 200) -> List[Dict]:
    """
    解析目录下所有法律文件

    Args:
        data_dir: 数据目录
        max_files: 最大解析文件数

    Returns:
        List[Dict]: 所有法条列表
    """
    data_path = Path(data_dir)
    all_laws = []
    file_count = 0

    # 优先解析的重要目录
    priority_dirs = ["民法典", "民法商法", "合同编", "劳动", "公司"]

    # 先解析优先目录
    for priority in priority_dirs:
        for md_file in data_path.rglob("*.md"):
            if priority in str(md_file) and file_count < max_files:
                if md_file.name == "_index.md":
                    continue
                laws = parse_law_file(md_file)
                if laws:
                    all_laws.extend(laws)
                    file_count += 1
                    print(f"[{file_count}] {md_file.relative_to(data_path)}: {len(laws)} 条")

    # 再解析其他文件
    for md_file in data_path.rglob("*.md"):
        if file_count >= max_files:
            break
        if md_file.name == "_index.md":
            continue
        if any(p in str(md_file) for p in priority_dirs):
            continue  # 已处理过

        laws = parse_law_file(md_file)
        if laws:
            all_laws.extend(laws)
            file_count += 1
            print(f"[{file_count}] {md_file.relative_to(data_path)}: {len(laws)} 条")

    print(f"\n总计解析 {file_count} 个文件，{len(all_laws)} 条法条")
    return all_laws


def import_to_vector_db(laws: list, api_key: str):
    """导入到向量数据库"""
    from src.rag.retriever import LegalRetriever

    print(f"\n开始导入 {len(laws)} 条法条到向量数据库...")

    retriever = LegalRetriever(
        api_key=api_key,
        persist_directory="./data/vector_db"
    )

    # 分批导入
    batch_size = 50
    total_indexed = 0

    for i in range(0, len(laws), batch_size):
        batch = laws[i:i+batch_size]
        indexed = retriever.index_documents(batch, batch_size=batch_size)
        total_indexed += indexed
        print(f"  批次 {i//batch_size + 1}: 导入 {indexed} 条 (累计 {total_indexed})")

    print(f"\n导入完成，共 {total_indexed} 条")
    print(f"数据库统计: {retriever.get_stats()}")

    return retriever


def test_retrieval(retriever, query: str):
    """测试检索功能"""
    print(f"\n查询: {query}")
    print("-" * 50)

    results = retriever.retrieve(query, n_results=3)
    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        print(f"[{i}] {metadata.get('law_name', '')} {metadata.get('article_number', '')}")
        print(f"    相似度: {result['score']:.4f}")
        print(f"    内容: {result['text'][:80]}...")
        print()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("ZHIPU_API_KEY")

    if not api_key:
        print("请设置 ZHIPU_API_KEY 环境变量")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description="解析本地法律文件")
    parser.add_argument("--parse", action="store_true", help="解析法律文件")
    parser.add_argument("--import", dest="import_db", action="store_true", help="导入向量数据库")
    parser.add_argument("--test", action="store_true", help="测试检索")
    parser.add_argument("--all", action="store_true", help="执行全部步骤")
    parser.add_argument("--max-files", type=int, default=150, help="最大解析文件数")
    parser.add_argument("--data-dir", type=str,
                        default="G:/project/JurisAgent/src/rag/laws_github/Laws-master",
                        help="法律数据目录")

    args = parser.parse_args()

    if args.all:
        args.parse = True
        args.import_db = True
        args.test = True

    if not any([args.parse, args.import_db, args.test]):
        parser.print_help()
        print("\n示例:")
        print("  python scripts/parse_local_laws.py --all              # 执行全部")
        print("  python scripts/parse_local_laws.py --parse --import  # 解析并导入")
        print("  python scripts/parse_local_laws.py --test            # 仅测试")
        sys.exit(0)

    laws = []

    if args.parse:
        laws = parse_all_laws(args.data_dir, max_files=args.max_files)

    if args.import_db:
        if not laws:
            # 如果没有解析，尝试从缓存加载
            import json
            cache_file = "./data/parsed_laws.json"
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    laws = json.load(f)
                print(f"从缓存加载 {len(laws)} 条法条")
            else:
                laws = parse_all_laws(args.data_dir, max_files=args.max_files)

        if laws:
            # 保存缓存
            import json
            os.makedirs("./data", exist_ok=True)
            with open("./data/parsed_laws.json", 'w', encoding='utf-8') as f:
                json.dump(laws, f, ensure_ascii=False, indent=2)
            print(f"已保存解析结果到 ./data/parsed_laws.json")

            retriever = import_to_vector_db(laws, api_key)

            if args.test:
                test_retrieval(retriever, "违约金过高怎么办")
                test_retrieval(retriever, "劳动合同解除赔偿")
                test_retrieval(retriever, "合同无效的情形")

    elif args.test:
        from src.rag.retriever import LegalRetriever
        retriever = LegalRetriever(
            api_key=api_key,
            persist_directory="./data/vector_db"
        )
        test_retrieval(retriever, "违约金过高怎么办")
        test_retrieval(retriever, "合同无效")
