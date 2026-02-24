# Fetch Laws from GitHub
"""
从GitHub开源项目获取法律数据
数据源：https://github.com/LawRefBook/Laws（中国法律快查手册）
许可：开源项目，可自由使用
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import requests
from pathlib import Path
import re

# GitHub仓库信息
GITHUB_REPO = "LawRefBook/Laws"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}/contents"
RAW_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main"


def fetch_github_directory(path=""):
    """获取GitHub目录内容"""
    url = f"{GITHUB_API}/{path}" if path else GITHUB_API
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def download_file(path: str, save_dir: Path) -> bool:
    """下载单个文件"""
    try:
        url = f"{RAW_URL}/{path}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        save_path = save_dir / path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        return True
    except Exception as e:
        print(f"下载失败 {path}: {e}")
        return False


def fetch_all_laws(max_files: int = 100):
    """
    获取所有法律文件

    Args:
        max_files: 最大下载文件数（防止下载过多）
    """
    save_dir = Path("./data/raw/laws_github")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始从 {GITHUB_REPO} 获取法律数据...")

    downloaded = 0
    all_laws = []

    try:
        # 获取根目录内容
        root_contents = fetch_github_directory()

        for item in root_contents:
            if downloaded >= max_files:
                break

            if item['type'] == 'file' and item['name'].endswith('.md'):
                # 下载markdown文件
                if download_file(item['path'], save_dir):
                    downloaded += 1
                    print(f"[{downloaded}] 下载: {item['name']}")

            elif item['type'] == 'dir':
                # 获取子目录内容
                try:
                    sub_contents = fetch_github_directory(item['path'])
                    for sub_item in sub_contents:
                        if downloaded >= max_files:
                            break
                        if sub_item['type'] == 'file' and sub_item['name'].endswith('.md'):
                            if download_file(sub_item['path'], save_dir):
                                downloaded += 1
                                print(f"[{downloaded}] 下载: {item['name']}/{sub_item['name']}")
                except Exception as e:
                    print(f"获取目录 {item['path']} 失败: {e}")

    except Exception as e:
        print(f"获取数据失败: {e}")

    print(f"\n下载完成，共 {downloaded} 个文件")
    print(f"保存目录: {save_dir}")

    return save_dir


def parse_law_files(data_dir: str = "./data/raw/laws_github"):
    """
    解析法律文件，提取法条

    Args:
        data_dir: 数据目录
    """
    data_path = Path(data_dir)
    laws = []

    for md_file in data_path.rglob("*.md"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取法律名称（通常在第一行或标题中）
            lines = content.strip().split('\n')
            title = ""
            for line in lines[:5]:
                if line.startswith('# '):
                    title = line[2:].strip()
                    break

            if not title:
                title = md_file.stem

            # 按条款分割
            article_pattern = r'第([一二三四五六七八九十百零\d]+)条\s*([^\n]+(?:\n(?!第)[^\n]+)*)'
            articles = re.findall(article_pattern, content)

            for article_num, article_text in articles:
                article_text = article_text.strip()
                if len(article_text) > 20:  # 过滤太短的内容
                    laws.append({
                        "id": f"{md_file.stem}_{article_num}",
                        "text": article_text,
                        "metadata": {
                            "law_name": title,
                            "article_number": f"第{article_num}条",
                            "source_file": str(md_file.relative_to(data_path)),
                            "doc_type": "law"
                        }
                    })

        except Exception as e:
            print(f"解析 {md_file} 失败: {e}")

    print(f"解析完成，共 {len(laws)} 条法条")
    return laws


def import_to_vector_db(laws: list, api_key: str):
    """导入到向量数据库"""
    from src.rag.retriever import LegalRetriever

    print(f"\n开始导入 {len(laws)} 条法条到向量数据库...")

    retriever = LegalRetriever(
        api_key=api_key,
        persist_directory="./data/vector_db"
    )

    indexed = retriever.index_documents(laws, batch_size=50)

    print(f"导入完成，共 {indexed} 条")
    print(f"数据库统计: {retriever.get_stats()}")

    return retriever


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("ZHIPU_API_KEY")

    if not api_key:
        print("请设置 ZHIPU_API_KEY 环境变量")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description="获取法律数据")
    parser.add_argument("--fetch", action="store_true", help="从GitHub下载数据")
    parser.add_argument("--parse", action="store_true", help="解析已下载的数据")
    parser.add_argument("--import", dest="import_db", action="store_true", help="导入到向量数据库")
    parser.add_argument("--all", action="store_true", help="执行全部步骤")
    parser.add_argument("--max-files", type=int, default=50, help="最大下载文件数")

    args = parser.parse_args()

    if args.all:
        args.fetch = True
        args.parse = True
        args.import_db = True

    if not any([args.fetch, args.parse, args.import_db]):
        parser.print_help()
        print("\n示例:")
        print("  python scripts/fetch_laws_from_github.py --all        # 执行全部步骤")
        print("  python scripts/fetch_laws_from_github.py --fetch      # 仅下载数据")
        print("  python scripts/fetch_laws_from_github.py --parse      # 仅解析数据")
        print("  python scripts/fetch_laws_from_github.py --import     # 仅导入数据库")
        sys.exit(0)

    laws = []

    if args.fetch:
        fetch_all_laws(max_files=args.max_files)

    if args.parse:
        laws = parse_law_files()

    if args.import_db:
        if not laws:
            laws = parse_law_files()
        if laws:
            import_to_vector_db(laws, api_key)
