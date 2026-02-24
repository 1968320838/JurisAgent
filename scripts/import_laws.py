# Import Laws Script
"""
法规数据导入脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path

from src.rag.embeddings import get_embedding_engine
from src.rag.vector_store import VectorStore
from src.rag.retriever import LegalRetriever


# 示例法规数据（民法典部分条款）
SAMPLE_LAWS = [
    {
        "id": "civil_code_577",
        "text": "当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。",
        "metadata": {
            "law_name": "中华人民共和国民法典",
            "article_number": "第五百七十七条",
            "category": "合同编",
            "doc_type": "law"
        }
    },
    {
        "id": "civil_code_578",
        "text": "当事人一方明确表示或者以自己的行为表明不履行合同义务的，对方可以在履行期限届满前请求其承担违约责任。",
        "metadata": {
            "law_name": "中华人民共和国民法典",
            "article_number": "第五百七十八条",
            "category": "合同编",
            "doc_type": "law"
        }
    },
    {
        "id": "civil_code_579",
        "text": "当事人一方未支付价款、报酬、租金、利息，或者不履行其他金钱债务的，对方可以请求其支付。",
        "metadata": {
            "law_name": "中华人民共和国民法典",
            "article_number": "第五百七十九条",
            "category": "合同编",
            "doc_type": "law"
        }
    },
    {
        "id": "civil_code_580",
        "text": "当事人一方不履行非金钱债务或者履行非金钱债务不符合约定的，对方可以请求履行，但是有下列情形之一的除外：（一）法律上或者事实上不能履行；（二）债务的标的不适于强制履行或者履行费用过高；（三）债权人在合理期限内未请求履行。",
        "metadata": {
            "law_name": "中华人民共和国民法典",
            "article_number": "第五百八十条",
            "category": "合同编",
            "doc_type": "law"
        }
    },
    {
        "id": "civil_code_584",
        "text": "当事人一方不履行合同义务或者履行合同义务不符合约定，造成对方损失的，损失赔偿额应当相当于因违约所造成的损失，包括合同履行后可以获得的利益；但是，不得超过违约一方订立合同时预见到或者应当预见到的因违约可能造成的损失。",
        "metadata": {
            "law_name": "中华人民共和国民法典",
            "article_number": "第五百八十四条",
            "category": "合同编",
            "doc_type": "law"
        }
    },
    {
        "id": "civil_code_585",
        "text": "当事人可以约定一方违约时应当根据违约情况向对方支付一定数额的违约金，也可以约定因违约产生的损失赔偿额的计算方法。约定的违约金低于造成的损失的，人民法院或者仲裁机构可以根据当事人的请求予以增加；约定的违约金过分高于造成的损失的，人民法院或者仲裁机构可以根据当事人的请求予以适当减少。",
        "metadata": {
            "law_name": "中华人民共和国民法典",
            "article_number": "第五百八十五条",
            "category": "合同编",
            "doc_type": "law"
        }
    },
    {
        "id": "civil_code_465",
        "text": "依法成立的合同，受法律保护。依法成立的合同，仅对当事人具有法律约束力，但是法律另有规定的除外。",
        "metadata": {
            "law_name": "中华人民共和国民法典",
            "article_number": "第四百六十五条",
            "category": "合同编",
            "doc_type": "law"
        }
    },
    {
        "id": "civil_code_466",
        "text": "当事人对合同条款的理解有争议的，应当依据本法第一百四十二条第一款的规定，确定争议条款的含义。合同文本采用两种以上文字订立并约定具有同等效力的，对各文本使用的词句推定具有相同含义。各文本使用的词句不一致的，应当根据合同的相关条款、性质、目的以及诚信原则等予以解释。",
        "metadata": {
            "law_name": "中华人民共和国民法典",
            "article_number": "第四百六十六条",
            "category": "合同编",
            "doc_type": "law"
        }
    },
    {
        "id": "civil_code_470",
        "text": "合同的内容由当事人约定，一般包括下列条款：（一）当事人的姓名或者名称和住所；（二）标的；（三）数量；（四）质量；（五）价款或者报酬；（六）履行期限、地点和方式；（七）违约责任；（八）解决争议的方法。",
        "metadata": {
            "law_name": "中华人民共和国民法典",
            "article_number": "第四百七十条",
            "category": "合同编",
            "doc_type": "law"
        }
    },
    {
        "id": "civil_code_509",
        "text": "当事人应当按照约定全面履行自己的义务。当事人应当遵循诚信原则，根据合同的性质、目的和交易习惯履行通知、协助、保密等义务。",
        "metadata": {
            "law_name": "中华人民共和国民法典",
            "article_number": "第五百零九条",
            "category": "合同编",
            "doc_type": "law"
        }
    },
    # 劳动法相关
    {
        "id": "labor_contract_47",
        "text": "经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。六个月以上不满一年的，按一年计算；不满六个月的，向劳动者支付半个月工资的经济补偿。",
        "metadata": {
            "law_name": "中华人民共和国劳动合同法",
            "article_number": "第四十七条",
            "category": "劳动合同",
            "doc_type": "law"
        }
    },
    {
        "id": "labor_contract_48",
        "text": "用人单位违反本法规定解除或者终止劳动合同，劳动者要求继续履行劳动合同的，用人单位应当继续履行；劳动者不要求继续履行劳动合同或者劳动合同已经不能继续履行的，用人单位应当依照本法第八十七条规定支付赔偿金。",
        "metadata": {
            "law_name": "中华人民共和国劳动合同法",
            "article_number": "第四十八条",
            "category": "劳动合同",
            "doc_type": "law"
        }
    },
    {
        "id": "labor_contract_82",
        "text": "用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。",
        "metadata": {
            "law_name": "中华人民共和国劳动合同法",
            "article_number": "第八十二条",
            "category": "劳动合同",
            "doc_type": "law"
        }
    },
    # 民间借贷相关
    {
        "id": "lending_regulation_26",
        "text": "出借人请求借款人按照合同约定利率支付利息的，人民法院应予支持，但是双方约定的利率超过合同成立时一年期贷款市场报价利率四倍的除外。",
        "metadata": {
            "law_name": "最高人民法院关于审理民间借贷案件适用法律若干问题的规定",
            "article_number": "第二十六条",
            "category": "民间借贷",
            "doc_type": "law"
        }
    },
]


def import_laws(api_key: str, laws: list = None, persist_directory: str = "./data/vector_db"):
    """
    导入法规数据到向量数据库

    Args:
        api_key: 智谱AI API密钥
        laws: 法规数据列表
        persist_directory: 向量数据库目录
    """
    laws = laws or SAMPLE_LAWS

    print(f"开始导入 {len(laws)} 条法规数据...")

    # 初始化检索器
    retriever = LegalRetriever(
        api_key=api_key,
        persist_directory=persist_directory
    )

    # 索引文档
    indexed = retriever.index_documents(laws, batch_size=50)

    print(f"成功导入 {indexed} 条法规数据")
    print(f"向量数据库统计: {retriever.get_stats()}")

    return retriever


def test_search(retriever: LegalRetriever, query: str):
    """测试检索功能"""
    print(f"\n检索查询: {query}")
    print("-" * 50)

    results = retriever.retrieve(query, n_results=3)

    for i, result in enumerate(results, 1):
        metadata = result["metadata"]
        print(f"[{i}] {metadata.get('law_name', '')} {metadata.get('article_number', '')}")
        print(f"    相似度: {result['score']:.4f}")
        print(f"    内容: {result['text'][:100]}...")
        print()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("ZHIPU_API_KEY")

    if not api_key:
        print("请设置 ZHIPU_API_KEY 环境变量")
        sys.exit(1)

    # 导入法规
    retriever = import_laws(api_key)

    # 测试检索
    test_search(retriever, "违约金过高怎么办")
    test_search(retriever, "劳动合同解除赔偿")
