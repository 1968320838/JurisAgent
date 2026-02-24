# JurisAgent Web Application
"""
法律智能体 - Streamlit界面
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from datetime import datetime

# 导入自定义模块
from glm_client import get_glm_client
from src.agents.contract_agent import ContractReviewAgent
from src.agents.legal_qa_agent import LegalQAAgent
from src.agents.case_agent import CaseAnalysisAgent
from config import APP_CONFIG, API_CONFIG

# 页面配置
st.set_page_config(
    page_title="JurisAgent - 法律智能体",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-top: 1rem;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 10px;
        margin: 5px 0;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 10px;
        margin: 5px 0;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 10px;
        margin: 5px 0;
    }
    .score-box {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .score-number {
        font-size: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def init_agent():
    """初始化合同审查Agent"""
    if "contract_agent" not in st.session_state:
        try:
            llm_client = get_glm_client()
            st.session_state.contract_agent = ContractReviewAgent(llm_client)
        except Exception as e:
            st.error(f"初始化失败: {str(e)}")
            st.info("请确保已正确配置API Key。您可以在 config.py 中设置或通过环境变量 ZHIPU_API_KEY 设置。")
            return None
    return st.session_state.contract_agent


def init_legal_qa_agent():
    """初始化法律咨询Agent（带RAG支持）"""
    if "legal_qa_agent" not in st.session_state:
        try:
            llm_client = get_glm_client()

            # 尝试初始化RAG检索器
            retriever = None
            try:
                from src.rag.retriever import LegalRetriever
                # 使用项目根目录的绝对路径
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                vector_db_path = os.path.join(project_root, "data", "vector_db")
                retriever = LegalRetriever(
                    api_key=API_CONFIG["api_key"],
                    persist_directory=vector_db_path
                )
                # 检查是否有数据
                stats = retriever.get_stats()
                if stats["total_documents"] > 0:
                    st.session_state.rag_enabled = True
                    st.session_state.rag_docs_count = stats["total_documents"]
                else:
                    retriever = None
                    st.session_state.rag_enabled = False
            except Exception as e:
                st.session_state.rag_enabled = False

            st.session_state.legal_qa_agent = LegalQAAgent(llm_client, retriever=retriever)
        except Exception as e:
            st.error(f"初始化失败: {str(e)}")
            return None
    return st.session_state.legal_qa_agent


def init_case_agent():
    """初始化案例分析Agent（带RAG支持）"""
    if "case_agent" not in st.session_state:
        try:
            llm_client = get_glm_client()

            # 尝试初始化RAG检索器
            retriever = None
            try:
                from src.rag.retriever import LegalRetriever
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                vector_db_path = os.path.join(project_root, "data", "vector_db")
                retriever = LegalRetriever(
                    api_key=API_CONFIG["api_key"],
                    persist_directory=vector_db_path
                )
                if retriever.get_stats()["total_documents"] == 0:
                    retriever = None
            except Exception:
                pass

            st.session_state.case_agent = CaseAnalysisAgent(llm_client, retriever=retriever)
        except Exception as e:
            st.error(f"初始化失败: {str(e)}")
            return None
    return st.session_state.case_agent


def main():
    """主函数"""
    # 标题
    st.markdown('<h1 class="main-header">⚖️ JurisAgent 法律智能体</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">专业的合同审查、法律咨询、案例分析智能助手</p>', unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/law.png", width=80)
        st.markdown(f"### {APP_CONFIG['app_name']}")
        st.markdown(f"版本: {APP_CONFIG['version']}")

        st.markdown("---")

        # 功能选择
        st.markdown("### 功能选择")
        page = st.radio(
            "选择功能",
            ["📄 合同审查", "💬 法律咨询", "📊 案例分析"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # API状态
        st.markdown("### 系统状态")
        if API_CONFIG["api_key"]:
            st.success("✅ API已配置")
        else:
            st.error("❌ API未配置")
            st.markdown("请在 config.py 中设置 API Key")

        # RAG状态
        rag_enabled = st.session_state.get("rag_enabled", False)
        rag_docs_count = st.session_state.get("rag_docs_count", 0)
        if rag_enabled:
            st.success(f"✅ RAG已启用 ({rag_docs_count:,}条法条)")
        else:
            st.info("ℹ️ RAG未启用")
            if st.button("初始化法条库", key="init_rag"):
                st.info("请运行: python scripts/parse_local_laws.py --import")

        st.markdown("---")

        # 使用统计
        if "review_count" not in st.session_state:
            st.session_state.review_count = 0
        st.markdown(f"📊 本次审查: {st.session_state.review_count} 次")

    # 主内容区
    if page == "📄 合同审查":
        show_contract_review()
    elif page == "💬 法律咨询":
        show_legal_qa()
    else:
        show_case_analysis()


def show_contract_review():
    """显示合同审查界面"""
    st.markdown('<h2 class="sub-header">📄 合同审查</h2>', unsafe_allow_html=True)
    st.markdown("上传合同文件，AI将自动识别风险条款并提供修改建议。")

    # 初始化Agent
    agent = init_agent()
    if not agent:
        return

    # 文件上传区域
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 上传合同")
        uploaded_file = st.file_uploader(
            "支持 PDF、Word (.docx) 格式",
            type=["pdf", "docx"],
            help="文件大小不超过 10MB"
        )

        # 或者直接输入文本
        st.markdown("### 或直接输入合同文本")
        contract_text = st.text_area(
            "粘贴合同内容",
            height=200,
            placeholder="将合同文本粘贴到此处..."
        )

    with col2:
        st.markdown("### 审查选项")
        review_mode = st.radio(
            "审查模式",
            ["标准审查", "快速审查"],
            help="标准审查提供详细报告，快速审查只列出主要风险"
        )

        st.markdown("### 审查重点（可选）")
        focus_areas = st.multiselect(
            "选择重点关注领域",
            ["违约责任", "付款条款", "知识产权", "保密条款", "争议解决", "合同解除"]
        )

    # 审查按钮
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        review_button = st.button("🔍 开始审查", type="primary", use_container_width=True)

    # 处理审查请求
    if review_button:
        if not uploaded_file and not contract_text.strip():
            st.warning("请上传合同文件或输入合同文本")
            return

        with st.spinner("正在审查合同，请稍候..."):
            try:
                # 准备输入数据
                if uploaded_file:
                    file_bytes = uploaded_file.read()
                    file_type = uploaded_file.name.split(".")[-1].lower()
                    input_data = {
                        "file_bytes": file_bytes,
                        "file_type": file_type,
                        "simple_mode": review_mode == "快速审查"
                    }
                    st.info(f"📄 已加载文件: {uploaded_file.name}")
                else:
                    input_data = {
                        "contract_text": contract_text,
                        "simple_mode": review_mode == "快速审查"
                    }

                # 执行审查
                result = agent.execute(input_data)

                # 更新统计
                st.session_state.review_count += 1

                # 显示结果
                display_review_result(result)

            except Exception as e:
                st.error(f"审查失败: {str(e)}")


def display_review_result(result):
    """显示审查结果"""
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📋 审查报告</h2>', unsafe_allow_html=True)

    if not result.success:
        st.error(f"审查失败: {result.error_message}")
        return

    # 顶部统计
    col1, col2, col3 = st.columns(3)
    with col1:
        score = result.metadata.get("compliance_score", "N/A")
        if score != "N/A":
            try:
                score_val = float(score)
                color = "#4caf50" if score_val >= 70 else "#ff9800" if score_val >= 50 else "#f44336"
                st.markdown(f"""
                <div class="score-box">
                    <div class="score-number" style="color: {color}">{score}</div>
                    <div>合规评分</div>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.metric("合规评分", score)
        else:
            st.metric("合规评分", "N/A")

    with col2:
        risk_count = result.metadata.get("risk_count", 0)
        st.metric("识别风险", f"{risk_count} 项")

    with col3:
        contract_length = result.metadata.get("contract_length", 0)
        st.metric("合同字数", f"{contract_length} 字")

    # 显示详细结果
    st.markdown("### 审查详情")

    # 尝试解析JSON结果
    import json
    try:
        # 尝试提取JSON
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', result.content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(1))
        else:
            parsed = json.loads(result.content)

        # 显示合同摘要
        if "合同摘要" in parsed:
            st.markdown("#### 📝 合同摘要")
            st.info(parsed["合同摘要"])

        # 显示风险条款
        if "风险条款" in parsed and parsed["风险条款"]:
            st.markdown("#### ⚠️ 风险条款")

            for i, risk in enumerate(parsed["风险条款"], 1):
                risk_level = risk.get("风险等级", "中")
                if risk_level == "高":
                    css_class = "risk-high"
                    emoji = "🔴"
                elif risk_level == "中":
                    css_class = "risk-medium"
                    emoji = "🟠"
                else:
                    css_class = "risk-low"
                    emoji = "🟢"

                with st.expander(f"{emoji} 风险 {i}: {risk.get('风险类型', '未知风险')} [{risk_level}]", expanded=(risk_level == "高")):
                    st.markdown(f"""
                    <div class="{css_class}">
                    <p><strong>条款位置:</strong> {risk.get('条款位置', 'N/A')}</p>
                    <p><strong>条款内容:</strong> {risk.get('条款内容', 'N/A')}</p>
                    <p><strong>风险描述:</strong> {risk.get('风险描述', 'N/A')}</p>
                    <p><strong>修改建议:</strong> {risk.get('修改建议', 'N/A')}</p>
                    <p><strong>法律依据:</strong> {risk.get('法律依据', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)

        # 显示总体评估
        if "总体评估" in parsed:
            st.markdown("#### 📊 总体评估")
            assessment = parsed["总体评估"]

            col1, col2 = st.columns(2)
            with col1:
                if "主要风险" in assessment:
                    st.markdown("**主要风险:**")
                    for risk in assessment["主要风险"]:
                        st.markdown(f"- {risk}")

            with col2:
                if "签约建议" in assessment:
                    st.markdown(f"**签约建议:** {assessment['签约建议']}")

        # 显示注意事项
        if "注意事项" in parsed and parsed["注意事项"]:
            st.markdown("#### 📌 注意事项")
            for item in parsed["注意事项"]:
                st.markdown(f"- {item}")

    except (json.JSONDecodeError, KeyError):
        # 无法解析为JSON，直接显示原始结果
        st.markdown(result.content)

    # 导出选项
    st.markdown("### 📥 导出报告")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "下载文本报告",
            result.content,
            file_name=f"合同审查报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )


def show_legal_qa():
    """显示法律咨询界面"""
    st.markdown('<h2 class="sub-header">💬 法律咨询</h2>', unsafe_allow_html=True)
    st.markdown("输入您的法律问题，AI将为您提供专业解答。")

    # 初始化Agent
    agent = init_legal_qa_agent()
    if not agent:
        return

    # 初始化对话历史
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # 初始化输入框内容
    if "qa_input" not in st.session_state:
        st.session_state.qa_input = ""

    # 显示对话历史
    if st.session_state.qa_history:
        st.markdown("### 📜 对话历史")
        for i, (q, a) in enumerate(st.session_state.qa_history):
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.markdown(a)
        st.markdown("---")

    # 快捷问题按钮（放在输入区域之前）
    st.markdown("### 🔥 常见问题")
    quick_questions = [
        "违约金过高怎么办？",
        "劳动合同解除赔偿",
        "民间借贷利息上限",
    ]
    quick_cols = st.columns(len(quick_questions))
    for i, qq in enumerate(quick_questions):
        with quick_cols[i]:
            if st.button(qq, key=f"quick_{i}", use_container_width=True):
                st.session_state.qa_input = qq  # 直接设置输入框的值
                st.rerun()

    st.markdown("---")

    # 输入区域
    col1, col2 = st.columns([3, 1])

    with col1:
        question = st.text_area(
            "请输入您的法律问题",
            height=120,
            placeholder="例如：合同违约金过高怎么办？劳动仲裁的时效是多久？",
            key="qa_input"
        )

    with col2:
        st.markdown("### 选项")
        answer_mode = st.radio(
            "回答模式",
            ["详细回答", "快速回答"],
            label_visibility="collapsed"
        )

    # 提交按钮
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn1:
        submit_btn = st.button("💬 提交问题", type="primary", use_container_width=True)
    with col_btn3:
        clear_btn = st.button("🗑️ 清空对话", use_container_width=True)

    # 清空对话
    if clear_btn:
        st.session_state.qa_history = []
        st.session_state.qa_input = ""
        agent.clear_conversation()
        st.rerun()

    # 处理提问
    if submit_btn and question.strip():
        with st.spinner("正在思考中..."):
            try:
                # 执行咨询
                result = agent.execute({
                    "question": question,
                    "simple_mode": answer_mode == "快速回答"
                })

                if result.success:
                    # 保存到历史
                    st.session_state.qa_history.append((question, result.content))

                    # 显示回答
                    with st.chat_message("user"):
                        st.write(question)
                    with st.chat_message("assistant"):
                        display_legal_answer(result)
                else:
                    st.error(f"回答失败: {result.error_message}")

            except Exception as e:
                st.error(f"发生错误: {str(e)}")


def display_legal_answer(result):
    """显示法律咨询回答"""
    # 显示法律依据
    if result.sources:
        st.markdown("**📚 引用法条:**")
        for source in result.sources:
            # 支持字典格式和字符串格式
            law = source.get("law", str(source)) if isinstance(source, dict) else str(source)
            st.markdown(f"- {law}")
        st.markdown("---")

    # 显示回答内容
    st.markdown(result.content)

    # 下载按钮
    st.download_button(
        "📥 下载回答",
        result.content,
        file_name=f"法律咨询_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )


def show_case_analysis():
    """显示案例分析界面"""
    st.markdown('<h2 class="sub-header">📊 案例分析</h2>', unsafe_allow_html=True)
    st.markdown("输入案件内容，AI将为您进行专业分析，包括争议焦点、法律适用、裁判预测等。")

    # 初始化Agent
    agent = init_case_agent()
    if not agent:
        return

    # 案例输入区域
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 输入案件内容")
        case_text = st.text_area(
            "粘贴案件内容",
            height=250,
            placeholder="请详细描述案件事实，包括：\n- 当事人信息\n- 案件经过\n- 争议事项\n- 诉求等..."
        )

    with col2:
        st.markdown("### 分析选项")
        analysis_type = st.radio(
            "分析类型",
            ["完整分析", "争议焦点", "裁判预测"],
            help="完整分析提供全面报告，争议焦点提取核心争议，裁判预测分析可能结果"
        )

        st.markdown("### 分析深度")
        analysis_depth = st.radio(
            "分析深度",
            ["详细分析", "快速分析"],
            label_visibility="collapsed"
        )

    # 分析按钮
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        analyze_button = st.button("🔍 开始分析", type="primary", use_container_width=True)

    # 处理分析请求
    if analyze_button:
        if not case_text.strip():
            st.warning("请输入案件内容")
            return

        with st.spinner("正在分析案件，请稍候..."):
            try:
                # 映射分析类型
                type_map = {
                    "完整分析": "full",
                    "争议焦点": "focus",
                    "裁判预测": "prediction"
                }

                result = agent.execute({
                    "case_content": case_text,
                    "simple_mode": analysis_depth == "快速分析",
                    "analysis_type": type_map.get(analysis_type, "full")
                })

                if result.success:
                    display_case_result(result)
                else:
                    st.error(f"分析失败: {result.error_message}")

            except Exception as e:
                st.error(f"发生错误: {str(e)}")


def display_case_result(result):
    """显示案例分析结果"""
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📋 分析报告</h2>', unsafe_allow_html=True)

    # 顶部统计
    col1, col2, col3 = st.columns(3)
    with col1:
        case_type = result.metadata.get("case_type", "待分析")
        st.metric("案件类型", case_type if case_type else "待分析")

    with col2:
        focus_count = result.metadata.get("focus_count", 0)
        st.metric("争议焦点", f"{focus_count} 项")

    with col3:
        rag_status = "已启用" if result.metadata.get("rag_enabled") else "未启用"
        st.metric("RAG增强", rag_status)

    # 显示法律依据
    if result.sources:
        st.markdown("### 📚 相关法条")
        for source in result.sources:
            law = source.get("law", str(source)) if isinstance(source, dict) else str(source)
            st.markdown(f"- {law}")
        st.markdown("---")

    # 显示分析内容
    st.markdown("### 分析详情")
    st.markdown(result.content)

    # 下载按钮
    st.markdown("### 📥 导出报告")
    st.download_button(
        "下载分析报告",
        result.content,
        file_name=f"案例分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )


if __name__ == "__main__":
    main()
