# app_rag_agent.py  —  Agent版（Function Calling + RAG 双工具）
import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv

# ========== 基础配置 ==========
load_dotenv()
My_Key = os.getenv("SILICONFLOW_KEY")

# 嵌入模型
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=My_Key,
    base_url="https://api.siliconflow.cn/v1"
)

# 大模型（LangChain 封装，保留用于其他场景）
llm = ChatOpenAI(
    model="Qwen/Qwen3-8B",
    api_key=My_Key,
    base_url="https://api.siliconflow.cn/v1"
)

# 原始 OpenAI 客户端（用于工具调用 Loop）
client = OpenAI(
    api_key=My_Key,
    base_url="https://api.siliconflow.cn/v1",
)

# ========== 工具函数定义 ==========

# 工具1：设备状态查询（模拟真实接口）
def get_device_status(device_name: str) -> str:
    """查询指定设备的当前运行状态。真实项目中替换为查数据库/调PLC接口"""
    table = {
        "设备A": "正常运行",
        "设备B": "停机维修",
        "设备C": "正常运行",
    }
    return table.get(device_name, f"未找到设备：{device_name}")


# retriever 全局变量（后面初始化）
retriever = None


# 工具2：知识库检索（RAG）
def search_knowledge_base(query: str) -> str:
    """在设备维修手册知识库中搜索相关内容"""
    if retriever is None:
        return "知识库尚未初始化，请稍后再试。"
    retrieved_docs = retriever.invoke(query)
    context_text = "\n".join([doc.page_content for doc in retrieved_docs])
    return context_text if context_text else "未在知识库中找到相关内容。"


# ========== 工具 Schema 定义（告诉大模型有哪些工具可用） ==========
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_device_status",
            "description": (
                "查询指定设备的当前运行状态。"
                "当用户询问某台设备是否正常运行、是否停机、状态如何时，必须调用此函数。"
                "例如：'设备A现在什么状态？'、'设备B停机了吗？'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "device_name": {
                        "type": "string",
                        "description": "设备名称，例如 '设备A'、'设备B'"
                    }
                },
                "required": ["device_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "在设备维修手册知识库中搜索相关内容。"
                "当用户询问设备维修方法、故障排除步骤、设备参数、保养说明、"
                "操作规范等手册相关内容时，必须调用此函数。"
                "例如：'设备A怎么开机？'、'报错E01怎么处理？'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题描述，用于在知识库中检索相关内容"
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# 函数名 -> 真实函数 的映射表
FUNCTION_MAP = {
    "get_device_status": get_device_status,
    "search_knowledge_base": search_knowledge_base,
}


# ========== RAG 知识库初始化 ==========
@st.cache_resource
def init_rag_system():
    """加载PDF、文本分割、构建向量库，返回 retriever"""
    loader = PyPDFLoader("manual.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=20
    )
    chunks = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings
    )
    retriever_instance = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )
    return retriever_instance


# ========== Agent 对话引擎（核心 Loop） ==========
def agent_chat(user_input: str, history_messages: list) -> tuple:
    """
    Agent 主循环：发送消息 -> 大模型决定调工具 -> 执行工具 -> 喂回结果 -> 循环...
    返回: (最终回答文本, 工具调用日志列表)
    """
    system_prompt = (
        "你是一个工业设备维修专家助手。你拥有以下工具：\n"
        "1. get_device_status(device_name) - 查询设备实时运行状态\n"
        "2. search_knowledge_base(query) - 搜索设备维修手册知识库\n\n"
        "你的工作原则：\n"
        "- 用户问设备状态 -> 调用 get_device_status\n"
        "- 用户问维修方法/手册内容 -> 调用 search_knowledge_base\n"
        "- 用户问题同时涉及两方面 -> 连续调用多个工具，综合回答\n"
        "- 如果问题既不需要工具也不需要检索 -> 直接回答\n"
        "- 回答要简洁、专业、针对用户问题"
    )

    # 构建消息历史
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # 追加历史对话（最近3轮，每轮2条消息 = 6条）
    for msg in history_messages[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # 当前用户消息
    messages.append({"role": "user", "content": user_input})

    # Agent 循环
    max_rounds = 5
    rounds = 0
    tool_log = []  # 记录工具调用过程

    while rounds < max_rounds:
        rounds += 1

        resp = client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",  # 模型自主决定是否需要调用工具
        )

        msg = resp.choices[0].message

        # ---- 判断：模型是否需要调用工具 ----
        if msg.tool_calls:
            # 1. 把 assistant 的 tool_calls 消息加回历史（重要！否则 tool 结果对不上）
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })

            # 2. 执行所有工具调用
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")

                # 记录日志
                tool_log.append(f"[工具调用] {name}")
                tool_log.append(f"  参数: {args}")

                fn = FUNCTION_MAP.get(name)
                if fn:
                    try:
                        result = fn(**args)
                    except Exception as e:
                        result = f"工具执行异常: {e}"
                else:
                    result = f"未知工具: {name}"

                tool_log.append(f"  结果: {result}")
                tool_log.append("")

                # 3. 把工具执行结果以 role="tool" 喂回对话
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

            continue  # 回到循环开头，让模型基于工具结果生成最终回答

        # ---- 模型不需要工具，直接返回文本回答 ----
        return msg.content, tool_log

    # 超过最大轮数
    return "处理超时，请换个问法", tool_log


# ========== Streamlit UI ==========

# 初始化 RAG 知识库
with st.spinner("正在加载知识库..."):
    retriever = init_rag_system()

# 侧边栏
with st.sidebar:
    st.header("关于本系统")
    st.write("基于 LangChain + Chroma 搭建的工业设备维修 Agent 助手")
    st.write("支持 Function Calling 工具调用")
    st.write("开发者：所以然")
    st.write("当前版本：v2.0 (Agent版)")

# 标题
st.title("设备维修专家AI（Agent版）")

# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 用户输入
user_query = st.chat_input("请输入故障设备代码或问题：")

if user_query:
    # 存储用户消息
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )
    with st.chat_message("user"):
        st.write(user_query)

    # Agent 处理（自动判断走工具还是直接回答）
    with st.spinner("Agent 正在思考并查询..."):
        answer, tool_log = agent_chat(
            user_query, st.session_state.messages
        )

    # 展示 Agent 执行过程（折叠面板）
    if tool_log:
        with st.expander("查看 Agent 执行过程"):
            for log_line in tool_log:
                st.write(log_line)

    # 存储并展示 AI 回答
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    with st.chat_message("assistant"):
        st.write(answer)