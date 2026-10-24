# 应用工具库
import os
from langchain_community.document_loaders import  PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import streamlit as st
from dotenv import load_dotenv


load_dotenv()
My_Key = os.getenv("SILICONFLOW_KEY")
# 配置大模型
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=My_Key,
    base_url="https://api.siliconflow.cn/v1"
)
llm = ChatOpenAI(
    model="Qwen/Qwen3-8B",
    api_key=My_Key,
    base_url="https://api.siliconflow.cn/v1"
)

@st.cache_resource
def init_rag_system():
    loader = PyPDFLoader("manual.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever

# 创建文本加载器，指定要读取的文件和编码，此时还没真正读文件
loader = PyPDFLoader("manual.pdf")

retriever = init_rag_system()
# 设置提示词
prompt = ChatPromptTemplate([
    ("system", "你是一个设备专家。请严格根据以下知识库回答：\n\n{context}"),
    ("user", "{question}")
])
# 通过管道符把提示词和大模型链接起来
chain = prompt | llm
# 初始化状态,创建一个空列表，有一个疑问st.session_state是列表还是字典，为什么选择它？（全局的大仓库，特殊的字典）
if "messages" not in st.session_state:
    st.session_state.messages = []
with st.sidebar:
    st.header("关于本系统")
    st.write("这是一个基于LangChain和Chroma搭建的工业设备维修RAG助手")
    st.write("开发者：所以然")
    st.write("当前版本：v1.0（支持PDF解析）")
# 网页的标题
st.title("设备维修专家AI（RAG版）")
# 遍历并显示历史对话，这边with这两行不理解，（with会在ui里画出关于用户或回答机器人专门的聊天框和头像）
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
# 配置输入并引导用户输入的提示词
user_query = st.chat_input("请输入故障设备代码或问题：")

# 如果用户输入了
if user_query:
    # 往列表里存储用户名及他所提的问题
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
    # 创建空字符串来存储用户的上文内容，作为下文的补充信息
    history_context = ""
    # 选中messages列表里从右往左第n个字典里的'content'所对应的键对值， 列表顺序[用户1, ai1, 用户2, ai2]
    if len(st.session_state.messages) > 2:
        prev_user = st.session_state.messages[-3]["content"]
        prev_ai = st.session_state.messages[-2]["content"]
        history_context = f"上轮对话：用户问：'{prev_user}', 你回答：'{prev_ai}'"
    # 都是字符串，直接用+缝合
    final_search_query = history_context + " " + user_query
    # 显示滚轮加正在查询手册
    with st.spinner("正在查询手册"):
        # 寻找相似度前五的块
        retrieved_docs = retriever.invoke(final_search_query)
        # 将检索到的内容进行上下，这里的jion不知道用法
        context_text = "\n".join([doc.page_content for doc in retrieved_docs])
        # 把从知识库里相似度前五的块和用户的问题一起打包给含有提示词的大模型
        result = chain.invoke({
            "context": context_text,
            "question": final_search_query
        })
        # 往列表里存ai名及对应对话的回答内容
        st.session_state.messages.append({"role": "assistant", "content": result.content})
    with st.chat_message("assistant"):
        st.success("回答如下：")
        # 写下大模型的回答内容
        st.write(result.content)