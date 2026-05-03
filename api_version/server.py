import os
import sys
from dotenv import load_dotenv
from fastmcp import FastMCP
from mem0 import Memory
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 1. 强制设定工作目录，防止 Cursor 调用时路径错乱
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ENV_PATH = os.path.join(BASE_DIR, ".env")
# Persistent mem0/qdrant store lives inside the project, NOT /tmp (which macOS may wipe).
MEM0_QDRANT_PATH = os.path.join(BASE_DIR, "data", "mem0_qdrant")

# 2. 加载环境变量 (读取 .env)
load_dotenv(ENV_PATH)

# Fail fast if the OpenAI key is missing — Mem0 needs it to build the embedding model.
# Without this check, MCP stdio gets polluted by a noisy traceback during startup.
if not os.environ.get("OPENAI_API_KEY"):
    print(
        "❌ OPENAI_API_KEY 未设置。请在 .env 中写成: OPENAI_API_KEY=sk-...",
        file=sys.stderr,
    )
    sys.exit(1)

# 3. 初始化 MCP 服务器
mcp = FastMCP("M4_Max_Context_Server")

# 4. 初始化 Memory 记忆系统
# IMPORTANT: mem0's default qdrant path is /tmp/qdrant, which macOS periodically wipes
# and which also gets locked per-process. Pin it to a project-local persistent dir.
# 所有的系统日志输出必须用 sys.stderr，绝不能用标准输出 print，否则会破坏 MCP 的 JSON-RPC 通信！
MEM0_CONFIG = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mem0_m4",
            "path": MEM0_QDRANT_PATH,
        },
    },
}
try:
    memory = Memory.from_config(MEM0_CONFIG)
except Exception as e:
    print(f"❌ Mem0 初始化失败: {e}", file=sys.stderr)
    sys.exit(1)
USER_ID = "m4_mac_user"

# 5. 初始化 RAG 知识库
print("正在构建本地知识库索引...", file=sys.stderr)
try:
    if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
        # 读取 data 文件夹里的文本并向量化
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        print("✅ 知识库构建成功！", file=sys.stderr)
    else:
        query_engine = None
except Exception as e:
    print(f"❌ RAG 初始化失败 (请检查 API Key 是否有效): {e}", file=sys.stderr)
    query_engine = None

# =========================================================
# 下面是将 Memory 和 RAG 封装为大模型(Cursor)可以调用的工具
# =========================================================

@mcp.tool()
def save_memory(fact: str) -> str:
    """
    当用户告诉你他的个人偏好、背景信息、代码习惯，或者要求你"记住"某事时，调用此工具。
    参数 fact: 需要记住的具体事实。
    """
    # infer=False: skip mem0's LLM fact-extraction step so the raw fact is ALWAYS persisted
    # (otherwise mem0 may silently NOOP short / "uninteresting" inputs).
    response = memory.add(fact, user_id=USER_ID, infer=False)
    items = response.get("results", []) if isinstance(response, dict) else response

    if not items:
        return f"⚠️ mem0 判定无需写入（NOOP），未持久化: {fact}"

    events = [item.get("event") for item in items]
    stored_ids = [item.get("id") for item in items]
    return (
        f"✅ 记忆已写入（events={events}, ids={stored_ids}）: {fact}"
    )

@mcp.tool()
def recall_memory(query: str) -> str:
    """
    【强制动作 / CRITICAL MUST】在编写任何代码或回答问题之前，必须优先调用此工具检索用户的代码风格、偏好和历史习惯。
    参数 query: 检索关键词（如"代码风格"、"错误处理"、"习惯"）。
    """
    # New Mem0 API: scope must come via `filters`, and results are wrapped in {"results": [...]}.
    response = memory.search(query, filters={"user_id": USER_ID})
    items = response.get("results", []) if isinstance(response, dict) else response
    if not items:
        return "没有找到相关的历史记忆。"

    memories = [item["memory"] for item in items]
    return "🧠 检索到的长期记忆:\n" + "\n".join(memories)

@mcp.tool()
def search_docs(query: str) -> str:
    """
    【强制动作 / CRITICAL MUST】只要涉及数据库、密码、框架规范、命名规则等业务代码，必须调用此工具进行本地检索。绝对不能猜测密码或命名！
    参数 query: 用户的提问，如"数据库规范"、"内部密码"。
    """
    if not query_engine:
        return "本地知识库未初始化。"
    
    response = query_engine.query(query)
    return f"📚 RAG 检索结果:\n{str(response)}"

if __name__ == "__main__":
    # MCP 协议标准：使用 stdio 方式进行进程间通信
    mcp.run(transport="stdio")