"""Streamlit 멀티유저·멀티세션 RAG 챗봇: user 테이블 로그인, chat_sessions/chat_messages, Supabase 벡터."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import bcrypt
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"
LOG_DIR = REPO_ROOT / "logs"

load_dotenv(dotenv_path=ENV_PATH)

KEY_NAMES = ("SUPABASE_URL", "SUPABASE_ANON_KEY", "OPENAI_API_KEY")


def _secret_get(name: str) -> str:
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            v = st.secrets[name]
            if v is not None and str(v).strip():
                return str(v).strip()
    except (FileNotFoundError, RuntimeError, KeyError, AttributeError):
        pass
    return os.getenv(name, "").strip()


def config_get(name: str) -> str:
    """우선순위: st.secrets → 환경변수(.env 포함)."""
    s = _secret_get(name)
    if s:
        return s
    return os.getenv(name, "").strip()


def missing_keys() -> list[str]:
    return [k for k in KEY_NAMES if not config_get(k)]


def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_name = f"multiusers_{datetime.now().strftime('%Y%m%d')}.log"
    log_path = LOG_DIR / log_name

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)

    for name in (
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "langchain",
        "langchain_openai",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("multiusers")


logger = _setup_logging()

ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 AI 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요. 주요 주제는 #, 세부는 ##, 구체 설명은 ###.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 구분선(---, ===, ___)은 사용하지 마세요.
- 취소선(~~텍스트~~)은 사용하지 마세요.
- 참조 표시, 각주, 출처 문구, URL 인용 문장은 넣지 마세요.
"""


def remove_separators(text: str) -> str:
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def get_supabase() -> Client | None:
    url = config_get("SUPABASE_URL")
    key = config_get("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def get_llm(model_name: str, temperature: float = 0.7) -> ChatOpenAI:
    if model_name != "gpt-4o-mini":
        raise ValueError("이 앱은 gpt-4o-mini만 지원합니다.")
    key = config_get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, api_key=key)


def get_embeddings() -> OpenAIEmbeddings:
    api_key = config_get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 필요합니다.")
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536,
        api_key=api_key,
    )


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def register_user(sb: Client, login_id: str, password: str) -> tuple[bool, str]:
    lid = login_id.strip()
    if len(lid) < 2:
        return False, "로그인 ID는 2자 이상이어야 합니다."
    if len(password) < 4:
        return False, "비밀번호는 4자 이상이어야 합니다."
    dup = sb.table("user").select("id").eq("login_id", lid).limit(1).execute()
    if dup.data:
        return False, "이미 사용 중인 로그인 ID입니다."
    ph = hash_password(password)
    sb.table("user").insert({"login_id": lid, "password_hash": ph}).execute()
    return True, "회원가입이 완료되었습니다. 로그인해 주세요."


def login_user(sb: Client, login_id: str, password: str) -> dict[str, Any] | None:
    lid = login_id.strip()
    res = sb.table("user").select("id,login_id,password_hash").eq("login_id", lid).limit(1).execute()
    rows = res.data or []
    if not rows:
        return None
    row = rows[0]
    if not verify_password(password, str(row["password_hash"])):
        return None
    return {"id": str(row["id"]), "login_id": str(row["login_id"])}


def list_sessions(sb: Client, user_id: str) -> list[dict[str, Any]]:
    res = (
        sb.table("chat_sessions")
        .select("id,title,updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return list(res.data or [])


def fetch_session(sb: Client, user_id: str, session_id: str) -> dict[str, Any] | None:
    sres = (
        sb.table("chat_sessions")
        .select("*")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    rows = sres.data or []
    if not rows:
        return None
    row = dict(rows[0])
    mres = (
        sb.table("chat_messages")
        .select("role,content,position")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("position", desc=False)
        .execute()
    )
    msgs = [{"role": m["role"], "content": m["content"]} for m in (mres.data or [])]
    row["messages"] = msgs
    return row


def insert_session_row(sb: Client, user_id: str, title: str, messages: list[dict[str, str]]) -> str:
    sres = sb.table("chat_sessions").insert({"user_id": user_id, "title": title}).execute()
    srows = sres.data or []
    if not srows:
        raise RuntimeError("세션 INSERT 후 id를 받지 못했습니다.")
    sid = str(srows[0]["id"])
    rows = [
        {
            "user_id": user_id,
            "session_id": sid,
            "role": m["role"],
            "content": m["content"],
            "position": i,
        }
        for i, m in enumerate(messages)
    ]
    if rows:
        sb.table("chat_messages").insert(rows).execute()
    return sid


def update_session_messages(sb: Client, user_id: str, session_id: str, messages: list[dict[str, str]]) -> None:
    sb.table("chat_messages").delete().eq("session_id", session_id).eq("user_id", user_id).execute()
    rows = [
        {
            "user_id": user_id,
            "session_id": session_id,
            "role": m["role"],
            "content": m["content"],
            "position": i,
        }
        for i, m in enumerate(messages)
    ]
    if rows:
        sb.table("chat_messages").insert(rows).execute()
    sb.table("chat_sessions").update({"updated_at": datetime.now().isoformat()}).eq("id", session_id).eq(
        "user_id", user_id
    ).execute()


def delete_session_row(sb: Client, user_id: str, session_id: str) -> None:
    sb.table("chat_sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()


def delete_draft_vectors(sb: Client, user_id: str, draft_key: str) -> None:
    res = (
        sb.table("vector_documents")
        .select("id,metadata")
        .eq("user_id", user_id)
        .is_("session_id", "null")
        .execute()
    )
    ids: list[int] = []
    for row in res.data or []:
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        if str(meta.get("draft_key", "")) == draft_key:
            ids.append(int(row["id"]))
    for vid in ids:
        sb.table("vector_documents").delete().eq("id", vid).execute()


def _vector_to_pg(emb: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" if not math.isnan(x) else "0" for x in emb) + "]"


def insert_vector_batch(sb: Client, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    for row in rows:
        ins = dict(row)
        emb = ins.get("embedding")
        if isinstance(emb, list):
            ins["embedding"] = _vector_to_pg([float(x) for x in emb])
        sb.table("vector_documents").insert(ins).execute()


def duplicate_vectors_for_session(
    sb: Client, user_id: str, source_session_id: str, target_session_id: str
) -> None:
    res = (
        sb.table("vector_documents")
        .select("file_name,content,metadata,embedding")
        .eq("user_id", user_id)
        .eq("session_id", source_session_id)
        .execute()
    )
    batch: list[dict[str, Any]] = []
    for row in res.data or []:
        emb = row.get("embedding")
        if isinstance(emb, str):
            pass
        elif isinstance(emb, list):
            emb = _vector_to_pg([float(x) for x in emb])
        else:
            continue
        meta = row.get("metadata") or {}
        if isinstance(meta, dict) and "draft_key" in meta:
            meta = {k: v for k, v in meta.items() if k != "draft_key"}
        batch.append(
            {
                "user_id": user_id,
                "session_id": target_session_id,
                "file_name": row["file_name"],
                "content": row.get("content"),
                "metadata": meta,
                "embedding": emb,
            }
        )
        if len(batch) >= 10:
            sb.table("vector_documents").insert(batch).execute()
            batch = []
    if batch:
        sb.table("vector_documents").insert(batch).execute()


def attach_draft_vectors_to_session(sb: Client, user_id: str, draft_key: str, new_session_id: str) -> None:
    res = (
        sb.table("vector_documents")
        .select("id,metadata")
        .eq("user_id", user_id)
        .is_("session_id", "null")
        .execute()
    )
    for row in res.data or []:
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        if str(meta.get("draft_key", "")) != draft_key:
            continue
        new_meta = {k: v for k, v in meta.items() if k != "draft_key"}
        sb.table("vector_documents").update({"session_id": new_session_id, "metadata": new_meta}).eq(
            "id", row["id"]
        ).eq("user_id", user_id).execute()


def process_pdfs_to_supabase(
    sb: Client,
    user_id: str,
    uploaded_files: list[Any],
    *,
    session_id: str | None,
    draft_key: str,
    embeddings: OpenAIEmbeddings,
) -> list[str]:
    if not uploaded_files:
        return []

    all_docs: list[Document] = []
    names: list[str] = []
    for uf in uploaded_files:
        suffix = Path(uf.name).suffix.lower() or ".pdf"
        names.append(uf.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            for d in pages:
                d.metadata = dict(d.metadata or {})
                d.metadata["file_name"] = uf.name
                d.metadata["draft_key"] = draft_key
            all_docs.extend(pages)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if not all_docs:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(all_docs)

    texts = [d.page_content for d in splits]
    metas = [d.metadata for d in splits]

    batch_size = 10
    for i in range(0, len(texts), batch_size):
        chunk_texts = texts[i : i + batch_size]
        chunk_meta = metas[i : i + batch_size]
        embs = embeddings.embed_documents(chunk_texts)
        rows: list[dict[str, Any]] = []
        for text, meta, emb in zip(chunk_texts, chunk_meta, embs, strict=True):
            fn = str(meta.get("file_name") or "unknown.pdf")
            row: dict[str, Any] = {
                "user_id": user_id,
                "file_name": fn,
                "content": text,
                "metadata": {k: v for k, v in meta.items() if v is not None},
                "embedding": emb,
            }
            if session_id:
                row["session_id"] = session_id
            else:
                row["session_id"] = None
            rows.append(row)
        insert_vector_batch(sb, rows)

    return names


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _parse_embedding(val: Any) -> list[float] | None:
    if isinstance(val, list):
        return [float(x) for x in val]
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1]
            if not inner.strip():
                return []
            return [float(x) for x in inner.split(",")]
    return None


def retrieve_documents(
    sb: Client,
    user_id: str,
    embeddings: OpenAIEmbeddings,
    question: str,
    *,
    session_id: str | None,
    draft_key: str,
    match_count: int = 10,
) -> list[Document]:
    qemb = embeddings.embed_query(question)

    try:
        res = sb.rpc(
            "match_vector_documents",
            {
                "query_embedding": _vector_to_pg(qemb),
                "match_count": match_count,
                "filter_session_id": session_id,
                "filter_draft_key": draft_key if not session_id else None,
                "filter_user_id": user_id,
            },
        ).execute()
        docs: list[Document] = []
        for row in res.data or []:
            docs.append(
                Document(
                    page_content=str(row.get("content") or ""),
                    metadata={
                        "file_name": row.get("file_name"),
                        "id": row.get("id"),
                        "similarity": row.get("similarity"),
                    },
                )
            )
        if docs:
            return docs
    except Exception as exc:  # noqa: BLE001
        logger.warning("RPC match_vector_documents failed, using fallback: %s", exc)

    q = sb.table("vector_documents").select("id,file_name,content,metadata,embedding,session_id").eq(
        "user_id", user_id
    ).limit(5000).execute()
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in q.data or []:
        sid = row.get("session_id")
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        ok = False
        if session_id and sid and str(sid) == str(session_id):
            ok = True
        elif not session_id and sid is None and str(meta.get("draft_key", "")) == draft_key:
            ok = True
        if not ok:
            continue
        emb = _parse_embedding(row.get("embedding"))
        if not emb:
            continue
        scored.append((_cosine(qemb, emb), row))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[Document] = []
    for _, row in scored[:match_count]:
        out.append(
            Document(
                page_content=str(row.get("content") or ""),
                metadata={"file_name": row.get("file_name")},
            )
        )
    return out


def list_vector_filenames(
    sb: Client,
    user_id: str,
    *,
    session_id: str | None,
    draft_key: str,
) -> list[str]:
    res = sb.table("vector_documents").select("file_name,session_id,metadata").eq("user_id", user_id).execute()
    names: set[str] = set()
    for row in res.data or []:
        sid = row.get("session_id")
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        if session_id and sid and str(sid) == str(session_id):
            names.add(str(row.get("file_name") or ""))
        elif not session_id and sid is None and str(meta.get("draft_key", "")) == draft_key:
            names.add(str(row.get("file_name") or ""))
    return sorted(n for n in names if n)


def apply_session_to_state(row: dict[str, Any]) -> None:
    raw = row.get("messages") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = []
    msgs = [dict(m) for m in raw if isinstance(m, dict)]
    st.session_state.chat_history = msgs
    st.session_state.conversation_memory = msgs[-50:]
    st.session_state.current_session_id = str(row["id"])


def _format_memory_block(messages: list[dict[str, str]], max_items: int = 50) -> str:
    tail = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "사용자" if role == "user" else "어시스턴트"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _build_rag_messages(
    question: str,
    context: str,
    memory_text: str,
) -> list[SystemMessage | HumanMessage]:
    sys = f"""{ANSWER_STYLE_SYSTEM}

아래 [대화 맥락]과 [참고 문서]를 활용해 답하세요. 참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.
[대화 맥락]
{memory_text or "(없음)"}

[참고 문서]
{context}
"""
    return [SystemMessage(content=sys), HumanMessage(content=question)]


def _generate_followup_section(llm: ChatOpenAI, user_q: str, answer: str) -> str:
    trimmed = answer[:8000]
    prompt = (
        "다음 사용자 질문과 답변을 바탕으로, 이어서 물어볼 만한 후속 질문을 한국어로 정확히 3개만 작성하세요.\n"
        "형식:\n1. ...\n2. ...\n3. ...\n"
        "설명 문장이나 다른 텍스트는 출력하지 마세요.\n\n"
        f"[사용자 질문]\n{user_q}\n\n[답변]\n{trimmed}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        raw = getattr(out, "content", str(out)) or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Follow-up generation failed: %s", exc)
        return ""

    raw = remove_separators(str(raw))
    if not raw.strip():
        return ""
    return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{raw.strip()}\n"


def _append_followup_questions(
    llm: ChatOpenAI,
    user_q: str,
    answer_without_follow: str,
) -> str:
    return _generate_followup_section(llm, user_q, answer_without_follow)


def generate_session_title(llm: ChatOpenAI, messages: list[dict[str, str]]) -> str:
    first_user = ""
    first_asst = ""
    for m in messages:
        if m.get("role") == "user" and not first_user:
            first_user = (m.get("content") or "").strip()[:2000]
        elif m.get("role") == "assistant" and first_user and not first_asst:
            first_asst = (m.get("content") or "").strip()[:2000]
            break
    if not first_user:
        return f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    prompt = (
        "다음 첫 질문과 첫 답변을 한 줄로 요약해 세션 제목만 출력하세요. "
        "30자 이내, 따옴표나 접두어 없이 제목만.\n\n"
        f"[질문]\n{first_user}\n\n[답변]\n{first_asst or '(답변 없음)'}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        title = str(getattr(out, "content", out) or "").strip()
        title = re.sub(r"^[\"']|[\"']$", "", title)
        return title[:120] if title else f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Title generation failed: %s", exc)
        return f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"


def _init_session() -> None:
    defaults: dict[str, Any] = {
        "chat_history": [],
        "conversation_memory": [],
        "current_session_id": None,
        "draft_key": str(uuid.uuid4()),
        "processed_names": [],
        "vector_file_names": [],
        "_session_pick": None,
        "_vectordb_expanded": False,
        "_vectordb_list": None,
        "logged_user_id": None,
        "logged_login_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _logout() -> None:
    st.session_state.logged_user_id = None
    st.session_state.logged_login_id = None
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.current_session_id = None
    st.session_state.draft_key = str(uuid.uuid4())
    st.session_state.processed_names = []
    st.session_state.vector_file_names = []
    st.session_state._session_pick = None


def main() -> None:
    st.set_page_config(
        page_title="기획예산처 RAG 챗봇",
        page_icon="📚",
        layout="wide",
    )
    _init_session()

    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div.stButton > button:first-child {
  background-color: #ff69b4;
  color: #ffffff;
}
</style>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            """
<div style="text-align:center; margin:0;">
  <span style="font-size:4rem !important; font-weight:700;">
    <span style="color:#1f77b4 !important;">기획예산처</span>
    <span style="color:#ff8c00 !important;">RAG 챗봇</span>
  </span>
</div>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()

    missing = missing_keys()
    sb = get_supabase() if not missing else None
    uid = st.session_state.logged_user_id

    with st.sidebar:
        st.markdown("### 계정")
        if not sb:
            st.info("Supabase 연결 정보를 설정하면 로그인할 수 있습니다.")
        elif not uid:
            tab_login, tab_reg = st.tabs(["로그인", "회원가입"])
            with tab_login:
                li = st.text_input("로그인 ID", key="login_id_input")
                lp = st.text_input("비밀번호", type="password", key="login_pw_input")
                if st.button("로그인", key="btn_login"):
                    user = login_user(sb, li, lp)
                    if user:
                        st.session_state.logged_user_id = user["id"]
                        st.session_state.logged_login_id = user["login_id"]
                        st.success("로그인되었습니다.")
                        st.rerun()
                    else:
                        st.error("로그인 ID 또는 비밀번호가 올바르지 않습니다.")
            with tab_reg:
                ri = st.text_input("새 로그인 ID", key="reg_id_input")
                rp = st.text_input("새 비밀번호", type="password", key="reg_pw_input")
                rp2 = st.text_input("비밀번호 확인", type="password", key="reg_pw2_input")
                if st.button("회원가입", key="btn_reg"):
                    if rp != rp2:
                        st.error("비밀번호가 일치하지 않습니다.")
                    else:
                        ok, msg = register_user(sb, ri, rp)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
        else:
            st.success(f"로그인: **{st.session_state.logged_login_id}**")
            if st.button("로그아웃", key="btn_logout"):
                _logout()
                st.rerun()

        openai_key = config_get("OPENAI_API_KEY")

        st.radio(
            "LLM 모델 선택",
            ("gpt-4o-mini",),
            index=0,
            disabled=True,
        )
        rag_choice = st.radio(
            "RAG (PDF 검색) 선택",
            ("사용 안 함", "RAG 사용"),
            index=0,
        )
        uploads = st.file_uploader(
            "PDF 파일 업로드",
            type=["pdf"],
            accept_multiple_files=True,
        )

        sessions: list[dict[str, Any]] = []
        if sb and uid:
            try:
                sessions = list_sessions(sb, uid)
            except Exception as exc:  # noqa: BLE001
                logger.warning("list_sessions: %s", exc)
                st.error(f"세션 목록을 불러오지 못했습니다: {exc}")

        labels: list[str] = ["(선택 없음)"]
        label_to_id: dict[str, str] = {}
        for s in sessions:
            tid = str(s["id"])
            title = str(s.get("title") or "제목 없음")
            updated = str(s.get("updated_at") or "")[:19]
            lab = f"{title} — {updated} — {tid[:8]}"
            labels.append(lab)
            label_to_id[lab] = tid

        pick = st.selectbox("세션 선택", labels, key="session_select_widget")
        if pick == "(선택 없음)":
            st.session_state._session_pick = None
        else:
            chosen_id = label_to_id.get(pick)
            if chosen_id and chosen_id != st.session_state.get("_session_pick"):
                st.session_state._session_pick = chosen_id
                if sb and uid:
                    try:
                        row = fetch_session(sb, uid, chosen_id)
                        if row:
                            apply_session_to_state(row)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("auto load session: %s", exc)
                        st.error(f"세션 로드 실패: {exc}")

        col_a, col_b = st.columns(2)
        with col_a:
            do_save = st.button("세션저장")
        with col_b:
            do_load = st.button("세션로드")
        col_c, col_d = st.columns(2)
        with col_c:
            do_delete = st.button("세션삭제")
        with col_d:
            do_reset_screen = st.button("화면초기화")
        do_vectordb = st.button("vectordb")

        if st.button("파일 처리하기"):
            if not uid:
                st.warning("먼저 로그인해 주세요.")
            elif not sb:
                st.error("Supabase 설정이 필요합니다.")
            elif not uploads:
                st.warning("업로드된 PDF가 없습니다.")
            elif not openai_key:
                st.error("OPENAI_API_KEY가 필요합니다.")
            else:
                try:
                    emb = get_embeddings()
                    sid = st.session_state.current_session_id
                    names = process_pdfs_to_supabase(
                        sb,
                        str(uid),
                        list(uploads),
                        session_id=sid,
                        draft_key=st.session_state.draft_key,
                        embeddings=emb,
                    )
                    st.session_state.processed_names = sorted(
                        set(st.session_state.processed_names) | set(names)
                    )
                    st.session_state.vector_file_names = list(dict.fromkeys(st.session_state.processed_names))
                    if sid:
                        update_session_messages(sb, str(uid), sid, list(st.session_state.chat_history))
                    st.success("PDF 처리 및 벡터 저장이 완료되었습니다.")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("PDF 처리 실패: %s", exc)
                    st.error(f"PDF 처리 중 오류: {exc}")

        if do_load and pick != "(선택 없음)" and sb and uid:
            cid = label_to_id.get(pick)
            if cid:
                try:
                    row = fetch_session(sb, uid, cid)
                    if row:
                        apply_session_to_state(row)
                        st.success("세션을 불러왔습니다.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"세션 로드 실패: {exc}")

        if do_save and sb and uid and openai_key:
            msgs = list(st.session_state.chat_history)
            if not msgs:
                st.warning("저장할 대화가 없습니다.")
            else:
                try:
                    llm = get_llm("gpt-4o-mini")
                    title = generate_session_title(llm, msgs)
                    cur = st.session_state.current_session_id
                    dk = st.session_state.draft_key
                    new_id = insert_session_row(sb, str(uid), title, msgs)
                    if cur:
                        duplicate_vectors_for_session(sb, str(uid), cur, new_id)
                    else:
                        attach_draft_vectors_to_session(sb, str(uid), dk, new_id)
                    st.session_state.current_session_id = new_id
                    st.session_state.draft_key = str(uuid.uuid4())
                    st.session_state._session_pick = new_id
                    st.success("새 세션이 저장되었습니다.")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("세션 저장 실패: %s", exc)
                    st.error(f"세션 저장 실패: {exc}")

        if do_delete and pick != "(선택 없음)" and sb and uid:
            cid = label_to_id.get(pick)
            if cid:
                try:
                    delete_session_row(sb, str(uid), cid)
                    if st.session_state.current_session_id == cid:
                        st.session_state.chat_history = []
                        st.session_state.conversation_memory = []
                        st.session_state.current_session_id = None
                        st.session_state.draft_key = str(uuid.uuid4())
                    st.success("세션이 삭제되었습니다.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"세션 삭제 실패: {exc}")

        if do_reset_screen:
            old_dk = st.session_state.draft_key
            if sb and uid and openai_key:
                try:
                    delete_draft_vectors(sb, str(uid), old_dk)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("draft vector cleanup: %s", exc)
            st.session_state.chat_history = []
            st.session_state.conversation_memory = []
            st.session_state.current_session_id = None
            st.session_state.processed_names = []
            st.session_state.vector_file_names = []
            st.session_state.draft_key = str(uuid.uuid4())
            st.session_state._session_pick = None
            st.rerun()

        if do_vectordb and sb and uid:
            try:
                names = list_vector_filenames(
                    sb,
                    str(uid),
                    session_id=st.session_state.current_session_id,
                    draft_key=st.session_state.draft_key,
                )
                st.session_state._vectordb_list = names
                st.session_state._vectordb_expanded = True
            except Exception as exc:  # noqa: BLE001
                st.error(f"vectordb 조회 실패: {exc}")

        if st.session_state.get("_vectordb_expanded") and st.session_state.get("_vectordb_list") is not None:
            st.markdown("**vectordb 파일명**")
            for n in st.session_state._vectordb_list:
                st.text(f"- {n}")

        if st.session_state.processed_names:
            st.markdown("**처리된 파일**")
            for name in st.session_state.processed_names:
                st.text(f"- {name}")

        mem_count = len(st.session_state.conversation_memory)
        cur_sid = st.session_state.current_session_id or "(없음)"
        settings_text = (
            f"모델: gpt-4o-mini\n"
            f"RAG: {rag_choice}\n"
            f"현재 세션 id: {cur_sid}\n"
            f"draft_key: {st.session_state.draft_key[:8]}…\n"
            f"대화 기록(메시지) 수: {mem_count}"
        )
        if missing:
            settings_text += "\n누락된 설정: " + ", ".join(missing)
        st.text(settings_text)

    if missing:
        st.warning(
            "다음 값을 Streamlit **Secrets** 또는 `AI-Education/.env`에 설정해 주세요: " + ", ".join(missing)
        )

    if not uid:
        st.info("사이드바에서 로그인하거나 회원가입한 뒤 챗봇을 사용할 수 있습니다.")
        return

    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = remove_separators(str(msg.get("content") or ""))
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    if not openai_key:
        st.error("OPENAI_API_KEY가 필요합니다.")
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.conversation_memory.append({"role": "user", "content": user_input})
    if len(st.session_state.conversation_memory) > 50:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-50:]

    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""

        try:
            llm = get_llm("gpt-4o-mini")
            if rag_choice == "RAG 사용" and sb and uid:
                mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
                emb = get_embeddings()
                docs = retrieve_documents(
                    sb,
                    str(uid),
                    emb,
                    user_input,
                    session_id=st.session_state.current_session_id,
                    draft_key=st.session_state.draft_key,
                    match_count=10,
                )
                if not docs:
                    full_answer = (
                        "# 안내\n\n"
                        "RAG를 사용하려면 PDF를 업로드한 뒤 **파일 처리하기**를 눌러 주세요."
                    )
                    placeholder.markdown(remove_separators(full_answer))
                else:
                    context = "\n\n".join(d.page_content for d in docs)
                    messages = _build_rag_messages(user_input, context, mem_txt)
                    acc = ""
                    for chunk in llm.stream(messages):
                        piece = getattr(chunk, "content", "") or ""
                        if piece:
                            acc += piece
                            placeholder.markdown(remove_separators(acc) + "▌")
                    full_answer = remove_separators(acc)
                    placeholder.markdown(full_answer)
            else:
                mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
                sys = f"{ANSWER_STYLE_SYSTEM}\n\n[대화 맥락]\n{mem_txt or '(없음)'}"
                msgs = [SystemMessage(content=sys), HumanMessage(content=user_input)]
                acc = ""
                for chunk in llm.stream(msgs):
                    piece = getattr(chunk, "content", "") or ""
                    if piece:
                        acc += piece
                        placeholder.markdown(remove_separators(acc) + "▌")
                full_answer = remove_separators(acc)
                placeholder.markdown(full_answer)

            follow = _append_followup_questions(llm, user_input, full_answer)
            if follow:
                full_answer += follow
                placeholder.markdown(remove_separators(full_answer))

        except Exception as exc:  # noqa: BLE001
            logger.warning("답변 생성 실패: %s", exc)
            full_answer = f"# 오류\n\n요청을 처리하는 중 문제가 발생했습니다.\n\n`{exc}`"
            placeholder.markdown(remove_separators(full_answer))

        st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
        st.session_state.conversation_memory.append({"role": "assistant", "content": full_answer})
        if len(st.session_state.conversation_memory) > 50:
            st.session_state.conversation_memory = st.session_state.conversation_memory[-50:]

        if sb and uid and st.session_state.current_session_id:
            try:
                update_session_messages(
                    sb,
                    str(uid),
                    st.session_state.current_session_id,
                    list(st.session_state.chat_history),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("auto-save session: %s", exc)


if __name__ == "__main__":
    main()
