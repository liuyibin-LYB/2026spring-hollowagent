import io
import json
import mimetypes
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
PID_CACHE_DIR = DATA_DIR / "pid_post_cache"
COMMENT_CACHE_DIR = DATA_DIR / "comment_cache"
WEB_UI_DIR = DATA_DIR / "web_ui"
CHAT_HISTORY_FILE = WEB_UI_DIR / "chat_history.json"


class ChatError(Exception):
    def __init__(self, message, status=500, payload=None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.payload = payload or {}


def read_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def to_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def normalize_text(value):
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def deleted_title(pid):
    return f"#{pid} 已删除"


def with_deleted_tag(tags):
    return list(dict.fromkeys(["已删除"] + list(tags or [])))[:4]


def cached_comments_for_pid(pid):
    if pid <= 0:
        return []
    payload = read_json(COMMENT_CACHE_DIR / f"{pid}.json")
    if isinstance(payload, dict):
        comments = payload.get("comments")
    else:
        comments = payload
    if not isinstance(comments, list):
        return []
    return [normalize_comment(item) for item in comments if isinstance(item, dict)]


def has_cached_comments(pid):
    return bool(cached_comments_for_pid(pid))


def is_empty_deleted_placeholder(post):
    if not post or not post.get("deleted"):
        return False
    pid = to_int(post.get("pid"), 0)
    return (
        not normalize_text(post.get("text"))
        and not (post.get("comments") or [])
        and not to_int(post.get("timestamp"), 0)
        and not has_cached_comments(pid)
    )


def normalize_workflow(mode, message=""):
    mode = normalize_text(mode).lower()
    message = normalize_text(message).lower()
    if message.startswith("/daily"):
        return "daily"
    if message.startswith("/thorough"):
        return "thorough"
    if mode in {"quick", "deep", "daily", "thorough"}:
        return mode
    return "quick"


def title_from_text(text):
    for line in normalize_text(text).split("\n"):
        line = line.strip()
        if line:
            return line[:42] + ("..." if len(line) > 42 else "")
    return "无正文帖子"


def date_label(post):
    post_time = post.get("post_time") or post.get("created_at") or ""
    if isinstance(post_time, str) and len(post_time) >= 16:
        return post_time[5:16]
    ts = to_int(post.get("timestamp"), 0)
    if ts:
        return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
    return ""


def relative_label(post):
    ts = to_int(post.get("timestamp"), 0)
    if not ts:
        return "未知"
    delta = max(0, int(time.time() - ts))
    if delta < 3600:
        minutes = max(1, delta // 60)
        return f"{minutes}分钟前"
    if delta < 24 * 3600:
        return f"{delta // 3600}小时前"
    if delta < 48 * 3600:
        return "昨天"
    days = delta // (24 * 3600)
    if days < 365:
        return f"{days}天前"
    return datetime.fromtimestamp(ts).strftime("%Y年")


def normalize_comment(comment):
    name = (
        comment.get("name")
        or comment.get("name_tag")
        or ("洞主" if comment.get("islz") else "")
        or "洞友"
    )
    return {
        "cid": to_int(comment.get("cid") or comment.get("comment_id"), 0),
        "pid": to_int(comment.get("pid"), 0),
        "name": normalize_text(name),
        "text": normalize_text(comment.get("text")),
        "time": normalize_text(comment.get("reply_time"))[-5:] if comment.get("reply_time") else "",
        "reply_time": normalize_text(comment.get("reply_time")),
        "timestamp": to_int(comment.get("timestamp"), 0),
    }


def normalize_post(raw, source="cache"):
    if not isinstance(raw, dict):
        return None
    post = raw.get("post") if "post" in raw else raw
    if raw.get("found") is False:
        pid = to_int(raw.get("pid") or (post.get("pid") if isinstance(post, dict) else 0), 0)
        if pid <= 0:
            return None
        comments = cached_comments_for_pid(pid)
        return {
            "pid": pid,
            "title": deleted_title(pid),
            "text": "",
            "relative": "",
            "dateLabel": "",
            "post_time": "",
            "timestamp": 0,
            "reply": len(comments),
            "star": 0,
            "hasImage": False,
            "tags": with_deleted_tag([]),
            "comments": comments,
            "source": source,
            "deleted": True,
        }
    if not isinstance(post, dict):
        return None
    pid = to_int(post.get("pid"), 0)
    if pid <= 0:
        return None

    text = normalize_text(post.get("text"))
    comments = post.get("comments") or post.get("comment_list") or []
    comments = [normalize_comment(item) for item in comments if isinstance(item, dict)]
    post_time = normalize_text(post.get("post_time"))
    ts = to_int(post.get("timestamp"), 0)
    if not post_time and ts:
        post_time = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    deleted = bool(raw.get("deleted") or post.get("deleted"))
    tags = build_tags(post)
    if deleted:
        tags = with_deleted_tag(tags)

    return {
        "pid": pid,
        "title": deleted_title(pid) if deleted else title_from_text(text),
        "text": text,
        "relative": relative_label(post),
        "dateLabel": date_label(post),
        "post_time": post_time,
        "timestamp": ts,
        "reply": to_int(post.get("reply_count", post.get("reply", post.get("comment_total", len(comments))))),
        "star": to_int(post.get("star_count", post.get("likenum", post.get("praise_num", 0)))),
        "hasImage": bool(post.get("has_image") or post.get("media_ids") or post.get("type") == "image"),
        "tags": tags,
        "comments": comments,
        "source": source,
        "deleted": deleted,
    }


def build_tags(post):
    tags = []
    raw_tag = post.get("tag")
    if raw_tag:
        tags.append(str(raw_tag))
    if post.get("type") == "image" or post.get("media_ids"):
        tags.append("图片")
    if post.get("is_top"):
        tags.append("置顶")
    if not tags:
        tags.append("文本")
    return tags[:4]


class LocalIndex:
    def __init__(self):
        self.lock = threading.RLock()
        self.posts = {}
        self.sorted_posts = []
        self.last_built_at = None
        self.stats = {}
        self.last_llm_status = "agent.py 尚未调用"
        self.file_cache = {}

    @staticmethod
    def _file_signature(path):
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size)

    def _load_index_file(self, path, kind):
        source = f"pid_post_cache/{path.name}" if kind == "pid" else f"cache/{path.name}"
        cache_key = (kind, str(path))
        try:
            signature = self._file_signature(path)
        except OSError:
            return [], False

        with self.lock:
            cached = self.file_cache.get(cache_key)
            if cached and cached.get("signature") == signature:
                return [dict(post) for post in cached.get("posts", [])], True

        payload = read_json(path)
        posts = []
        if kind == "pid":
            post = normalize_post(payload, source=source)
            if post:
                posts.append(post)
        elif isinstance(payload, list):
            for item in payload:
                post = normalize_post(item, source=source)
                if post:
                    posts.append(post)
        elif isinstance(payload, dict):
            post = normalize_post(payload, source=source)
            if post:
                posts.append(post)

        with self.lock:
            self.file_cache[cache_key] = {
                "signature": signature,
                "posts": [dict(post) for post in posts],
            }
        return posts, False

    def _load_index_files(self, paths, kind):
        if not paths:
            return [], 0, 0
        paths = list(paths)
        max_workers = max(1, min(32, len(paths)))
        results = {}
        reused = 0
        parsed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self._load_index_file, path, kind): path
                for path in paths
            }
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    posts, was_reused = future.result()
                except Exception:
                    posts, was_reused = [], False
                results[path] = posts
                if was_reused:
                    reused += 1
                else:
                    parsed += 1
        ordered_posts = []
        for path in paths:
            ordered_posts.extend(results.get(path, []))
        return ordered_posts, reused, parsed

    def rebuild(self):
        started_at = time.time()
        posts = {}
        cache_files = sorted(CACHE_DIR.glob("*.json")) if CACHE_DIR.exists() else []
        pid_files = sorted(PID_CACHE_DIR.glob("*.json")) if PID_CACHE_DIR.exists() else []
        active_cache_keys = {("pid", str(path)) for path in pid_files}
        active_cache_keys.update(("cache", str(path)) for path in cache_files)

        with self.lock:
            for cache_key in list(self.file_cache):
                if cache_key not in active_cache_keys:
                    self.file_cache.pop(cache_key, None)

        pid_posts, pid_reused, pid_parsed = self._load_index_files(pid_files, "pid")
        cache_posts, cache_reused, cache_parsed = self._load_index_files(cache_files, "cache")

        for post in pid_posts:
            posts[post["pid"]] = post

        for post in cache_posts:
            posts[post["pid"]] = merge_posts(posts.get(post["pid"]), post)

        posts = {pid: post for pid, post in posts.items() if not is_empty_deleted_placeholder(post)}
        with self.lock:
            self.posts = posts
            self.sorted_posts = sorted(posts.values(), key=lambda item: item.get("pid", 0), reverse=True)
            self._refresh_stats_locked(cache_files=cache_files, pid_files=pid_files)
            self.stats["build_seconds"] = round(time.time() - started_at, 3)
            self.stats["index_files_reused"] = pid_reused + cache_reused
            self.stats["index_files_parsed"] = pid_parsed + cache_parsed

    def list_posts(
        self,
        query="",
        filter_name="all",
        sort="recent",
        offset=0,
        limit=40,
        min_star=0,
        min_reply=0,
    ):
        query = normalize_text(query).lower()
        min_star = max(0, to_int(min_star, 0))
        min_reply = max(0, to_int(min_reply, 0))
        with self.lock:
            items = list(self.sorted_posts)
        if query:
            tokens = [token for token in re.split(r"\s+", query) if token]
            items = [
                post
                for post in items
                if all(token in searchable_post_text(post).lower() for token in tokens)
            ]
        if filter_name == "starred":
            min_star = max(min_star, 10)
        elif filter_name == "active":
            min_reply = max(min_reply, 10)
        elif filter_name == "image":
            items = [post for post in items if post.get("hasImage")]
        if min_star:
            items = [post for post in items if post.get("star", 0) >= min_star]
        if min_reply:
            items = [post for post in items if post.get("reply", 0) >= min_reply]

        if sort == "star":
            items = sorted(items, key=lambda post: (post.get("star", 0), post.get("pid", 0)), reverse=True)
        elif sort == "reply":
            items = sorted(items, key=lambda post: (post.get("reply", 0), post.get("pid", 0)), reverse=True)

        total = len(items)
        page = items[offset : offset + limit]
        with self.lock:
            stats = dict(self.stats)
        return {
            "posts": [public_post(post, include_text=False, comment_limit=0) for post in page],
            "total": total,
            "offset": offset,
            "limit": limit,
            "stats": stats,
        }

    def get_post(self, pid):
        with self.lock:
            post = self.posts.get(to_int(pid))
            post = dict(post) if post else None
        if not post:
            return None
        full = dict(post)
        full["comments"] = self.load_comments(post["pid"], post.get("comments") or [])
        return public_post(full, include_text=True, comment_limit=200)

    def upsert_post(self, post):
        if not post or not post.get("pid"):
            return None
        pid = to_int(post.get("pid"))
        with self.lock:
            self.posts[pid] = post
            self.sorted_posts = sorted(self.posts.values(), key=lambda item: item.get("pid", 0), reverse=True)
            self._refresh_stats_locked()
        return self.get_post(pid)

    def mark_deleted(self, pid):
        pid = to_int(pid)
        with self.lock:
            existing = dict(self.posts.get(pid) or {})
        post = {
            "pid": pid,
            "title": deleted_title(pid),
            "text": existing.get("text", ""),
            "relative": existing.get("relative", ""),
            "dateLabel": existing.get("dateLabel", ""),
            "post_time": existing.get("post_time", ""),
            "timestamp": existing.get("timestamp", 0),
            "reply": existing.get("reply", 0),
            "star": existing.get("star", 0),
            "hasImage": existing.get("hasImage", False),
            "tags": with_deleted_tag(existing.get("tags") or []),
            "comments": self.load_comments(pid, existing.get("comments") or []),
            "source": existing.get("source") or f"pid_post_cache/{pid}.json",
            "deleted": True,
        }
        with self.lock:
            self.posts[pid] = post
            self.sorted_posts = sorted(self.posts.values(), key=lambda item: item.get("pid", 0), reverse=True)
            self._refresh_stats_locked()
        return self.get_post(pid)

    def refresh_post(self, pid):
        pid = to_int(pid)
        if pid <= 0:
            raise ChatError("PID 不合法。", status=400)
        try:
            with MAINTENANCE_BRIDGE.lock:
                agent = MAINTENANCE_BRIDGE.get_agent()
                agent._pid_fetch_rate_limiter.acquire()
                try:
                    result = agent.client.get_post(pid)
                except Exception as exc:
                    status_code = getattr(getattr(exc, "response", None), "status_code", None)
                    if status_code not in {404, 410}:
                        raise
                    result = {"success": False, "data": None, "status_code": status_code}

                if not result.get("success"):
                    if not looks_deleted_response(result, result.get("status_code")):
                        detail = response_error_text(result) or "树洞接口未返回成功状态"
                        raise ChatError(f"帖子刷新失败：{detail}", status=502)
                    MAINTENANCE_BRIDGE.status = f"帖子 #{pid} 已删除或不可访问"
                    return self.mark_deleted(pid)

                post = agent._normalize_post_metadata(result.get("data", {}))
                comments = []
                page = 1
                while True:
                    agent._comment_rate_limiter.acquire()
                    agent._record_comment_api_request()
                    comment_result = agent.client.get_comment(pid, page=page, limit=100, sort="asc")
                    if not comment_result.get("success"):
                        break
                    page_data = comment_result.get("data", {})
                    page_comments = [
                        agent._normalize_comment_metadata(comment)
                        for comment in page_data.get("data", [])
                        if isinstance(comment, dict)
                    ]
                    comments.extend(page_comments)
                    last_page = int(page_data.get("last_page") or page)
                    if page >= last_page or not page_comments:
                        break
                    page += 1

                post["comments"] = comments
                agent._all_comments_cache[pid] = comments
                agent._save_persistent_comments(pid, comments)
                agent._save_pid_post_cache(pid, post)

            normalized = normalize_post({"post": post}, source=f"pid_post_cache/{pid}.json")
            if not normalized:
                raise ChatError(f"帖子 #{pid} 刷新后数据无法解析。", status=502)
            normalized["deleted"] = False
            normalized["comments"] = self.load_comments(pid, normalized.get("comments") or [])
            return self.upsert_post(normalized)
        except ChatError:
            raise
        except Exception as exc:
            MAINTENANCE_BRIDGE.status = f"帖子刷新失败：{exc}"
            raise ChatError(f"帖子刷新失败：{exc}", status=502)

    def load_comments(self, pid, existing):
        comments = list(existing or [])
        path = COMMENT_CACHE_DIR / f"{pid}.json"
        payload = read_json(path)
        if isinstance(payload, dict) and isinstance(payload.get("comments"), list):
            comments = [normalize_comment(item) for item in payload["comments"] if isinstance(item, dict)]
        seen = set()
        unique = []
        for comment in comments:
            key = comment.get("cid") or (comment.get("name"), comment.get("text"), comment.get("time"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(comment)
        return unique

    def _refresh_stats_locked(self, cache_files=None, pid_files=None):
        if cache_files is None:
            cache_files = list(CACHE_DIR.glob("*.json")) if CACHE_DIR.exists() else []
        if pid_files is None:
            pid_files = list(PID_CACHE_DIR.glob("*.json")) if PID_CACHE_DIR.exists() else []
        self.last_built_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stats = {
            "posts": len(self.sorted_posts),
            "cache_files": len(cache_files),
            "pid_files": len(pid_files),
            "comment_files": len(list(COMMENT_CACHE_DIR.glob("*.json"))) if COMMENT_CACHE_DIR.exists() else 0,
            "built_at": self.last_built_at,
        }


def apply_post_controls(items, filter_name="all", sort="recent", min_star=0, min_reply=0):
    items = list(items)
    min_star = max(0, to_int(min_star, 0))
    min_reply = max(0, to_int(min_reply, 0))
    if filter_name == "starred":
        min_star = max(min_star, 10)
    elif filter_name == "active":
        min_reply = max(min_reply, 10)
    elif filter_name == "image":
        items = [post for post in items if post.get("hasImage")]
    if min_star:
        items = [post for post in items if post.get("star", 0) >= min_star]
    if min_reply:
        items = [post for post in items if post.get("reply", 0) >= min_reply]

    if sort == "star":
        items = sorted(items, key=lambda post: (post.get("star", 0), post.get("pid", 0)), reverse=True)
    elif sort == "reply":
        items = sorted(items, key=lambda post: (post.get("reply", 0), post.get("pid", 0)), reverse=True)
    return items

def merge_posts(old, new):
    if not old:
        if new.get("deleted"):
            new = dict(new)
            new["title"] = deleted_title(new.get("pid"))
            new["tags"] = with_deleted_tag(new.get("tags") or [])
        return new
    merged = dict(old)
    deleted = bool(merged.get("deleted") or new.get("deleted"))
    for key in ["text", "post_time", "dateLabel", "relative", "source"]:
        if len(str(new.get(key, ""))) > len(str(merged.get(key, ""))):
            merged[key] = new[key]
    if deleted:
        merged["title"] = deleted_title(merged.get("pid") or new.get("pid"))
    elif len(str(new.get("title", ""))) > len(str(merged.get("title", ""))):
        merged["title"] = new["title"]
    for key in ["timestamp", "reply", "star"]:
        merged[key] = max(to_int(merged.get(key)), to_int(new.get(key)))
    if new.get("comments") and len(new["comments"]) > len(merged.get("comments") or []):
        merged["comments"] = new["comments"]
    merged["hasImage"] = bool(merged.get("hasImage") or new.get("hasImage"))
    merged["tags"] = list(dict.fromkeys((merged.get("tags") or []) + (new.get("tags") or [])))[:4]
    if deleted:
        merged["tags"] = with_deleted_tag(merged.get("tags") or [])
    merged["deleted"] = deleted
    return merged


def searchable_post_text(post):
    comments = " ".join(comment.get("text", "") for comment in post.get("comments", [])[:8])
    return f"{post.get('pid')} {post.get('title')} {post.get('text')} {' '.join(post.get('tags', []))} {comments}"


def public_post(post, include_text=True, comment_limit=30):
    payload = {
        "pid": post["pid"],
        "title": post.get("title") or title_from_text(post.get("text", "")),
        "relative": post.get("relative") or relative_label(post),
        "dateLabel": post.get("dateLabel") or date_label(post),
        "post_time": post.get("post_time", ""),
        "timestamp": post.get("timestamp", 0),
        "reply": post.get("reply", 0),
        "star": post.get("star", 0),
        "hasImage": post.get("hasImage", False),
        "tags": post.get("tags", []),
        "source": post.get("source", ""),
        "deleted": bool(post.get("deleted")),
    }
    if include_text:
        payload["text"] = post.get("text", "")
    else:
        payload["text"] = (post.get("text", "")[:180] + "...") if len(post.get("text", "")) > 180 else post.get("text", "")
    comments = post.get("comments", []) or []
    payload["comments"] = comments[:comment_limit] if comment_limit else []
    payload["comment_total"] = max(post.get("reply", 0), len(comments))
    return payload


def get_model_name():
    try:
        import config_private as cfg
    except ImportError:
        import config as cfg
    return getattr(cfg, "LLM_MODEL", "未配置模型")


def response_error_text(payload):
    if not isinstance(payload, dict):
        return ""
    parts = []
    for key in ["message", "msg", "error", "code", "status"]:
        value = payload.get(key)
        if value is not None:
            parts.append(str(value))
    return " ".join(parts)


def looks_deleted_response(payload, status_code=None):
    if status_code in {404, 410}:
        return True
    text = response_error_text(payload).lower()
    return any(
        marker in text
        for marker in ["not found", "not exist", "不存在", "已删除", "删除", "找不到"]
    )


class AgentOutputWriter(io.TextIOBase):
    def __init__(self, emit):
        self.emit = emit
        self.line_buffer = ""
        self.last_status = ""
        self.last_status_at = 0.0
        self.lock = threading.Lock()

    def writable(self):
        return True

    def write(self, text):
        text = str(text)
        if not text:
            return 0

        with self.lock:
            chunks = re.split(r"(\r|\n)", text)
            for chunk in chunks:
                if chunk == "\n":
                    self._flush_line()
                elif chunk == "\r":
                    self._flush_status()
                else:
                    self.line_buffer += chunk
        return len(text)

    def flush(self):
        return None

    def flush_pending(self):
        with self.lock:
            self._flush_line()

    def _flush_line(self):
        line = self.line_buffer.strip()
        self.line_buffer = ""
        if line:
            self.emit({"role": "system", "kind": "line", "content": line})

    def _flush_status(self):
        line = self.line_buffer.strip()
        self.line_buffer = ""
        if not line:
            return
        now = time.time()
        if line == self.last_status and now - self.last_status_at < 0.45:
            return
        self.last_status = line
        self.last_status_at = now
        self.emit({"role": "system", "kind": "status", "key": "timer", "content": line})


class TranscriptStore:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        payload = read_json(path) or {}
        events = payload.get("events") if isinstance(payload, dict) else []
        if not isinstance(events, list):
            events = []
        self.events = [event for event in events if isinstance(event, dict)]
        self.next_id = max([to_int(event.get("id"), 0) for event in self.events] or [0]) + 1
        self.active_job_id = payload.get("active_job_id") if isinstance(payload, dict) else None
        self.last_save_at = 0.0

    def append(self, event, force_save=False):
        content = str(event.get("content", ""))
        if event.get("kind") != "delta":
            content = normalize_text(content)
        if not content and event.get("kind") not in {"done"}:
            return None

        with self.lock:
            payload = {
                "id": self.next_id,
                "ts": time.time(),
                "role": event.get("role") or "system",
                "kind": event.get("kind") or "line",
                "content": content,
            }
            for key in ["job_id", "key", "mode", "status"]:
                if event.get(key) is not None:
                    payload[key] = event[key]
            self.next_id += 1
            self.events.append(payload)
            if len(self.events) > 5000:
                self.events = self.events[-5000:]
            self._maybe_save_locked(force_save=force_save)
            return payload

    def set_active(self, job_id):
        with self.lock:
            self.active_job_id = job_id
            self._save_locked()

    def snapshot(self, after=0):
        after = to_int(after, 0)
        with self.lock:
            events = [event for event in self.events if to_int(event.get("id"), 0) > after]
            cursor = max([to_int(event.get("id"), 0) for event in self.events] or [0])
            return {
                "events": events,
                "cursor": cursor,
                "activeJob": self.active_job_id,
            }

    def _maybe_save_locked(self, force_save=False):
        now = time.time()
        if force_save or now - self.last_save_at >= 0.5:
            self._save_locked()

    def _save_locked(self):
        self.last_save_at = time.time()
        write_json(
            self.path,
            {
                "events": self.events,
                "active_job_id": self.active_job_id,
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )


class AgentBridge:
    def __init__(self):
        self.agent = None
        self.status = "agent.py 尚未启动"
        self.lock = threading.Lock()
        self.cli_bootstrapped = False

    def get_agent(self):
        if self.agent is None:
            from agent import TreeholeRAGAgent

            self.agent = TreeholeRAGAgent(interactive=False)
            self.status = "agent.py 已启动"
        return self.agent

    def current_mode(self):
        if self.agent is not None:
            return getattr(self.agent, "_default_cli_mode", "quick")
        return "quick"

    def bootstrap_cli(self, agent):
        if self.cli_bootstrapped:
            return
        from utils import print_header
        from agent import AGENT_PREFIX

        print_header("PKU Treehole RAG Agent")
        if not agent.load_conversation():
            agent._begin_new_session("启动会话", getattr(agent, "_default_cli_mode", "quick"))
        print(
            f"\n{AGENT_PREFIX}默认模式: {agent.MODE_PROFILES[agent._default_cli_mode]['label']}\n"
            f"{AGENT_PREFIX}输入 /help 查看命令，直接提问会走默认模式。\n"
        )
        self.cli_bootstrapped = True

    def run_cli_input(self, user_input, emit):
        command = normalize_text(user_input)
        if not command:
            self.status = "agent.py 未调用：缺少输入"
            raise ChatError("请输入内容后再发送。", status=400)

        assistant_chunks = []
        result = None
        try:
            with self.lock:
                writer = AgentOutputWriter(emit)
                with redirect_stdout(writer):
                    agent = None
                    previous_stream_callback = None
                    try:
                        agent = self.get_agent()
                        self.bootstrap_cli(agent)
                        previous_stream_callback = agent.stream_callback

                        def stream_callback(chunk):
                            text = str(chunk)
                            if not text:
                                return
                            assistant_chunks.append(text)
                            emit({"role": "assistant", "kind": "delta", "content": text})

                        agent.stream_callback = stream_callback
                        if command.startswith("/"):
                            try:
                                agent._handle_cli_command(command)
                            except KeyboardInterrupt:
                                pass
                            self.status = "agent.py 已处理 CLI 命令"
                            return {
                                "answer": "".join(assistant_chunks),
                                "sources": [],
                                "command": True,
                                "mode": normalize_workflow(self.current_mode(), command),
                            }

                        mode = getattr(agent, "_default_cli_mode", "quick")
                        if mode == "deep":
                            result = agent.mode_deep_research(command)
                        else:
                            result = agent.mode_quick_qa(command)
                    finally:
                        if agent is not None:
                            agent.stream_callback = previous_stream_callback
                        writer.flush_pending()

            answer = normalize_text(result.get("answer") if isinstance(result, dict) else "")
            if answer and not assistant_chunks:
                emit({"role": "assistant", "kind": "message", "content": answer})

            sources = []
            if isinstance(result, dict):
                for source in result.get("sources") or []:
                    pid = to_int(source.get("pid") if isinstance(source, dict) else source)
                    if pid and pid not in sources:
                        sources.append(pid)
            mode = result.get("mode") if isinstance(result, dict) else self.current_mode()
            self.status = f"agent.py 已返回 {mode} 回答"
            return {
                "answer": answer or "".join(assistant_chunks),
                "sources": sources,
                "mode": mode,
            }
        except ChatError:
            raise
        except Exception as exc:
            self.status = f"agent.py 调用失败：{exc}"
            raise ChatError(f"agent.py 调用失败：{exc}", status=502)

    def ask(self, query, context_posts=None, mode="quick"):
        events = []

        def emit(event):
            events.append(event)

        result = self.run_cli_input(query, emit)
        if not result.get("answer"):
            answer = "".join(event.get("content", "") for event in events if event.get("role") == "assistant")
            result["answer"] = answer
        result["events"] = events
        return result


class MaintenanceBridge:
    def __init__(self):
        self.agent = None
        self.status = "维护通道尚未启动"
        self.lock = threading.Lock()

    def get_agent(self):
        if self.agent is None:
            from agent import TreeholeRAGAgent

            self.agent = TreeholeRAGAgent(interactive=False)
            self.status = "维护通道已启动"
        return self.agent

class AgentJobManager:
    def __init__(self, transcript):
        self.transcript = transcript
        self.lock = threading.Lock()
        self.jobs = {}

    def create(self, message):
        message = normalize_text(message)
        if not message:
            raise ChatError("请输入内容后再发送。", status=400)

        job_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
        job = {
            "id": job_id,
            "status": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "message": message,
        }
        with self.lock:
            self.jobs[job_id] = job
        self.transcript.set_active(job_id)
        self.transcript.append(
            {"role": "user", "kind": "message", "job_id": job_id, "content": message},
            force_save=True,
        )
        thread = threading.Thread(target=self._run, args=(job_id, message), daemon=True)
        thread.start()
        return dict(job)

    def get(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def summary(self):
        with self.lock:
            return {job_id: dict(job) for job_id, job in self.jobs.items()}

    def _update(self, job_id, **updates):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            job.update(updates)
            job["updated_at"] = time.time()

    def _run(self, job_id, message):
        started_at = time.time()
        self._update(job_id, status="running", started_at=started_at)

        def emit(event):
            payload = dict(event)
            payload["job_id"] = job_id
            self.transcript.append(payload)

        try:
            result = AGENT_BRIDGE.run_cli_input(message, emit)
            elapsed = time.time() - started_at
            self._update(
                job_id,
                status="done",
                completed_at=time.time(),
                mode=result.get("mode"),
                sources=result.get("sources") or [],
                elapsed=elapsed,
            )
            self.transcript.append(
                {"role": "meta", "kind": "done", "job_id": job_id, "content": f"{elapsed:.2f}s"},
                force_save=True,
            )
        except ChatError as exc:
            self._update(job_id, status="error", error=exc.message, completed_at=time.time())
            self.transcript.append(
                {"role": "error", "kind": "message", "job_id": job_id, "content": exc.message},
                force_save=True,
            )
            self.transcript.append(
                {"role": "meta", "kind": "done", "job_id": job_id, "content": "error"},
                force_save=True,
            )
        except Exception as exc:
            message = f"agent.py 调用失败：{exc}"
            self._update(job_id, status="error", error=message, completed_at=time.time())
            self.transcript.append(
                {"role": "error", "kind": "message", "job_id": job_id, "content": message},
                force_save=True,
            )
            self.transcript.append(
                {"role": "meta", "kind": "done", "job_id": job_id, "content": "error"},
                force_save=True,
            )
        finally:
            if self.transcript.active_job_id == job_id:
                self.transcript.set_active(None)


AGENT_BRIDGE = AgentBridge()
MAINTENANCE_BRIDGE = MaintenanceBridge()
TRANSCRIPT = TranscriptStore(CHAT_HISTORY_FILE)
TRANSCRIPT.active_job_id = None
AGENT_JOBS = AgentJobManager(TRANSCRIPT)


INDEX = LocalIndex()
INDEX.rebuild()


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        parsed = urlparse(path)
        clean = unquote(parsed.path).lstrip("/")
        if not clean:
            clean = "index.html"
        return str((FRONTEND_DIR / clean).resolve())

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/posts":
            params = parse_qs(parsed.query)
            data = INDEX.list_posts(
                query=params.get("q", [""])[0],
                filter_name=params.get("filter", ["all"])[0],
                sort=params.get("sort", ["recent"])[0],
                offset=to_int(params.get("offset", ["0"])[0]),
                limit=min(120, max(1, to_int(params.get("limit", ["40"])[0], 40))),
                min_star=to_int(params.get("min_star", ["0"])[0]),
                min_reply=to_int(params.get("min_reply", ["0"])[0]),
            )
            return self.send_json(data)
        if parsed.path.startswith("/api/posts/") and not parsed.path.endswith("/refresh"):
            pid = parsed.path.rsplit("/", 1)[-1]
            post = INDEX.get_post(pid)
            if not post:
                return self.send_json({"error": "not found"}, status=404)
            return self.send_json(post)
        if parsed.path == "/api/stats":
            data = dict(INDEX.stats)
            data["model"] = get_model_name()
            data["agent_status"] = AGENT_BRIDGE.status
            data["maintenance_status"] = MAINTENANCE_BRIDGE.status
            return self.send_json(data)
        if parsed.path == "/api/agent/history":
            params = parse_qs(parsed.query)
            after = to_int(params.get("after", ["0"])[0], 0)
            data = TRANSCRIPT.snapshot(after=after)
            active_job = AGENT_JOBS.get(data.get("activeJob")) if data.get("activeJob") else None
            data["activeJobStatus"] = active_job
            data["model"] = get_model_name()
            data["agentStatus"] = AGENT_BRIDGE.status
            return self.send_json(data)
        if parsed.path.startswith("/api/agent/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            job = AGENT_JOBS.get(job_id)
            if not job:
                return self.send_json({"error": "not found"}, status=404)
            return self.send_json(job)
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/reindex":
            INDEX.rebuild()
            data = dict(INDEX.stats)
            data["model"] = get_model_name()
            return self.send_json({"ok": True, "stats": data})
        if parsed.path.startswith("/api/posts/") and parsed.path.endswith("/refresh"):
            pid = parsed.path.split("/")[-2]
            try:
                post = INDEX.refresh_post(pid)
                return self.send_json({"ok": True, "post": post, "stats": INDEX.stats})
            except ChatError as exc:
                return self.send_json({"error": exc.message}, status=exc.status)
        if parsed.path == "/api/agent/jobs":
            payload = self.read_json_body()
            try:
                job = AGENT_JOBS.create(payload.get("message", ""))
                return self.send_json({"ok": True, "job": job}, status=202)
            except ChatError as exc:
                return self.send_json({"error": exc.message}, status=exc.status)
        if parsed.path == "/api/chat":
            payload = self.read_json_body()
            try:
                data = AGENT_BRIDGE.ask(payload.get("message", ""))
                return self.send_json(data)
            except ChatError as exc:
                return self.send_json({"error": exc.message, **exc.payload}, status=exc.status)
        return self.send_json({"error": "not found"}, status=404)

    def read_json_body(self):
        length = to_int(self.headers.get("Content-Length"), 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def send_json(self, data, status=200):
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def guess_type(self, path):
        if path.endswith(".js"):
            return "application/javascript; charset=utf-8"
        if path.endswith(".css"):
            return "text/css; charset=utf-8"
        return mimetypes.guess_type(path)[0] or "application/octet-stream"


def main():
    port = to_int(os.environ.get("PORT"), 5177)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Serving offline Treehole UI on http://127.0.0.1:{port}")
    print(f"Indexed {INDEX.stats.get('posts', 0)} posts from local cache")
    server.serve_forever()


if __name__ == "__main__":
    main()
