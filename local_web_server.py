import io
import hashlib
import json
import mimetypes
import os
import pickle
import re
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
PID_CACHE_DIR = DATA_DIR / "pid_post_cache"
COMMENT_CACHE_DIR = DATA_DIR / "comment_cache"
DAILY_DIGEST_DIR = DATA_DIR / "daily_digest"
WEB_UI_DIR = DATA_DIR / "web_ui"
CHAT_HISTORY_FILE = WEB_UI_DIR / "chat_history.json"
POST_METADATA_FILE = WEB_UI_DIR / "post_metadata.json"
INDEX_DB_FILE = WEB_UI_DIR / "local_index.sqlite3"
INDEX_CACHE_VERSION = 2
SNAPSHOT_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL
SNAPSHOT_FILE_PREFIX = f"local_index_snapshot_v{INDEX_CACHE_VERSION}_"
SNAPSHOT_FILE_SUFFIX = ".pkl"
FAST_REINDEX_PID_LIMIT = 12000
PID_IMPORT_MAX_ITEMS = 500
PID_IMPORT_COMMENT_PREVIEW_LIMIT = 5
TITLE_GENERATION_BATCH_LIMIT = 24
FULLWIDTH_DIGIT_TRANSLATION = str.maketrans("０１２３４５６７８９", "0123456789")
PID_TEXT_PATTERN = re.compile(r"(?<!\d)\d{5,10}(?!\d)")


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


def extract_pids_from_text(value, limit=None):
    text = normalize_text(value).translate(FULLWIDTH_DIGIT_TRANSLATION)
    pids = []
    seen = set()
    for match in PID_TEXT_PATTERN.finditer(text):
        pid = to_int(match.group(), 0)
        if pid <= 0 or pid in seen:
            continue
        seen.add(pid)
        pids.append(pid)
        if limit is not None and len(pids) >= limit:
            break
    return pids


def now_label():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_date_key(value):
    value = normalize_text(value)
    if not value:
        return ""
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        return ""


def month_key(value=None):
    if value:
        value = normalize_text(value)
        try:
            return datetime.strptime(value[:7], "%Y-%m").strftime("%Y-%m")
        except Exception:
            pass
    return datetime.now().strftime("%Y-%m")


def post_date_key(post):
    post_time = normalize_text(post.get("post_time") or post.get("created_at"))
    if len(post_time) >= 10:
        parsed = parse_date_key(post_time[:10])
        if parsed:
            return parsed
    ts = to_int(post.get("timestamp"), 0)
    if ts:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    return ""


def resolve_date_range(start_date="", end_date="", preset=""):
    preset = normalize_text(preset).lower()
    today = date.today()
    if preset in {"today", "day"}:
        return today.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
    match = re.match(r"^(?:last_|days_)?(\d+)(?:d|days?)?$", preset)
    if match:
        days = max(1, to_int(match.group(1), 0))
        start = today - timedelta(days=days - 1)
        return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
    match = re.match(r"^(?:last_|months_)?(\d+)(?:m|months?)$", preset)
    if match:
        months = max(1, to_int(match.group(1), 0))
        start = today - timedelta(days=months * 31)
        return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
    return parse_date_key(start_date), parse_date_key(end_date)


def collection_slug(name):
    value = normalize_text(name).lower()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^0-9a-zA-Z_\-\u4e00-\u9fff]+", "", value)
    return value[:40] or uuid.uuid4().hex[:10]


def compact_line(value, limit=320):
    text = re.sub(r"\s+", " ", normalize_text(value))
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def clean_generated_title(value):
    text = normalize_text(value)
    if not text:
        return ""
    first_line = next((line.strip() for line in text.split("\n") if line.strip()), "")
    first_line = re.sub(r"^[-*#\d\.\s:：]+", "", first_line)
    first_line = first_line.strip("「」『』“”\"'` ，。；;：:")
    first_line = re.sub(r"\s+", "", first_line)
    return first_line[:30]


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
        return post_time[5:16].replace("-", "/")
    ts = to_int(post.get("timestamp"), 0)
    if ts:
        return datetime.fromtimestamp(ts).strftime("%m/%d %H:%M")
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
            "dateKey": "",
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
        "dateKey": post_time[:10] if len(post_time) >= 10 else (datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""),
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


class SQLiteIndexCache:
    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        self.available = True
        self._ensure_schema()

    def _connect(self):
        return sqlite3.connect(str(self.path))

    def _snapshot_file_path(self, manifest_hash):
        return self.path.parent / f"{SNAPSHOT_FILE_PREFIX}{manifest_hash}{SNAPSHOT_FILE_SUFFIX}"

    def _load_snapshot_file(self, manifest_hash):
        try:
            with self._snapshot_file_path(manifest_hash).open("rb") as f:
                posts = pickle.load(f)
            if not isinstance(posts, list):
                return None
            return [post for post in posts if isinstance(post, dict)]
        except Exception:
            return None

    def _write_snapshot_file(self, manifest_hash, posts_blob):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            path = self._snapshot_file_path(manifest_hash)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with tmp_path.open("wb") as f:
                f.write(posts_blob)
            os.replace(tmp_path, path)
            self._prune_snapshot_files(manifest_hash)
        except Exception:
            pass

    def _prune_snapshot_files(self, keep_manifest_hash):
        try:
            for path in self.path.parent.glob(f"{SNAPSHOT_FILE_PREFIX}*{SNAPSHOT_FILE_SUFFIX}"):
                if path.name != f"{SNAPSHOT_FILE_PREFIX}{keep_manifest_hash}{SNAPSHOT_FILE_SUFFIX}":
                    path.unlink(missing_ok=True)
        except Exception:
            pass

    def _ensure_schema(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock, self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS source_posts (
                        cache_version INTEGER NOT NULL,
                        kind TEXT NOT NULL,
                        path TEXT NOT NULL,
                        mtime_ns INTEGER NOT NULL,
                        size INTEGER NOT NULL,
                        posts_json TEXT NOT NULL,
                        updated_at REAL NOT NULL,
                        PRIMARY KEY (cache_version, kind, path)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS index_snapshots (
                        cache_version INTEGER NOT NULL,
                        manifest_hash TEXT NOT NULL,
                        posts_json TEXT NOT NULL,
                        posts_blob BLOB,
                        updated_at REAL NOT NULL,
                        PRIMARY KEY (cache_version, manifest_hash)
                    )
                    """
                )
                columns = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(index_snapshots)").fetchall()
                }
                if "posts_blob" not in columns:
                    conn.execute("ALTER TABLE index_snapshots ADD COLUMN posts_blob BLOB")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_source_posts_kind "
                    "ON source_posts(cache_version, kind)"
                )
        except Exception:
            self.available = False

    def load_records(self, kind):
        if not self.available:
            return {}
        try:
            with self.lock, self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT path, mtime_ns, size, posts_json
                    FROM source_posts
                    WHERE cache_version = ? AND kind = ?
                    """,
                    (INDEX_CACHE_VERSION, kind),
                ).fetchall()
            return {
                path: {
                    "mtime_ns": mtime_ns,
                    "size": size,
                    "posts_json": posts_json,
                }
                for path, mtime_ns, size, posts_json in rows
            }
        except Exception:
            return {}

    def upsert_records(self, records):
        if not self.available or not records:
            return
        now = time.time()
        rows = [
            (
                INDEX_CACHE_VERSION,
                record["kind"],
                record["path"],
                record["mtime_ns"],
                record["size"],
                record["posts_json"],
                now,
            )
            for record in records
        ]
        try:
            with self.lock, self._connect() as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO source_posts
                    (cache_version, kind, path, mtime_ns, size, posts_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        except Exception:
            self.available = False

    def load_snapshot(self, manifest_hash):
        if not manifest_hash:
            return None
        file_posts = self._load_snapshot_file(manifest_hash)
        if file_posts is not None:
            return file_posts
        if not self.available:
            return None
        try:
            with self.lock, self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT posts_json, posts_blob
                    FROM index_snapshots
                    WHERE cache_version = ? AND manifest_hash = ?
                    """,
                    (INDEX_CACHE_VERSION, manifest_hash),
                ).fetchone()
            if not row:
                return None
            posts_json, posts_blob = row
            posts = None
            loaded_from_blob = False
            if posts_blob:
                try:
                    posts = pickle.loads(posts_blob)
                    loaded_from_blob = True
                    self._write_snapshot_file(manifest_hash, posts_blob)
                except Exception:
                    posts = None
                    loaded_from_blob = False
            if posts is None:
                if not posts_json:
                    return None
                posts = json.loads(posts_json)
                self.backfill_snapshot_file(manifest_hash, posts)
            if not isinstance(posts, list):
                return None
            if loaded_from_blob:
                return [post for post in posts if isinstance(post, dict)]
            return [self.refresh_cached_post(post) for post in posts if isinstance(post, dict)]
        except Exception:
            return None

    def backfill_snapshot_file(self, manifest_hash, posts):
        if not self.available or not manifest_hash:
            return
        try:
            posts_blob = pickle.dumps(posts, protocol=SNAPSHOT_PICKLE_PROTOCOL)
            self._write_snapshot_file(manifest_hash, posts_blob)
        except Exception:
            pass

    def save_snapshot(self, manifest_hash, posts):
        if not manifest_hash:
            return
        try:
            posts_blob = pickle.dumps(posts, protocol=SNAPSHOT_PICKLE_PROTOCOL)
            self._write_snapshot_file(manifest_hash, posts_blob)
            if not self.available:
                return
            with self.lock, self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO index_snapshots
                    (cache_version, manifest_hash, posts_json, posts_blob, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (INDEX_CACHE_VERSION, manifest_hash, "", None, time.time()),
                )
                conn.execute(
                    """
                    DELETE FROM index_snapshots
                    WHERE cache_version = ? AND manifest_hash <> ?
                    """,
                    (INDEX_CACHE_VERSION, manifest_hash),
                )
        except Exception:
            self.available = False

    @staticmethod
    def refresh_cached_post(post):
        post = dict(post)
        post["pid"] = to_int(post.get("pid"), 0)
        post["timestamp"] = to_int(post.get("timestamp"), 0)
        post["reply"] = to_int(post.get("reply"), 0)
        post["star"] = to_int(post.get("star"), 0)
        post["relative"] = relative_label(post)
        post["dateLabel"] = date_label(post)
        post["dateKey"] = post.get("dateKey") or post_date_key(post)
        post["tags"] = list(post.get("tags") or [])
        post["comments"] = list(post.get("comments") or [])
        return post

    @staticmethod
    def decode_posts(posts_json):
        posts = json.loads(posts_json)
        if not isinstance(posts, list):
            return None
        return [SQLiteIndexCache.refresh_cached_post(post) for post in posts if isinstance(post, dict)]


class MetadataStore:
    DEFAULT_COLLECTION_ID = "default"

    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        self.payload = self._normalize_payload(read_json(path) or {})
        self._save()

    def _normalize_payload(self, payload):
        if not isinstance(payload, dict):
            payload = {}
        collections = payload.get("collections")
        if not isinstance(collections, list):
            collections = []
        normalized_collections = []
        seen = set()
        for item in collections:
            if not isinstance(item, dict):
                continue
            name = normalize_text(item.get("name"))
            cid = normalize_text(item.get("id")) or collection_slug(name)
            if not name or cid in seen:
                continue
            seen.add(cid)
            normalized_collections.append(
                {
                    "id": cid,
                    "name": name,
                    "created_at": item.get("created_at") or now_label(),
                }
            )
        if self.DEFAULT_COLLECTION_ID not in seen:
            normalized_collections.insert(
                0,
                {
                    "id": self.DEFAULT_COLLECTION_ID,
                    "name": "默认收藏",
                    "created_at": now_label(),
                },
            )

        posts = payload.get("posts")
        if not isinstance(posts, dict):
            posts = {}
        normalized_posts = {}
        valid_collection_ids = {item["id"] for item in normalized_collections}
        for raw_pid, meta in posts.items():
            pid = str(to_int(raw_pid, 0))
            if pid == "0" or not isinstance(meta, dict):
                continue
            collection_ids = [
                normalize_text(cid)
                for cid in meta.get("collections", [])
                if normalize_text(cid) in valid_collection_ids
            ]
            normalized_posts[pid] = {
                "custom_title": normalize_text(meta.get("custom_title"))[:80],
                "collections": list(dict.fromkeys(collection_ids)),
                "updated_at": meta.get("updated_at") or now_label(),
            }
        return {
            "version": 1,
            "collections": normalized_collections,
            "posts": normalized_posts,
            "updated_at": payload.get("updated_at") or now_label(),
        }

    def _save(self):
        self.payload["updated_at"] = now_label()
        write_json(self.path, self.payload)

    def _collections_by_id_locked(self):
        return {item["id"]: item for item in self.payload.get("collections", [])}

    def get_post_meta(self, pid):
        pid_key = str(to_int(pid, 0))
        with self.lock:
            meta = dict(self.payload.get("posts", {}).get(pid_key) or {})
            meta["collections"] = list(meta.get("collections") or [])
            return meta

    def public_meta(self, pid):
        meta = self.get_post_meta(pid)
        with self.lock:
            by_id = self._collections_by_id_locked()
        collection_ids = [cid for cid in meta.get("collections", []) if cid in by_id]
        return {
            "custom_title": meta.get("custom_title", ""),
            "collection_ids": collection_ids,
            "collection_names": [by_id[cid]["name"] for cid in collection_ids],
            "is_favorite": bool(collection_ids),
        }

    def search_text(self, pid):
        meta = self.public_meta(pid)
        return " ".join(
            [meta.get("custom_title", "")]
            + meta.get("collection_ids", [])
            + meta.get("collection_names", [])
        )

    def post_in_collection(self, pid, collection_id):
        collection_id = normalize_text(collection_id)
        if not collection_id:
            return True
        return collection_id in self.get_post_meta(pid).get("collections", [])

    def list_collections(self, posts=None):
        posts = list(posts or [])
        counts = self.collection_counts(posts)
        with self.lock:
            collections = [dict(item) for item in self.payload.get("collections", [])]
        for item in collections:
            item["count"] = counts.get(item["id"], 0)
        return {"collections": collections}

    def collection_counts(self, posts):
        visible_pids = {str(to_int(post.get("pid"), 0)) for post in posts or []}
        counts = {}
        with self.lock:
            for pid, meta in self.payload.get("posts", {}).items():
                if visible_pids and pid not in visible_pids:
                    continue
                for cid in meta.get("collections", []) or []:
                    counts[cid] = counts.get(cid, 0) + 1
        return counts

    def create_collection(self, name):
        name = normalize_text(name)[:32]
        if not name:
            raise ChatError("收藏栏名称不能为空。", status=400)
        with self.lock:
            base = collection_slug(name)
            cid = base
            existing = self._collections_by_id_locked()
            suffix = 2
            while cid in existing:
                cid = f"{base}-{suffix}"
                suffix += 1
            item = {"id": cid, "name": name, "created_at": now_label()}
            self.payload.setdefault("collections", []).append(item)
            self._save()
            return dict(item)

    def update_post(self, pid, custom_title=None, collection_ids=None, add_collection=None, remove_collection=None):
        pid = to_int(pid, 0)
        if pid <= 0:
            raise ChatError("PID 不合法。", status=400)
        with self.lock:
            by_id = self._collections_by_id_locked()
            posts = self.payload.setdefault("posts", {})
            pid_key = str(pid)
            meta = posts.setdefault(pid_key, {"custom_title": "", "collections": [], "updated_at": now_label()})
            if custom_title is not None:
                meta["custom_title"] = normalize_text(custom_title)[:80]
            if collection_ids is not None:
                next_ids = [
                    normalize_text(cid)
                    for cid in collection_ids
                    if normalize_text(cid) in by_id
                ]
                meta["collections"] = list(dict.fromkeys(next_ids))
            if add_collection is not None:
                cid = normalize_text(add_collection)
                if cid not in by_id:
                    raise ChatError("收藏栏不存在。", status=404)
                meta["collections"] = list(dict.fromkeys((meta.get("collections") or []) + [cid]))
            if remove_collection is not None:
                cid = normalize_text(remove_collection)
                meta["collections"] = [item for item in (meta.get("collections") or []) if item != cid]
            meta["updated_at"] = now_label()
            if not meta.get("custom_title") and not meta.get("collections"):
                posts.pop(pid_key, None)
            self._save()
        return self.public_meta(pid)


class LocalIndex:
    def __init__(self):
        self.lock = threading.RLock()
        self.posts = {}
        self.sorted_posts = []
        self.posts_by_date = {}
        self.month_stats = {}
        self.last_built_at = None
        self.stats = {}
        self.last_llm_status = "agent.py 尚未调用"
        self.file_cache = {}
        self.sqlite_cache = SQLiteIndexCache(INDEX_DB_FILE)
        self.cache_dir_signature = None
        self.comment_count_signature = None
        self.comment_count = 0

    @staticmethod
    def _file_signature(path):
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size)

    @staticmethod
    def _json_files_with_signatures(directory, kind):
        paths = []
        signatures = {}
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if not entry.name.endswith(".json") or not entry.is_file():
                            continue
                        stat = entry.stat()
                    except OSError:
                        continue
                    path = directory / entry.name
                    paths.append(path)
                    signatures[(kind, str(path))] = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            return [], {}
        paths.sort(key=lambda path: path.name)
        return paths, signatures

    @staticmethod
    def _json_file_count(directory):
        count = 0
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.name.endswith(".json") and entry.is_file():
                            count += 1
                    except OSError:
                        continue
        except OSError:
            return 0
        return count

    @staticmethod
    def _directory_signature(path):
        try:
            stat = path.stat()
            return (stat.st_mtime_ns, stat.st_size)
        except OSError:
            return (0, 0)

    def _cache_directory_signature(self):
        return (
            ("cache", self._directory_signature(CACHE_DIR)),
            ("pid", self._directory_signature(PID_CACHE_DIR)),
            ("comment", self._directory_signature(COMMENT_CACHE_DIR)),
        )

    def _comment_file_count(self):
        signature = self._directory_signature(COMMENT_CACHE_DIR)
        if self.comment_count_signature == signature:
            return self.comment_count
        count = self._json_file_count(COMMENT_CACHE_DIR)
        self.comment_count_signature = signature
        self.comment_count = count
        return count

    def _reuse_current_index_if_unchanged(self, started_at, cache_dir_signature):
        with self.lock:
            if not self.posts or self.cache_dir_signature != cache_dir_signature:
                return False
            self.last_built_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            stats = dict(self.stats)
            stats["posts"] = len(self.sorted_posts)
            stats["built_at"] = self.last_built_at
            stats["build_seconds"] = round(time.time() - started_at, 3)
            stats["index_cache"] = "memory-current"
            stats["index_files_reused"] = stats.get("cache_files", 0) + stats.get("pid_files", 0)
            stats["index_files_parsed"] = 0
            self.stats = stats
            return True

    @staticmethod
    def _manifest_hash(source_signatures):
        digest = hashlib.sha256()
        digest.update(f"v{INDEX_CACHE_VERSION}\n".encode("utf-8"))
        for (kind, path), signature in sorted(source_signatures.items()):
            digest.update(f"{kind}\0{path}\0{signature[0]}\0{signature[1]}\n".encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _source_record(kind, path, signature, posts):
        return {
            "kind": kind,
            "path": str(path),
            "mtime_ns": signature[0],
            "size": signature[1],
            "posts_json": json.dumps(posts, ensure_ascii=False, separators=(",", ":")),
        }

    def _load_index_file(self, path, kind, signature=None):
        source = f"pid_post_cache/{path.name}" if kind == "pid" else f"cache/{path.name}"
        cache_key = (kind, str(path))
        if signature is None:
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
            payload_posts = payload.get("posts")
            if isinstance(payload_posts, list):
                for item in payload_posts:
                    post = normalize_post(item, source=source)
                    if post:
                        posts.append(post)
            else:
                post = normalize_post(payload, source=source)
                if post:
                    posts.append(post)

        with self.lock:
            self.file_cache[cache_key] = {
                "signature": signature,
                "posts": [dict(post) for post in posts],
            }
        return posts, False

    def _load_index_files(self, paths, kind, source_signatures=None):
        if not paths:
            return [], 0, 0
        paths = list(paths)
        results = {}
        reused = 0
        parsed = 0
        to_parse = []
        source_records = []
        sqlite_records = self.sqlite_cache.load_records(kind)

        for path in paths:
            cache_key = (kind, str(path))
            signature = source_signatures.get(cache_key) if source_signatures else None
            if signature is None:
                try:
                    signature = self._file_signature(path)
                except OSError:
                    results[path] = []
                    continue

            with self.lock:
                cached = self.file_cache.get(cache_key)
                if cached and cached.get("signature") == signature:
                    results[path] = [dict(post) for post in cached.get("posts", [])]
                    reused += 1
                    continue

            sqlite_record = sqlite_records.get(str(path))
            if (
                sqlite_record
                and sqlite_record.get("mtime_ns") == signature[0]
                and sqlite_record.get("size") == signature[1]
            ):
                try:
                    cached_posts = SQLiteIndexCache.decode_posts(sqlite_record.get("posts_json", ""))
                except Exception:
                    cached_posts = None
                if cached_posts is not None:
                    results[path] = cached_posts
                    with self.lock:
                        self.file_cache[cache_key] = {
                            "signature": signature,
                            "posts": [dict(post) for post in cached_posts],
                        }
                    reused += 1
                    continue

            to_parse.append((path, signature))

        if to_parse:
            max_workers = max(1, min(get_request_max_parallel(), len(to_parse)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {
                    executor.submit(self._load_index_file, path, kind, signature): (path, signature)
                    for path, signature in to_parse
                }
                for future in as_completed(future_to_path):
                    path, signature = future_to_path[future]
                    try:
                        posts, was_reused = future.result()
                    except Exception:
                        posts, was_reused = [], False
                    results[path] = posts
                    if was_reused:
                        reused += 1
                    else:
                        parsed += 1
                        if signature is not None:
                            source_records.append(self._source_record(kind, path, signature, posts))
            self.sqlite_cache.upsert_records(source_records)

        ordered_posts = []
        for path in paths:
            ordered_posts.extend(results.get(path, []))
        return ordered_posts, reused, parsed

    @staticmethod
    def _signature_item(signature, name):
        for key, value in signature or ():
            if key == name:
                return value
        return None

    def _latest_daily_candidate_pids(self):
        if not DAILY_DIGEST_DIR.exists():
            return []
        try:
            run_dirs = [path for path in DAILY_DIGEST_DIR.iterdir() if path.is_dir()]
        except OSError:
            return []
        if not run_dirs:
            return []
        latest_dir = max(run_dirs, key=lambda path: path.stat().st_mtime)
        payload = read_json(latest_dir / "daily_candidates.json")
        if not isinstance(payload, list):
            return []
        pids = []
        for item in payload:
            if isinstance(item, dict):
                pid = to_int(item.get("pid"), 0)
                if pid > 0:
                    pids.append(pid)
        return pids

    def _recent_pid_candidates(self):
        pids = []
        latest_state = read_json(DATA_DIR / "latest_pid_state.json") or {}
        latest_pid = to_int(latest_state.get("latest_pid"), 0) if isinstance(latest_state, dict) else 0
        with self.lock:
            current_max_pid = max(self.posts.keys(), default=0)
        if latest_pid > current_max_pid:
            span = latest_pid - current_max_pid
            if span <= FAST_REINDEX_PID_LIMIT:
                pids.extend(range(current_max_pid + 1, latest_pid + 1))
            else:
                pids.extend(self._latest_daily_candidate_pids())
        elif latest_pid <= 0:
            pids.extend(self._latest_daily_candidate_pids())

        unique = []
        seen = set()
        for pid in pids:
            pid = to_int(pid, 0)
            if pid <= 0 or pid in seen:
                continue
            seen.add(pid)
            unique.append(pid)
            if len(unique) >= FAST_REINDEX_PID_LIMIT:
                break
        return unique

    def refresh_pid_cache_entries(self, pids, index_cache="pid-incremental"):
        started_at = time.time()
        unique_pids = []
        seen = set()
        for pid in pids or []:
            pid = to_int(pid, 0)
            if pid <= 0 or pid in seen:
                continue
            seen.add(pid)
            unique_pids.append(pid)
        if not unique_pids:
            return 0

        with self.lock:
            existing_by_pid = {pid: dict(self.posts.get(pid) or {}) for pid in unique_pids}

        posts_to_upsert = {}
        removed_pids = set()
        source_records = []
        parsed = 0

        for pid in unique_pids:
            path = PID_CACHE_DIR / f"{pid}.json"
            if not path.exists():
                removed_pids.add(pid)
                continue
            try:
                signature = self._file_signature(path)
            except OSError:
                removed_pids.add(pid)
                continue

            payload = read_json(path)
            normalized = normalize_post(payload, source=f"pid_post_cache/{path.name}")
            posts = [normalized] if normalized else []
            cache_key = ("pid", str(path))
            with self.lock:
                self.file_cache[cache_key] = {
                    "signature": signature,
                    "posts": [dict(post) for post in posts],
                }
            source_records.append(self._source_record("pid", path, signature, posts))
            parsed += 1

            if normalized and not is_empty_deleted_placeholder(normalized):
                posts_to_upsert[pid] = normalized
                continue

            existing = existing_by_pid.get(pid)
            if existing and normalized:
                posts_to_upsert[pid] = merge_posts(normalized, existing)
            else:
                removed_pids.add(pid)

        self.sqlite_cache.upsert_records(source_records)
        changed = len(posts_to_upsert) + len(removed_pids)
        if not changed:
            return 0

        with self.lock:
            for pid in removed_pids:
                if pid not in posts_to_upsert:
                    self.posts.pop(pid, None)
            for pid, post in posts_to_upsert.items():
                self.posts[pid] = post
            self.sorted_posts = sorted(self.posts.values(), key=lambda item: item.get("pid", 0), reverse=True)
            self._rebuild_secondary_indexes_locked()
            self.last_built_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            stats = dict(self.stats)
            stats["posts"] = len(self.sorted_posts)
            stats["built_at"] = self.last_built_at
            stats["build_seconds"] = round(time.time() - started_at, 3)
            stats["index_cache"] = index_cache
            stats["index_files_reused"] = 0
            stats["index_files_parsed"] = parsed
            stats["incremental_pids"] = len(unique_pids)
            self.stats = stats
            self.cache_dir_signature = self._cache_directory_signature()
        return changed

    def fast_reindex(self):
        current_signature = self._cache_directory_signature()
        with self.lock:
            previous_signature = self.cache_dir_signature
        if self._signature_item(previous_signature, "cache") != self._signature_item(current_signature, "cache"):
            self.rebuild(force=True)
            return

        changed = self.refresh_pid_cache_entries(self._recent_pid_candidates(), index_cache="pid-incremental")
        if changed:
            return
        self.rebuild()

    def rebuild(self, force=False):
        started_at = time.time()
        cache_dir_signature = self._cache_directory_signature()
        if not force and self._reuse_current_index_if_unchanged(started_at, cache_dir_signature):
            return

        posts = {}
        cache_files, cache_signatures = self._json_files_with_signatures(CACHE_DIR, "cache")
        pid_files, pid_signatures = self._json_files_with_signatures(PID_CACHE_DIR, "pid")
        source_signatures = {}
        source_signatures.update(pid_signatures)
        source_signatures.update(cache_signatures)
        manifest_hash = self._manifest_hash(source_signatures)
        snapshot_posts = self.sqlite_cache.load_snapshot(manifest_hash)
        active_cache_keys = {("pid", str(path)) for path in pid_files}
        active_cache_keys.update(("cache", str(path)) for path in cache_files)

        with self.lock:
            for cache_key in list(self.file_cache):
                if cache_key not in active_cache_keys:
                    self.file_cache.pop(cache_key, None)

        if snapshot_posts is not None:
            posts = {
                to_int(post.get("pid"), 0): post
                for post in snapshot_posts
                if to_int(post.get("pid"), 0) > 0 and not is_empty_deleted_placeholder(post)
            }
            with self.lock:
                self.posts = posts
                self.sorted_posts = sorted(posts.values(), key=lambda item: item.get("pid", 0), reverse=True)
                self._rebuild_secondary_indexes_locked()
                self._refresh_stats_locked(cache_files=cache_files, pid_files=pid_files)
                self.stats["build_seconds"] = round(time.time() - started_at, 3)
                self.stats["index_cache"] = "sqlite-snapshot"
                self.stats["index_files_reused"] = len(source_signatures)
                self.stats["index_files_parsed"] = 0
                self.cache_dir_signature = cache_dir_signature
            return

        pid_posts, pid_reused, pid_parsed = self._load_index_files(pid_files, "pid", source_signatures)
        cache_posts, cache_reused, cache_parsed = self._load_index_files(cache_files, "cache", source_signatures)

        for post in pid_posts:
            posts[post["pid"]] = post

        for post in cache_posts:
            posts[post["pid"]] = merge_posts(posts.get(post["pid"]), post)

        posts = {pid: post for pid, post in posts.items() if not is_empty_deleted_placeholder(post)}
        with self.lock:
            self.posts = posts
            self.sorted_posts = sorted(posts.values(), key=lambda item: item.get("pid", 0), reverse=True)
            self._rebuild_secondary_indexes_locked()
            self._refresh_stats_locked(cache_files=cache_files, pid_files=pid_files)
            self.stats["build_seconds"] = round(time.time() - started_at, 3)
            self.stats["index_cache"] = "sqlite-sources" if self.sqlite_cache.available else "memory"
            self.stats["index_files_reused"] = pid_reused + cache_reused
            self.stats["index_files_parsed"] = pid_parsed + cache_parsed
            self.cache_dir_signature = cache_dir_signature
            snapshot_payload = [dict(post) for post in self.sorted_posts]
        self.sqlite_cache.save_snapshot(manifest_hash, snapshot_payload)

    def _rebuild_secondary_indexes_locked(self):
        posts_by_date = {}
        month_stats = {}
        for post in self.sorted_posts:
            day_key = post.get("dateKey") or post_date_key(post)
            post["dateKey"] = day_key
            if not day_key:
                continue
            posts_by_date.setdefault(day_key, []).append(post)
            month = day_key[:7]
            day_stats = month_stats.setdefault(month, {}).setdefault(
                day_key,
                {
                    "date": day_key,
                    "count": 0,
                    "hours": [0] * 24,
                    "first_ts": 0,
                    "last_ts": 0,
                    "top_pids": [],
                },
            )
            ts = to_int(post.get("timestamp"), 0)
            hour = datetime.fromtimestamp(ts).hour if ts else -1
            day_stats["count"] += 1
            if 0 <= hour <= 23:
                day_stats["hours"][hour] += 1
            if ts:
                day_stats["first_ts"] = ts if not day_stats["first_ts"] else min(day_stats["first_ts"], ts)
                day_stats["last_ts"] = max(day_stats["last_ts"], ts)
            if len(day_stats["top_pids"]) < 4:
                day_stats["top_pids"].append(post.get("pid"))
        self.posts_by_date = posts_by_date
        self.month_stats = month_stats

    def _items_for_date_range_locked(self, start_key="", end_key=""):
        if not start_key and not end_key:
            return list(self.sorted_posts)
        day_keys = [
            key
            for key in self.posts_by_date.keys()
            if (not start_key or key >= start_key) and (not end_key or key <= end_key)
        ]
        day_keys.sort(reverse=True)
        items = []
        for key in day_keys:
            items.extend(self.posts_by_date.get(key, []))
        return items

    def list_posts(
        self,
        query="",
        filter_name="all",
        sort="recent",
        offset=0,
        limit=40,
        min_star=0,
        min_reply=0,
        start_date="",
        end_date="",
        date_preset="",
        collection_id="",
    ):
        query = normalize_text(query).lower()
        min_star = max(0, to_int(min_star, 0))
        min_reply = max(0, to_int(min_reply, 0))
        start_key, end_key = resolve_date_range(start_date, end_date, date_preset)
        collection_id = normalize_text(collection_id)
        with self.lock:
            items = self._items_for_date_range_locked(start_key, end_key)
        if query:
            tokens = [token for token in re.split(r"\s+", query) if token]
            items = [
                post
                for post in items
                if all(token in searchable_post_text(post).lower() for token in tokens)
            ]
        if collection_id:
            items = [post for post in items if POST_METADATA.post_in_collection(post.get("pid"), collection_id)]
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
            "dateRange": {"start": start_key, "end": end_key},
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
            self._rebuild_secondary_indexes_locked()
            self._refresh_stats_locked()
            self.cache_dir_signature = self._cache_directory_signature()
        return self.get_post(pid)

    def upsert_posts(self, posts):
        valid_posts = [post for post in posts if post and to_int(post.get("pid"), 0) > 0]
        if not valid_posts:
            return
        with self.lock:
            for post in valid_posts:
                self.posts[to_int(post.get("pid"))] = post
            self.sorted_posts = sorted(self.posts.values(), key=lambda item: item.get("pid", 0), reverse=True)
            self._rebuild_secondary_indexes_locked()
            self._refresh_stats_locked()
            self.cache_dir_signature = self._cache_directory_signature()

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
            self._rebuild_secondary_indexes_locked()
            self._refresh_stats_locked()
            self.cache_dir_signature = self._cache_directory_signature()
        return self.get_post(pid)

    def _refresh_post_with_agent(self, agent, pid):
        pid = to_int(pid)
        if pid <= 0:
            raise ChatError("PID 不合法。", status=400)
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
        expected_comments = post.get("comment_total", post.get("reply_count", post.get("reply", 0)))
        post["comments"] = self._fetch_comments_with_agent(agent, pid, max_comments=-1, expected_total=expected_comments)
        agent._save_pid_post_cache(pid, post)

        normalized = normalize_post({"post": post}, source=f"pid_post_cache/{pid}.json")
        if not normalized:
            raise ChatError(f"帖子 #{pid} 刷新后数据无法解析。", status=502)
        normalized["deleted"] = False
        normalized["comments"] = self.load_comments(pid, normalized.get("comments") or [])
        return self.upsert_post(normalized)

    def _fetch_comments_with_agent(self, agent, pid, max_comments=-1, expected_total=0, fail_on_error=False):
        pid = to_int(pid)
        if pid <= 0 or max_comments == 0:
            return []
        expected_total = max(0, to_int(expected_total, 0))
        cached = agent._load_persistent_comments(pid)
        if cached is not None:
            if max_comments == -1:
                if not expected_total or len(cached) >= expected_total:
                    return cached
            elif len(cached) >= max_comments or (expected_total and len(cached) >= expected_total):
                return cached[:max_comments]

        comments = []
        page = 1
        while True:
            agent._comment_rate_limiter.acquire()
            agent._record_comment_api_request()
            request_limit = 100 if max_comments == -1 else max(1, min(100, max_comments - len(comments)))
            comment_result = agent.client.get_comment(pid, page=page, limit=request_limit, sort="asc")
            if not comment_result.get("success"):
                if fail_on_error and (expected_total > 0 or max_comments > 0):
                    detail = response_error_text(comment_result) or "评论接口未返回成功状态"
                    raise ChatError(f"评论抓取失败：{detail}", status=502)
                break
            page_data = comment_result.get("data", {})
            page_comments = [
                agent._normalize_comment_metadata(comment)
                for comment in page_data.get("data", [])
                if isinstance(comment, dict)
            ]
            comments.extend(page_comments)
            if max_comments != -1 and len(comments) >= max_comments:
                comments = comments[:max_comments]
                break
            last_page = int(page_data.get("last_page") or page)
            if page >= last_page or not page_comments:
                break
            page += 1

        if fail_on_error and expected_total > 0 and not comments:
            raise ChatError("评论抓取失败：评论接口返回空列表", status=502)

        agent._all_comments_cache[pid] = comments
        agent._save_persistent_comments(pid, comments)
        return comments

    def refresh_post(self, pid):
        pid = to_int(pid)
        if pid <= 0:
            raise ChatError("PID 不合法。", status=400)
        try:
            with MAINTENANCE_BRIDGE.lock:
                agent = MAINTENANCE_BRIDGE.get_agent()
            post = self._refresh_post_with_agent(agent, pid)
            MAINTENANCE_BRIDGE.status = f"帖子 #{pid} 已刷新"
            return post
        except ChatError:
            raise
        except Exception as exc:
            MAINTENANCE_BRIDGE.status = f"帖子刷新失败：{exc}"
            raise ChatError(f"帖子刷新失败：{exc}", status=502)

    def _import_post_preview_with_agent(self, agent, pid):
        pid = to_int(pid)
        if pid <= 0:
            raise ChatError("PID 不合法。", status=400)
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
                raise ChatError(f"帖子导入失败：{detail}", status=502)
            agent._save_pid_post_cache(pid, None)
            MAINTENANCE_BRIDGE.status = f"帖子 #{pid} 已删除或不可访问"
            return self.mark_deleted(pid)

        post = agent._normalize_post_metadata(result.get("data", {}))
        post["comments"] = list(post.get("comments") or post.get("comment_list") or [])[:PID_IMPORT_COMMENT_PREVIEW_LIMIT]
        agent._save_pid_post_cache(pid, post)

        normalized = normalize_post({"post": post}, source=f"pid_post_cache/{pid}.json")
        if not normalized:
            raise ChatError(f"帖子 #{pid} 导入后数据无法解析。", status=502)
        normalized["deleted"] = False
        normalized["comments"] = self.load_comments(pid, normalized.get("comments") or [])
        return normalized

    def import_pids_from_text(self, text):
        pids = extract_pids_from_text(text, limit=PID_IMPORT_MAX_ITEMS + 1)
        if not pids:
            raise ChatError("没有提取到有效 PID。", status=400)
        if len(pids) > PID_IMPORT_MAX_ITEMS:
            raise ChatError(f"一次最多导入 {PID_IMPORT_MAX_ITEMS} 个 PID，请分批处理。", status=400)

        with MAINTENANCE_BRIDGE.lock:
            agent = MAINTENANCE_BRIDGE.get_agent()

        started_at = time.time()
        max_workers = min(get_pid_import_max_parallel(), len(pids))
        MAINTENANCE_BRIDGE.status = f"批量导入 {len(pids)} 个 PID 中"
        results_by_pid = {}
        fetched_posts = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pid = {
                executor.submit(self._import_post_preview_with_agent, agent, pid): pid
                for pid in pids
            }
            for future in as_completed(future_to_pid):
                pid = future_to_pid[future]
                try:
                    post = future.result()
                    if not post.get("deleted"):
                        fetched_posts.append(post)
                    comment_count = len(post.get("comments") or [])
                    expected_comments = to_int(post.get("reply", post.get("comment_total", 0)), 0)
                    results_by_pid[pid] = {
                        "pid": pid,
                        "status": "deleted" if post.get("deleted") else "fetched",
                        "title": post.get("title") or deleted_title(pid),
                        "reply": post.get("reply", 0),
                        "star": post.get("star", 0),
                        "comments": comment_count,
                        "comment_total": max(expected_comments, comment_count),
                        "comment_status": "ok" if comment_count or expected_comments == 0 else "empty",
                    }
                except ChatError as exc:
                    results_by_pid[pid] = {"pid": pid, "status": "failed", "error": exc.message}
                except Exception as exc:
                    results_by_pid[pid] = {"pid": pid, "status": "failed", "error": str(exc)}

        self.upsert_posts(fetched_posts)
        results = [results_by_pid[pid] for pid in pids]
        fetched = sum(1 for item in results if item["status"] == "fetched")
        deleted = sum(1 for item in results if item["status"] == "deleted")
        failed = sum(1 for item in results if item["status"] == "failed")
        elapsed = time.time() - started_at
        MAINTENANCE_BRIDGE.status = f"批量导入完成：成功 {fetched}，已删除 {deleted}，失败 {failed}"
        return {
            "ok": True,
            "pids": pids,
            "results": results,
            "summary": {
                "requested": len(pids),
                "fetched": fetched,
                "deleted": deleted,
                "failed": failed,
                "max_workers": max_workers,
                "comment_preview_limit": PID_IMPORT_COMMENT_PREVIEW_LIMIT,
                "elapsed": elapsed,
            },
            "stats": self.stats,
        }

    @staticmethod
    def _infer_day_coverage(day_key, hours, count, first_ts, last_ts):
        if count <= 0:
            return {"status": "empty", "label": "无缓存", "complete": False}
        active_hours = [idx for idx, value in enumerate(hours) if value]
        if not active_hours:
            return {"status": "partial", "label": "时间未知", "complete": False}

        target = datetime.strptime(day_key, "%Y-%m-%d").date()
        today = date.today()
        min_hour = min(active_hours)
        max_hour = max(active_hours)
        span = max_hour - min_hour + 1
        covered = len(active_hours)

        if target > today:
            return {"status": "future", "label": "未来日期", "complete": False}
        if target == today:
            elapsed_hours = datetime.now().hour + 1
            expected_count = max(80, int(1000 * elapsed_hours / 24 * 0.7))
            enough_hours = covered >= max(3, int(elapsed_hours * 0.55))
            current = max_hour >= max(0, datetime.now().hour - 2)
            if count >= expected_count and enough_hours and current:
                return {"status": "likely_complete", "label": "今日基本连续", "complete": True}
            return {"status": "partial", "label": f"今日不足 {expected_count} 帖", "complete": False}

        if count >= 1000 and min_hour <= 2 and max_hour >= 22 and covered >= 18:
            return {"status": "likely_complete", "label": "推测完整", "complete": True}
        if count >= 1000:
            return {"status": "broad", "label": "数量达标，时间不连续", "complete": False}
        if span >= 16 and covered >= 10:
            return {"status": "partial", "label": "覆盖较广但少于 1000 帖", "complete": False}
        return {"status": "partial", "label": "少于 1000 帖", "complete": False}

    def calendar_month(self, month=""):
        key = month_key(month)
        first_day = datetime.strptime(f"{key}-01", "%Y-%m-%d").date()
        if first_day.month == 12:
            next_month = date(first_day.year + 1, 1, 1)
        else:
            next_month = date(first_day.year, first_day.month + 1, 1)
        days = {}
        current = first_day
        while current < next_month:
            day_key = current.strftime("%Y-%m-%d")
            days[day_key] = {
                "date": day_key,
                "count": 0,
                "hours": [0] * 24,
                "first_ts": 0,
                "last_ts": 0,
                "top_pids": [],
            }
            current += timedelta(days=1)

        with self.lock:
            cached_month = self.month_stats.get(key, {})
            for day_key, item in cached_month.items():
                if day_key in days:
                    days[day_key] = {
                        "date": day_key,
                        "count": item.get("count", 0),
                        "hours": list(item.get("hours") or [0] * 24),
                        "first_ts": item.get("first_ts", 0),
                        "last_ts": item.get("last_ts", 0),
                        "top_pids": list(item.get("top_pids") or []),
                    }

        day_list = []
        max_count = max((item["count"] for item in days.values()), default=0)
        for day_key, item in days.items():
            coverage = self._infer_day_coverage(
                day_key,
                item["hours"],
                item["count"],
                item["first_ts"],
                item["last_ts"],
            )
            item["coverage"] = coverage["status"]
            item["coverageLabel"] = coverage["label"]
            item["complete"] = coverage["complete"]
            item["intensity"] = 0 if max_count <= 0 else round(item["count"] / max_count, 3)
            item["firstTime"] = datetime.fromtimestamp(item["first_ts"]).strftime("%H:%M") if item["first_ts"] else ""
            item["lastTime"] = datetime.fromtimestamp(item["last_ts"]).strftime("%H:%M") if item["last_ts"] else ""
            day_list.append(item)

        return {
            "month": key,
            "today": date.today().strftime("%Y-%m-%d"),
            "days": day_list,
            "summary": {
                "posts": sum(item["count"] for item in day_list),
                "max_count": max_count,
                "likely_complete_days": sum(1 for item in day_list if item["complete"]),
            },
        }

    def generate_missing_titles(self, collection_id="", limit=TITLE_GENERATION_BATCH_LIMIT):
        collection_id = normalize_text(collection_id)
        limit = max(1, min(TITLE_GENERATION_BATCH_LIMIT, to_int(limit, TITLE_GENERATION_BATCH_LIMIT)))
        with self.lock:
            items = list(self.sorted_posts)
        if collection_id:
            items = [post for post in items if POST_METADATA.post_in_collection(post.get("pid"), collection_id)]
        targets = [
            post
            for post in items
            if not POST_METADATA.get_post_meta(post.get("pid")).get("custom_title")
        ][:limit]
        if not targets:
            return {"ok": True, "results": [], "summary": {"requested": 0, "updated": 0, "failed": 0}}

        MAINTENANCE_BRIDGE.status = f"正在为 {len(targets)} 条收藏生成标题"
        results = []
        updated = 0
        failed = 0
        with MAINTENANCE_BRIDGE.lock:
            agent = MAINTENANCE_BRIDGE.get_agent()
            for post in targets:
                pid = post.get("pid")
                full = self.get_post(pid) or public_post(post)
                comments = full.get("comments") or []
                comment_preview = "\n".join(
                    f"- {compact_line(comment.get('text', ''), 120)}"
                    for comment in comments[:5]
                    if comment.get("text")
                )
                prompt = (
                    "请给下面这条树洞帖子生成一个我个人收藏夹里使用的短标题。\n"
                    "要求：只输出标题本身，不要解释；不超过 18 个中文字符；不要包含 PID、引号或句号。\n\n"
                    f"PID: {pid}\n"
                    f"正文: {compact_line(full.get('text', ''), 900)}\n"
                    f"评论预览:\n{comment_preview or '无'}"
                )
                try:
                    raw_title = agent.call_llm(
                        user_message=prompt,
                        system_message="你是一个只输出短标题的中文信息整理助手。",
                        stream=False,
                    )
                    title = clean_generated_title(raw_title)
                    if not title or title.startswith("抱歉"):
                        raise ChatError(raw_title or "LLM 没有返回有效标题", status=502)
                    meta = POST_METADATA.update_post(pid, custom_title=title)
                    updated += 1
                    results.append({"pid": pid, "status": "updated", "title": title, **meta})
                except Exception as exc:
                    failed += 1
                    results.append({"pid": pid, "status": "failed", "error": str(exc)})

        MAINTENANCE_BRIDGE.status = f"标题生成完成：更新 {updated}，失败 {failed}"
        return {
            "ok": True,
            "results": results,
            "summary": {
                "requested": len(targets),
                "updated": updated,
                "failed": failed,
                "limit": limit,
            },
            "collections": POST_METADATA.list_collections(self.sorted_posts)["collections"],
        }

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
            cache_files, _cache_signatures = self._json_files_with_signatures(CACHE_DIR, "cache")
        if pid_files is None:
            pid_files, _pid_signatures = self._json_files_with_signatures(PID_CACHE_DIR, "pid")
        self.last_built_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stats = {
            "posts": len(self.sorted_posts),
            "cache_files": len(cache_files),
            "pid_files": len(pid_files),
            "comment_files": self._comment_file_count(),
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
    for key in ["text", "post_time", "dateLabel", "dateKey", "relative", "source"]:
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
    metadata_text = POST_METADATA.search_text(post.get("pid")) if "POST_METADATA" in globals() else ""
    return f"{post.get('pid')} {post.get('title')} {post.get('text')} {' '.join(post.get('tags', []))} {metadata_text} {comments}"


def public_post(post, include_text=True, comment_limit=30):
    meta = POST_METADATA.public_meta(post["pid"]) if "POST_METADATA" in globals() else {}
    original_title = post.get("title") or title_from_text(post.get("text", ""))
    custom_title = meta.get("custom_title", "")
    collection_names = meta.get("collection_names", [])
    tags = list(dict.fromkeys((post.get("tags", []) or []) + collection_names))[:8]
    payload = {
        "pid": post["pid"],
        "title": custom_title or original_title,
        "original_title": original_title,
        "custom_title": custom_title,
        "relative": relative_label(post),
        "dateLabel": post.get("dateLabel") or date_label(post),
        "dateKey": post.get("dateKey") or post_date_key(post),
        "post_time": post.get("post_time", ""),
        "timestamp": post.get("timestamp", 0),
        "reply": post.get("reply", 0),
        "star": post.get("star", 0),
        "hasImage": post.get("hasImage", False),
        "tags": tags,
        "base_tags": post.get("tags", []),
        "collection_ids": meta.get("collection_ids", []),
        "collection_names": collection_names,
        "is_favorite": bool(meta.get("is_favorite")),
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


def get_request_max_parallel():
    try:
        import config_private as cfg
    except ImportError:
        import config as cfg
    return max(1, to_int(getattr(cfg, "REQUEST_MAX_PARALLEL", 20), 20))


def get_pid_import_max_parallel():
    return get_request_max_parallel()


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
    if isinstance(payload, dict) and payload.get("code") in {40001, 41001}:
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
        touched_cache_pids = set()

        def refresh_touched_cache_pids():
            if not touched_cache_pids:
                return
            pids = list(touched_cache_pids)
            touched_cache_pids.clear()
            try:
                INDEX.refresh_pid_cache_entries(pids, index_cache="agent-pid-incremental")
            except Exception as exc:
                self.status = f"agent.py completed; incremental index refresh failed: {exc}"

        try:
            with self.lock:
                writer = AgentOutputWriter(emit)
                with redirect_stdout(writer):
                    agent = None
                    previous_stream_callback = None
                    previous_save_pid_post_cache = None
                    try:
                        agent = self.get_agent()
                        self.bootstrap_cli(agent)
                        previous_stream_callback = agent.stream_callback
                        previous_save_pid_post_cache = agent._save_pid_post_cache

                        def save_pid_post_cache(pid, post):
                            previous_save_pid_post_cache(pid, post)
                            pid_value = to_int(pid, 0)
                            if pid_value > 0:
                                touched_cache_pids.add(pid_value)

                        def stream_callback(chunk):
                            text = str(chunk)
                            if not text:
                                return
                            assistant_chunks.append(text)
                            emit({"role": "assistant", "kind": "delta", "content": text})

                        agent._save_pid_post_cache = save_pid_post_cache
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
                            if previous_save_pid_post_cache is not None:
                                agent._save_pid_post_cache = previous_save_pid_post_cache
                        writer.flush_pending()
                        refresh_touched_cache_pids()

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
POST_METADATA = MetadataStore(POST_METADATA_FILE)


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
                start_date=params.get("start_date", [""])[0],
                end_date=params.get("end_date", [""])[0],
                date_preset=params.get("date_preset", [""])[0],
                collection_id=params.get("collection", [""])[0],
            )
            return self.send_json(data)
        if parsed.path == "/api/calendar":
            params = parse_qs(parsed.query)
            return self.send_json(INDEX.calendar_month(params.get("month", [""])[0]))
        if parsed.path == "/api/collections":
            with INDEX.lock:
                posts = list(INDEX.sorted_posts)
            return self.send_json(POST_METADATA.list_collections(posts))
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
            params = parse_qs(parsed.query)
            full_rebuild = normalize_text(params.get("full", [""])[0]).lower() in {"1", "true", "yes", "full"}
            if full_rebuild:
                INDEX.rebuild(force=True)
            else:
                INDEX.fast_reindex()
            data = dict(INDEX.stats)
            data["model"] = get_model_name()
            return self.send_json({"ok": True, "stats": data})
        if parsed.path == "/api/pids/import":
            payload = self.read_json_body()
            try:
                data = INDEX.import_pids_from_text(payload.get("text", ""))
                return self.send_json(data)
            except ChatError as exc:
                return self.send_json({"error": exc.message}, status=exc.status)
        if parsed.path == "/api/collections":
            payload = self.read_json_body()
            try:
                item = POST_METADATA.create_collection(payload.get("name", ""))
                with INDEX.lock:
                    posts = list(INDEX.sorted_posts)
                return self.send_json({"ok": True, "collection": item, **POST_METADATA.list_collections(posts)}, status=201)
            except ChatError as exc:
                return self.send_json({"error": exc.message}, status=exc.status)
        if parsed.path.startswith("/api/posts/") and parsed.path.endswith("/metadata"):
            pid = parsed.path.split("/")[-2]
            payload = self.read_json_body()
            try:
                toggle_collection = normalize_text(payload.get("toggle_collection"))
                if toggle_collection:
                    current = POST_METADATA.get_post_meta(pid).get("collections", [])
                    if toggle_collection in current:
                        meta = POST_METADATA.update_post(pid, remove_collection=toggle_collection)
                    else:
                        meta = POST_METADATA.update_post(pid, add_collection=toggle_collection)
                else:
                    meta = POST_METADATA.update_post(
                        pid,
                        custom_title=payload.get("custom_title") if "custom_title" in payload else None,
                        collection_ids=payload.get("collection_ids") if "collection_ids" in payload else None,
                        add_collection=payload.get("add_collection") if "add_collection" in payload else None,
                        remove_collection=payload.get("remove_collection") if "remove_collection" in payload else None,
                    )
                with INDEX.lock:
                    posts = list(INDEX.sorted_posts)
                return self.send_json(
                    {
                        "ok": True,
                        "meta": meta,
                        "post": INDEX.get_post(pid),
                        "collections": POST_METADATA.list_collections(posts)["collections"],
                    }
                )
            except ChatError as exc:
                return self.send_json({"error": exc.message}, status=exc.status)
        if parsed.path == "/api/titles/generate":
            payload = self.read_json_body()
            try:
                data = INDEX.generate_missing_titles(
                    collection_id=payload.get("collection_id", ""),
                    limit=payload.get("limit", TITLE_GENERATION_BATCH_LIMIT),
                )
                return self.send_json(data)
            except ChatError as exc:
                return self.send_json({"error": exc.message}, status=exc.status)
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
