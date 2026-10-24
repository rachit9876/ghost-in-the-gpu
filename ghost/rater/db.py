"""Ghost in the GPU — Multi-Domain SQLite Database."""
import sqlite3
import time
import json
from contextlib import contextmanager

from ghost.config import DB_PATH


# ═══════════════════════════════════════════════════
# CONNECTION
# ═══════════════════════════════════════════════════
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(
        DB_PATH,
        timeout=30.0,
        isolation_level="DEFERRED",  # explicit form of "" (avoids 3.12 deprecation warning)
    )
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=30000")
    c.execute("PRAGMA foreign_keys=ON")
    # synchronous=NORMAL is safe under WAL and faster than the default FULL.
    c.execute("PRAGMA synchronous=NORMAL")
    return c


@contextmanager
def connection():
    """Context manager that always closes the connection, even on exception."""
    c = connect()
    try:
        yield c
    finally:
        c.close()


# ═══════════════════════════════════════════════════
# SCHEMA + MIGRATIONS
# ═══════════════════════════════════════════════════
def init():
    with connection() as c:
        # 1. Tables
        c.executescript("""
        CREATE TABLE IF NOT EXISTS domains (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            slug TEXT NOT NULL UNIQUE,
            description TEXT DEFAULT '',
            created_at REAL NOT NULL,
            last_trained_at REAL,
            accuracy REAL
        );

        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain_id INTEGER NOT NULL DEFAULT 1,
            sha256 TEXT,
            filename TEXT NOT NULL,
            source_url TEXT,
            added_at REAL NOT NULL,
            batch_id TEXT,
            UNIQUE(domain_id, sha256),
            FOREIGN KEY(domain_id) REFERENCES domains(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain_id INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL,
            group_size INTEGER NOT NULL,
            winner_id INTEGER NOT NULL,
            losers_json TEXT NOT NULL,
            shown_order_json TEXT NOT NULL,
            sampler TEXT,
            session_id INTEGER,
            is_retest INTEGER NOT NULL DEFAULT 0,
            retest_of INTEGER,
            FOREIGN KEY(domain_id) REFERENCES domains(id) ON DELETE CASCADE,
            FOREIGN KEY(retest_of) REFERENCES comparisons(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain_id INTEGER DEFAULT 1,
            started_at REAL NOT NULL,
            ended_at REAL,
            comparison_count INTEGER DEFAULT 0
        );
        """)

        # 2. Migrate images from legacy global UNIQUE(sha256) form
        img_sql_row = c.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='images'"
        ).fetchone()
        if (img_sql_row
                and "sha256 TEXT UNIQUE" in img_sql_row[0]
                and "UNIQUE(domain_id, sha256)" not in img_sql_row[0]):
            _migrate_images_table(c)

        # 3. Migrate comparisons: add retest columns if missing
        cmp_cols = {r["name"] for r in c.execute(
            "PRAGMA table_info(comparisons)"
        ).fetchall()}
        if "is_retest" not in cmp_cols:
            c.execute(
                "ALTER TABLE comparisons ADD COLUMN is_retest "
                "INTEGER NOT NULL DEFAULT 0"
            )
        if "retest_of" not in cmp_cols:
            c.execute(
                "ALTER TABLE comparisons ADD COLUMN retest_of INTEGER "
                "REFERENCES comparisons(id) ON DELETE SET NULL"
            )

        # 4. Indexes (created after migrations so column list is stable)
        c.executescript("""
        CREATE INDEX IF NOT EXISTS idx_img_domain ON images(domain_id);
        CREATE INDEX IF NOT EXISTS idx_img_domain_batch ON images(domain_id, batch_id);
        CREATE INDEX IF NOT EXISTS idx_img_sha ON images(sha256);
        CREATE INDEX IF NOT EXISTS idx_cmp_domain ON comparisons(domain_id);
        CREATE INDEX IF NOT EXISTS idx_cmp_domain_retest ON comparisons(domain_id, is_retest);
        CREATE INDEX IF NOT EXISTS idx_cmp_winner ON comparisons(winner_id);
        CREATE INDEX IF NOT EXISTS idx_cmp_retest ON comparisons(is_retest);
        CREATE INDEX IF NOT EXISTS idx_cmp_retest_of ON comparisons(retest_of);
        """)

        # 5. Seed starter domains if empty (INSERT OR IGNORE survives concurrent starts)
        count = c.execute("SELECT COUNT(*) FROM domains").fetchone()[0]
        if count == 0:
            now = time.time()
            starters = [
                ("Wallpapers", "wallpapers",
                 "Desktop & ultrawide visual sceneries, sci-fi, minimalism"),
                ("Portraits & Art", "portraits",
                 "Character illustrations, portraits, expressive compositions"),
                ("Photography & Objects", "photography",
                 "Architecture, street photography, product design & textures"),
            ]
            for name, slug, desc in starters:
                c.execute(
                    "INSERT OR IGNORE INTO domains "
                    "(name, slug, description, created_at) VALUES (?, ?, ?, ?)",
                    (name, slug, desc, now),
                )

        c.commit()


def _migrate_images_table(c):
    """Rebuild the images table from the legacy global-UNIQUE(sha256) form.

    Copies only columns that exist in the old schema, so an upgrade from a very
    old version (missing source_url / batch_id) still succeeds.
    """
    old_cols = {r["name"] for r in c.execute(
        "PRAGMA table_info(images)"
    ).fetchall()}
    wanted = ["id", "domain_id", "sha256", "filename",
              "source_url", "added_at", "batch_id"]
    copy_cols = [w for w in wanted if w in old_cols]
    col_list = ", ".join(copy_cols)

    # PRAGMA must be set outside a transaction, so do it separately.
    c.execute("PRAGMA foreign_keys=OFF")
    try:
        c.executescript(f"""
        BEGIN TRANSACTION;
        CREATE TABLE images_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain_id INTEGER NOT NULL DEFAULT 1,
            sha256 TEXT,
            filename TEXT NOT NULL,
            source_url TEXT,
            added_at REAL NOT NULL,
            batch_id TEXT,
            UNIQUE(domain_id, sha256),
            FOREIGN KEY(domain_id) REFERENCES domains(id) ON DELETE CASCADE
        );
        INSERT INTO images_new ({col_list})
            SELECT {col_list} FROM images;
        DROP TABLE images;
        ALTER TABLE images_new RENAME TO images;
        COMMIT;
        """)
    except Exception:
        try:
            c.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        c.execute("PRAGMA foreign_keys=ON")


# ═══════════════════════════════════════════════════
# SESSIONS & COMPARISONS
# ═══════════════════════════════════════════════════
def start_session(domain_id: int = 1):
    with connection() as c:
        cur = c.execute(
            "INSERT INTO sessions(domain_id, started_at) VALUES (?, ?)",
            (domain_id, time.time()),
        )
        sid = cur.lastrowid
        c.commit()
        return sid


def log_comparison(
    domain_id: int,
    winner_id: int,
    losers: list,
    shown_order: list,
    session_id=None,
    sampler="active",
    is_retest: int = 0,
    retest_of: int = None,
):
    """Insert a comparison and bump the session counter atomically."""
    with connection() as c:
        now = time.time()
        c.execute(
            "INSERT INTO comparisons("
            "  domain_id, created_at, group_size, winner_id, "
            "  losers_json, shown_order_json, sampler, session_id, "
            "  is_retest, retest_of"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                domain_id, now, len(shown_order), winner_id,
                json.dumps(losers), json.dumps(shown_order), sampler,
                session_id, is_retest, retest_of,
            ),
        )
        if session_id:
            c.execute(
                "UPDATE sessions SET comparison_count = comparison_count + 1 "
                "WHERE id=?",
                (session_id,),
            )
        c.commit()