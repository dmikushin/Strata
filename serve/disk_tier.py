"""A disk tier for conversations: the engine's session files, used automatically between requests.

WHY. The engine holds one live conversation. When requests from several conversations alternate - a main
session and its sub-agents - every request finds another conversation's state and reads its whole history
again. Measured on omen (RTX 3090, 256K context): 903 requests, 0.7 % of prompt tokens reused, prefill 69 %
of the engine's busy time, 74-81 s before the first token of a 120K+ request.

The engine's RAM parking (--conversation-cache-mib) is the in-memory answer, and on a 64 GB host at 256K it
does not fit: one long snapshot is 3-6 GB with ~10 GB free, and parking a 172,631-token conversation on top of
4.55 GB already parked got the engine OOM-killed. Session files go to disk instead. Measured for 176,375
tokens: a 2.93 GB file, SAVE 1.5 s (+0.35 GiB RAM, flat: it streams), RESTORE 2.0 s (+2.7 GiB RAM for the
transfer, released after), and the follow-up read 16 tokens instead of 176,371 - 0.11 s instead of 80.4 s.

WHAT. Called under the request FIFO, before the request's GEN:
  * LEAVING the live conversation (the new prompt shares less than half of its prompt) saves it, if it is
    long and has grown by `resave_growth` tokens since its last file. Saving only on growth bounds the
    writes: each save of a long conversation is ~3 GB of SSD wear, and an older file still saves nearly all
    of the prefill - the engine reads the difference again.
  * Then, if a saved conversation shares a longer prefix with the new prompt than the live one does (by
    `min_gain` tokens at least, and `min_tokens` in all), it is restored. This also brings a conversation
    back after the engine was restarted or unloaded for idleness, when nothing is live at all.

Matching is on exact token ids, the same rule the engine applies: each file keeps a sidecar with the ids it
holds. Only files named `tier-*` are this module's; anything else in the directory (a manual
/slots/0?action=save) is never read, replaced or deleted. Requests with images are left alone: their
identity is not the token ids alone.
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np

PREFIX = "tier-"


def common_prefix(a: np.ndarray, b: np.ndarray) -> int:
    """Length of the longest common start of two token arrays."""
    n = min(len(a), len(b))
    if n == 0:
        return 0
    diff = np.flatnonzero(a[:n] != b[:n])
    return int(diff[0]) if len(diff) else n


class Entry:
    def __init__(self, name: str, ids: np.ndarray, size: int, used: float):
        self.name, self.ids, self.size, self.used = name, ids, size, used


class DiskTier:
    def __init__(self, directory: str, refused: type, *, min_tokens: int = 16384, resave_growth: int = 16384,
                 min_gain: int = 4096, max_gib: float = 64.0, log=print):
        self.dir = directory
        self.refused = refused                  # the server's SessionRefused: a refusal leaves the engine in step
        self.min_tokens, self.resave_growth, self.min_gain = int(min_tokens), int(resave_growth), int(min_gain)
        self.max_bytes = int(float(max_gib) * (1 << 30))
        self.log = log
        self.entries: dict[str, Entry] = {}
        self._load()

    # -- the index ------------------------------------------------------------------------------------------
    def _paths(self, name: str) -> tuple[str, str]:
        return os.path.join(self.dir, name + ".bin"), os.path.join(self.dir, name + ".ids")

    def _load(self) -> None:
        """Rebuild the index from the directory: files survive a restart of the same engine version."""
        names = {f[:-4] for f in os.listdir(self.dir) if f.startswith(PREFIX) and f.endswith((".bin", ".ids"))}
        for name in sorted(names):
            bin_path, ids_path = self._paths(name)
            if not (os.path.exists(bin_path) and os.path.exists(ids_path)):
                self._remove(name, why="incomplete pair")      # a crash between the two writes
                continue
            ids = np.fromfile(ids_path, dtype=np.uint32)
            self.entries[name] = Entry(name, ids, os.path.getsize(bin_path), os.path.getmtime(bin_path))
        if self.entries:
            total = sum(e.size for e in self.entries.values())
            self.log(f"[strata] disk tier: {len(self.entries)} saved conversations, {total / (1 << 30):.1f} GiB "
                     f"in {self.dir}")

    def _remove(self, name: str, why: str) -> None:
        for path in self._paths(name):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        if self.entries.pop(name, None) is not None:
            self.log(f"[strata] disk tier: dropped {name} ({why})")

    def _evict(self) -> None:
        """Least recently used first, until the files fit max_gib."""
        while sum(e.size for e in self.entries.values()) > self.max_bytes and len(self.entries) > 1:
            oldest = min(self.entries.values(), key=lambda e: e.used)
            self._remove(oldest.name, why=f"over the {self.max_bytes / (1 << 30):.0f} GiB budget")

    # -- the decision ---------------------------------------------------------------------------------------
    @staticmethod
    def _live(engine) -> dict | None:
        """The engine's live conversation as generate() recorded it, if it belongs to the running process."""
        live = getattr(engine, "live", None)
        if not live or live.get("proc") is not getattr(engine, "proc", None):
            return None                         # never recorded, or recorded by a process that has since ended
        return live

    def before(self, engine, ids) -> None:
        """Under the FIFO, before GEN: save the conversation being left, restore the one being resumed."""
        prompt = np.asarray(ids, dtype=np.uint32)
        live = self._live(engine)
        live_ids = np.asarray(live["ids"], dtype=np.uint32) if live else np.zeros(0, dtype=np.uint32)
        live_shared = common_prefix(prompt, live_ids) if live else 0

        if live and live_shared < live["prompt_len"] // 2:
            self._save_if_grown(engine, live_ids)

        best, best_shared = None, 0
        for entry in self.entries.values():
            shared = common_prefix(prompt, entry.ids)
            if shared > best_shared:
                best, best_shared = entry, shared
        if best is not None and best_shared >= self.min_tokens and best_shared >= live_shared + self.min_gain:
            self._restore(engine, best, best_shared, len(prompt), live_shared)

    def _save_if_grown(self, engine, live_ids: np.ndarray) -> None:
        if len(live_ids) < self.min_tokens:
            return
        # The longest file this conversation already has: one whose every token is a prefix of it.
        covering = None
        for entry in self.entries.values():
            if len(entry.ids) <= len(live_ids) and common_prefix(entry.ids, live_ids) == len(entry.ids):
                if covering is None or len(entry.ids) > len(covering.ids):
                    covering = entry
        if covering is not None and len(live_ids) - len(covering.ids) < self.resave_growth:
            return                              # the file is close enough: the restore reads the rest again
        name = f"{PREFIX}{hashlib.sha1(live_ids[:self.min_tokens].tobytes()).hexdigest()[:16]}-{len(live_ids)}"
        bin_path, ids_path = self._paths(name)
        try:
            r = engine.session_file("save", bin_path)
        except self.refused as e:              # memory, storage, io: logged, the request goes on without it
            self.log(f"[strata] disk tier: save of {len(live_ids)} tokens refused ({getattr(e, 'kind', '?')}): {e}")
            return
        saved = live_ids[:int(r["tokens"])]     # the engine saves its deepest checkpoint, not every token
        tmp = ids_path + ".tmp"
        saved.tofile(tmp)
        os.replace(tmp, ids_path)
        self.entries[name] = Entry(name, saved, int(r["bytes"]), time.time())
        self.log(f"[strata] disk tier: saved {len(saved)} tokens to {name} ({int(r['bytes']) / (1 << 30):.2f} GiB, "
                 f"{float(r['ms']):.0f} ms)" + (f", replacing {covering.name}" if covering is not None else ""))
        if covering is not None:
            self._remove(covering.name, why=f"superseded by {name}")
        self._evict()

    def _restore(self, engine, entry: Entry, shared: int, prompt_len: int, live_shared: int) -> None:
        bin_path, _ = self._paths(entry.name)
        try:
            r = engine.session_file("restore", bin_path)
        except self.refused as e:
            kind = getattr(e, "kind", "?")
            self.log(f"[strata] disk tier: restore of {entry.name} refused ({kind}): {e}")
            if kind == "invalid":               # another engine version or configuration: it never will load
                self._remove(entry.name, why="the engine refused it as invalid")
            return
        entry.used = time.time()
        os.utime(bin_path)
        engine.live = {"proc": engine.proc, "prompt_len": int(r["tokens"]),
                       "ids": entry.ids[:int(r["tokens"])].tolist()}
        self.log(f"[strata] disk tier: restored {int(r['tokens'])} tokens from {entry.name} in {float(r['ms']):.0f} ms "
                 f"(prompt {prompt_len}, shares {shared} with it, {live_shared} with the live conversation)")
