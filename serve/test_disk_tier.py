"""Tests for serve/disk_tier.py with a scripted engine: the decisions, not the engine's file format.

  python3 serve/test_disk_tier.py
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serve.disk_tier import DiskTier  # noqa: E402


class Refused(Exception):
    def __init__(self, kind, message="refused"):
        super().__init__(message)
        self.kind = kind


class FakeEngine:
    """Holds a live conversation like StrataEngine.generate records it; SAVE writes the live ids' count."""

    def __init__(self):
        self.proc = object()
        self.live = None
        self.calls = []
        self.refuse = {}                        # action -> kind

    def run(self, ids, generated=()):
        self.live = {"proc": self.proc, "prompt_len": len(ids), "ids": list(ids) + list(generated)}

    def restart(self):
        self.proc = object()                    # the recorded live state belongs to the old process now

    def session_file(self, action, path):
        self.calls.append((action, os.path.basename(path)))
        if action in self.refuse:
            raise Refused(self.refuse[action])
        if action == "save":
            n = len(self.live["ids"])
            with open(path, "wb") as f:
                f.write(b"x" * n)               # one byte per token stands in for the snapshot
            return {"tokens": n, "bytes": n, "ms": 1.0}
        n = os.path.getsize(path)
        self.live = None
        return {"tokens": n, "bytes": n, "ms": 1.0}


def conv(seed, n):
    """A conversation's tokens: distinct per seed from the first token on."""
    return [seed * 100000 + i for i in range(n)]


class DiskTierTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.engine = FakeEngine()
        self.logs = []
        self.tier = self.new_tier()

    def tearDown(self):
        self.tmp.cleanup()

    def new_tier(self, **kw):
        args = dict(min_tokens=100, resave_growth=50, min_gain=10, max_gib=1.0, log=self.logs.append)
        args.update(kw)
        return DiskTier(self.dir, Refused, **args)

    def saves(self):
        return [c for c in self.engine.calls if c[0] == "save"]

    def restores(self):
        return [c for c in self.engine.calls if c[0] == "restore"]

    def test_leaving_a_long_conversation_saves_it(self):
        self.engine.run(conv(1, 300))
        self.tier.before(self.engine, conv(2, 50))
        self.assertEqual(len(self.saves()), 1)
        self.assertEqual(len(self.tier.entries), 1)

    def test_a_short_conversation_is_not_saved(self):
        self.engine.run(conv(1, 80))
        self.tier.before(self.engine, conv(2, 50))
        self.assertEqual(self.saves(), [])

    def test_continuing_the_live_conversation_neither_saves_nor_restores(self):
        a = conv(1, 300)
        self.engine.run(a)
        self.tier.before(self.engine, a + [7, 8, 9])
        self.assertEqual(self.engine.calls, [])

    def test_the_conversation_being_resumed_is_restored(self):
        a, b = conv(1, 300), conv(2, 300)
        self.engine.run(a)
        self.tier.before(self.engine, b)                    # leaves a: saved
        self.engine.run(b)
        self.tier.before(self.engine, a + [1, 2, 3])        # back to a: b saved, a restored
        self.assertEqual(len(self.saves()), 2)
        self.assertEqual(len(self.restores()), 1)
        self.assertEqual(self.engine.live["ids"], a)        # the tier records what the file holds

    def test_a_file_that_did_not_grow_enough_is_not_written_again(self):
        a, b = conv(1, 300), conv(2, 300)
        self.engine.run(a)
        self.tier.before(self.engine, b)
        self.engine.run(b)
        self.tier.before(self.engine, a + [1])              # a restored, b saved
        self.engine.run(a + [1] * 20)                       # a grew by 20 < resave_growth 50
        self.tier.before(self.engine, b + [1])
        self.assertEqual(len(self.saves()), 2, "a was not written a second time")
        self.assertEqual(len(self.restores()), 2, "b still comes back from its file")

    def test_growth_replaces_the_older_file(self):
        a = conv(1, 300)
        self.engine.run(a)
        self.tier.before(self.engine, conv(2, 50))
        first = set(self.tier.entries)
        self.engine.run(a + list(range(80)))                # a grew by 80 >= 50
        self.tier.before(self.engine, conv(3, 50))
        self.assertEqual(len(self.saves()), 2)
        self.assertEqual(len(self.tier.entries), 1, "the grown file supersedes the old one")
        self.assertNotEqual(set(self.tier.entries), first)
        self.assertFalse(any(os.path.exists(os.path.join(self.dir, n + ".bin")) for n in first))

    def test_the_next_turn_of_a_saved_conversation_is_not_written_again(self):
        # The chat template drops the reasoning from the history, so the next prompt diverges from the saved file
        # where the previous reply begins: on omen a 189,974-token file shared only 187,800 tokens with the next
        # prompt. Requiring the file to be an exact prefix wrote a new ~3 GB file at every switch.
        prompt = conv(1, 300)
        reply_with_reasoning, visible_reply = list(range(900, 920)), list(range(950, 955))
        self.engine.run(prompt, reply_with_reasoning)
        self.tier.before(self.engine, conv(2, 50))          # leaves it: saved with the reasoning
        next_turn = prompt + visible_reply + [7, 8, 9]      # what the template renders next time
        self.engine.restart()
        self.tier.before(self.engine, next_turn)            # restored: shares the 300-token prompt
        self.engine.run(next_turn, list(range(960, 970)))   # grew by 8 + 10, far below resave_growth 50
        self.tier.before(self.engine, conv(3, 50))          # leaves it again
        self.assertEqual(len(self.saves()), 1, "the same conversation, barely grown, is not saved again")
        self.assertEqual(len(self.tier.entries), 1)

    def test_a_grown_next_turn_replaces_its_older_file(self):
        prompt = conv(1, 300)
        self.engine.run(prompt, list(range(900, 920)))
        self.tier.before(self.engine, conv(2, 50))
        first = set(self.tier.entries)
        self.engine.run(prompt + list(range(950, 1050)))    # grew by 100 past the shared prompt >= 50
        self.tier.before(self.engine, conv(3, 50))
        self.assertEqual(len(self.saves()), 2)
        self.assertEqual(len(self.tier.entries), 1, "the older file of the same conversation is removed")
        self.assertNotEqual(set(self.tier.entries), first)

    def test_another_conversation_sharing_only_the_system_prompt_is_kept(self):
        system = conv(9, 120)                               # a shared system prompt, above min_tokens 100
        a, b = system + conv(1, 200), system + conv(2, 200)
        self.engine.run(a)
        self.tier.before(self.engine, conv(5, 50))
        self.engine.run(b)
        self.tier.before(self.engine, conv(6, 50))
        self.assertEqual(len(self.tier.entries), 2, "b shares 120 of a's 320 tokens: not the same conversation")

    def test_a_restarted_engine_gets_its_conversation_back(self):
        a = conv(1, 300)
        self.engine.run(a)
        self.tier.before(self.engine, conv(2, 50))          # a saved
        self.engine.restart()                               # idle unload, crash: nothing live
        self.tier.before(self.engine, a + [5])
        self.assertEqual(len(self.restores()), 1)

    def test_too_small_a_gain_restores_nothing(self):
        a = conv(1, 300)
        self.engine.run(a)
        self.tier.before(self.engine, conv(2, 50))
        self.engine.run(a[:295])                            # live already shares 295 of a
        self.tier.before(self.engine, a + [5])              # the file would add only 5 < min_gain 10
        self.assertEqual(self.restores(), [])

    def test_an_invalid_file_is_dropped_and_a_busy_one_kept(self):
        a = conv(1, 300)
        self.engine.run(a)
        self.tier.before(self.engine, conv(2, 50))
        self.engine.restart()
        self.engine.refuse["restore"] = "memory"
        self.tier.before(self.engine, a)
        self.assertEqual(len(self.tier.entries), 1, "a memory refusal keeps the file for later")
        self.engine.refuse["restore"] = "invalid"
        self.tier.before(self.engine, a)
        self.assertEqual(len(self.tier.entries), 0, "an invalid file never loads: it goes")

    def test_a_refused_save_lets_the_request_go_on(self):
        self.engine.run(conv(1, 300))
        self.engine.refuse["save"] = "storage"
        self.tier.before(self.engine, conv(2, 50))          # must not raise
        self.assertEqual(self.tier.entries, {})

    def test_the_budget_evicts_least_recently_used(self):
        self.tier = self.new_tier(max_gib=650 / (1 << 30))  # room for two 300-token files
        for seed in (1, 2, 3):
            self.engine.run(conv(seed, 300))
            self.tier.before(self.engine, conv(seed + 10, 50))
        self.assertEqual(len(self.tier.entries), 2)
        held = sorted(self.tier.entries.values(), key=lambda e: e.used)
        self.assertEqual(int(held[0].ids[0]) // 100000, 2, "conversation 1, the oldest, went")

    def test_the_index_survives_a_restart_and_leaves_other_files_alone(self):
        self.engine.run(conv(1, 300))
        self.tier.before(self.engine, conv(2, 50))
        manual = os.path.join(self.dir, "chat1.bin")
        with open(manual, "wb") as f:
            f.write(b"manual")
        orphan = os.path.join(self.dir, "tier-dead-1.bin")
        with open(orphan, "wb") as f:
            f.write(b"half a pair")
        again = self.new_tier()
        self.assertEqual(len(again.entries), 1)
        self.assertTrue(os.path.exists(manual), "a manual slot file is never touched")
        self.assertFalse(os.path.exists(orphan), "a tier file without its ids is removed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
