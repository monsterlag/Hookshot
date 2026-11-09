"""Actual parallel_emit: bounded submission/payloads, ordered results and cleanup; tiny authored inputs only."""
import argparse
import ast
import collections
import concurrent.futures
import multiprocessing
import os
from pathlib import Path
import queue
import sys
import threading
import types
import unittest
from unittest import mock
import weakref


SOURCE = Path(__file__).resolve().parents[1] / "nx2/chunks.py"


def load_emit(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"_counts", "_emit_run", "parallel_emit"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {f.name for f in functions} == names
    module = types.ModuleType("authored_emit_under_test")
    module.__dict__.update(collections=collections, multiprocessing=multiprocessing, _job=None)
    sys.modules[module.__name__] = module   # Pool pickles the actual _emit_run by its module/name
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


class AuthoredEmitter:
    def __init__(self, fail=None):
        self.fail = fail

    def __call__(self, e, texts):
        if e == self.fail:
            raise ValueError("authored emit failure")
        return texts[e], {"ref_%d" % e}


class Payload(list):
    pass


class StoppedEmitter:
    def __init__(self, counts):
        self.parent_pid, self.counts = os.getpid(), counts

    def __call__(self, e, texts):
        if e == 4 and os.getpid() != self.parent_pid:
            os._exit(17)   # Authored worker death only; serial fallback in the parent must finish this same batch.
        self.counts["emitted"] += 1
        return texts[e], {"ref_%d" % e}


class Result:
    def __init__(self, function, args, pool):
        try:
            out, deltas = function(*args)
            self.value, self.error = (Payload(out), deltas), None
            pool.payloads.append(weakref.ref(self.value[0]))
        except Exception as error:
            self.value, self.error = None, error

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value


class RecordingExecutor:
    """Record actual production submissions; finish tiny tasks immediately, like fast later workers."""
    def __init__(self):
        self.submitted, self.payloads, self.closed, self.max_live = [], [], False, 0

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.closed = True

    def submit(self, function, *args):
        self.submitted.append(tuple(args[0]))
        result = Result(function, args, self)
        self.max_live = max(self.max_live, sum(ref() is not None for ref in self.payloads))
        return result

    def map(self, function, batches):
        # Current main eagerly submits every run before yielding the first ordered result.
        results = [self.submit(function, batch) for batch in batches]
        for result in results:
            yield result.result()


class ProbeEmitter:
    def __init__(self, gate, messages, count):
        self.gate, self.messages, self.count = gate, messages, count

    def __call__(self, e, texts):
        if e == 0:
            self.messages.put(("blocked",))
            if not self.gate.wait(20):
                raise RuntimeError("authored first-batch gate was not released")
        else:
            self.messages.put(("done", e))
        self.count["emitted"] += 1
        return texts[e], {"ref_%d" % e}


def fork_probe(source, gate, messages):
    """Called in a fresh single-threaded process; only then creates the actual fork Pool."""
    assert threading.active_count() == 1
    module = load_emit(source)
    context = multiprocessing.get_context("fork")
    real_executor = concurrent.futures.ProcessPoolExecutor

    class ObservedResult:
        def __init__(self, result, pool):
            self.future, self.pool = result, pool

        def result(self):
            if not self.pool.noted:
                self.pool.noted = True
                messages.put(("window", self.pool.submitted))
            return self.future.result()

    class ObservedExecutor:
        def __init__(self, max_workers, mp_context):
            assert threading.active_count() == 1
            self.pool = real_executor(max_workers=max_workers, mp_context=mp_context)
            self.submitted, self.noted = 0, False

        def __enter__(self):
            self.pool.__enter__()
            return self

        def __exit__(self, *error):
            return self.pool.__exit__(*error)

        def submit(self, function, *args):
            self.submitted += 1
            return ObservedResult(self.pool.submit(function, *args), self)

        def map(self, function, batches):
            return self.pool.map(function, batches)

    count = collections.Counter()
    texts = {e: "body_%d" % e for e in range(12)}
    try:
        with mock.patch.object(concurrent.futures, "ProcessPoolExecutor", ObservedExecutor):
            values, counter_order = [], []
            for value in module.parallel_emit(ProbeEmitter(gate, messages, count), list(texts), texts, [count], 2, run=1):
                values.append(value)
                counter_order.append(count["emitted"])
        messages.put(("finished", values, counter_order, module._job is None,
                      [p.pid for p in multiprocessing.active_children()]))
    except BaseException as error:
        messages.put(("error", repr(error)))


class EmitBacklogTests(unittest.TestCase):
    def setUp(self):
        self.module = load_emit(SOURCE)
        self.texts = {e: "body_%d" % e for e in range(12)}

    def fake(self, fail=None, jobs=2, order=None):
        pool = RecordingExecutor()
        context = object()
        patch = mock.patch.object(self.module.multiprocessing, "get_context", return_value=context)
        patch.start()
        self.addCleanup(patch.stop)
        executor = mock.patch.object(concurrent.futures, "ProcessPoolExecutor",
                                     side_effect=lambda **kwargs: pool)
        executor.start()
        self.addCleanup(executor.stop)
        made = self.module.parallel_emit(AuthoredEmitter(fail), list(self.texts) if order is None else order,
                                         self.texts, [], jobs, run=2)
        self.addCleanup(made.close)
        return pool, made

    def test_submission_window_and_consumed_payload_release(self):
        pool, made = self.fake()
        self.assertEqual(pool.submitted, [])
        got = []
        for e in self.texts:
            got.append(next(made))
            # Pausing within a yielded batch cannot feed more work to the pool.
            self.assertEqual(len(pool.submitted), min(6, 2 + e // 2))
            self.assertLessEqual(pool.max_live, 2, "yielded payload retained beside its replacement")
        self.assertEqual(list(made), [])
        self.assertEqual(got, [(e, self.texts[e], {"ref_%d" % e}) for e in self.texts])
        self.assertTrue(pool.closed)
        self.assertIsNone(self.module._job)
        self.assertTrue(all(ref() is None for ref in pool.payloads))

    def test_worker_exception_preserves_prior_order_and_cleans_up(self):
        pool, made = self.fake(fail=4)
        self.assertEqual([next(made)[0] for _ in range(4)], [0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, "authored emit failure"):
            next(made)
        self.assertTrue(pool.closed)
        self.assertIsNone(self.module._job)

    def test_early_consumer_close_cleans_up(self):
        pool, made = self.fake()
        self.assertEqual(next(made)[0], 0)
        made.close()
        self.assertTrue(pool.closed)
        self.assertIsNone(self.module._job)
        self.assertTrue(all(ref() is None for ref in pool.payloads))

    def test_serial_and_short_inputs_do_not_create_a_pool(self):
        for jobs, order in ((1, list(self.texts)), (4, [0, 1]), (4, [])):
            with self.subTest(jobs=jobs, order=order):
                pool, made = self.fake(jobs=jobs, order=order)
                self.assertEqual([v[0] for v in made], order)
                self.assertEqual(pool.submitted, [])

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "real fork cleanup runs on POSIX")
    def test_broken_worker_falls_back_without_losing_or_doubling_results(self):
        self.assertEqual(threading.active_count(), 1)
        before = {p.pid for p in multiprocessing.active_children()}
        counts = collections.Counter()
        values = list(self.module.parallel_emit(StoppedEmitter(counts), list(self.texts), self.texts,
                                                [counts], 2, run=2))
        self.assertEqual(values, [(e, self.texts[e], {"ref_%d" % e}) for e in self.texts])
        self.assertEqual(counts, {"emitted": 12})
        self.assertIsNone(self.module._job)
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "real fork cleanup runs on POSIX")
    def test_real_worker_exception_and_early_close_cleanup(self):
        self.assertEqual(threading.active_count(), 1)
        before = {p.pid for p in multiprocessing.active_children()}
        made = self.module.parallel_emit(AuthoredEmitter(fail=4), list(self.texts), self.texts, [], 2, run=2)
        self.assertEqual([next(made)[0] for _ in range(4)], [0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, "authored emit failure"):
            next(made)
        self.assertIsNone(self.module._job)
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)
        self.assertEqual(threading.active_count(), 1)
        made = self.module.parallel_emit(AuthoredEmitter(), list(self.texts), self.texts, [], 2, run=2)
        self.assertEqual(next(made)[0], 0)
        made.close()
        self.assertIsNone(self.module._job)
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "real fork probe runs on POSIX")
    def test_real_fork_stalled_first_batch(self):
        # The coordinator starts from this single-threaded test process, never a thread calling fork.
        self.assertEqual(threading.active_count(), 1)
        ctx = multiprocessing.get_context("fork")
        gate, messages = ctx.Event(), ctx.Queue()
        child = ctx.Process(target=fork_probe, args=(SOURCE, gate, messages))
        child.start()
        blocked, done, window = False, set(), None
        try:
            while True:
                message = messages.get(timeout=15)   # timeout is a failure/cleanup deadline, never evidence of passing
                if message[0] == "blocked":
                    blocked = True
                elif message[0] == "done":
                    done.add(message[1])
                elif message[0] == "window":
                    window = message[1]
                else:
                    self.fail("unexpected probe message: %r" % (message,))
                if blocked and ((window is not None and len(done) >= 1) or len(done) == 11):
                    break
            # Old production reaches all eleven later completions while the first remains blocked.
            before_release = len(done)
            gate.set()
            while True:
                message = messages.get(timeout=15)
                if message[0] == "finished":
                    break
                if message[0] == "error":
                    self.fail(message[1])
            child.join(15)
            self.assertFalse(child.is_alive())
            self.assertEqual(child.exitcode, 0)
            _, values, counter_order, cleared, remaining = message
            self.assertEqual([v[0] for v in values], list(self.texts))
            self.assertEqual(counter_order, list(range(1, 13)))
            self.assertTrue(cleared)
            self.assertEqual(remaining, [])
            print("real fork: blocked first batch, %d later completions, window %s; ordered results/counters, no children"
                  % (before_release, window))
            self.assertEqual(window, 2, "production did not establish the two-batch submission window")
            self.assertEqual(before_release, 1, "later results ran beyond the bounded window")
        except queue.Empty:
            self.fail("authored fork probe missed its progress/deadline message")
        finally:
            gate.set()
            child.join(5)
            if child.is_alive():
                child.terminate()   # only this fixture's own timed-out coordinator
                child.join(5)
            messages.close()
            messages.join_thread()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args, rest = parser.parse_known_args()
    SOURCE = args.source
    unittest.main(argv=[sys.argv[0], *rest])
