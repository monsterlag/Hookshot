"""Actual startup marker/control prefix and cached-compute selection, without Vulkan or game data."""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT
CXX = "g++"
COMMANDS = []
FLAGS = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)


def owned(temporary):
    path = Path(temporary.name).resolve()
    if path.parent != Path(tempfile.gettempdir()).resolve() or not path.name.startswith("precompile-backoff-"):
        raise AssertionError("Not an authored precompile-backoff temporary directory")
    return path


def cleanup(temporary):
    owned(temporary)
    temporary.cleanup()


def invoke(command, **kwargs):
    start = datetime.now(timezone.utc).isoformat()
    timer = time.monotonic()
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=45, creationflags=FLAGS, **kwargs)
    COMMANDS.append({"command": list(map(str, command)), "started_utc": start,
                     "ended_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.monotonic() - timer,
                     "exit": result.returncode, "output": result.stdout.decode("utf-8", "replace")})
    return result


FIXTURE = r'''
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
struct Program { bool compute = false; };
struct DiskRecord { unsigned kind; };
std::unordered_map<uint64_t, std::vector<DiskRecord>> disk_records;
std::unordered_map<uint64_t, std::vector<std::unique_ptr<Program>>> programs;
std::mutex disk_mu, shader_mu;
bool disk_on = true;
std::string disk_path;
FILE* disk_in = nullptr;
void disk_init() {}
void program_cache_read(FILE*, std::mutex*, uint64_t, const std::vector<DiskRecord>&,
                        std::vector<std::unique_ptr<Program>>&) {}
struct Properties { uint32_t vendorID = 0x8086, deviceID = 7, driverVersion = 9; uint8_t pipelineCacheUUID[16]{}; };
struct Device { Properties props; } dev;
uint64_t stamp = 42;
uint64_t program_cache_stamp() { return stamp; }
void logf(const char* format, ...) {
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fputc('\n', stderr);
}
@HELPERS@
@SELECTION@
std::string notes_path, calls_path;
bool interrupt_attempt = false;
int prepare_calls = 0, compute_callbacks = 0;
std::string pipe_notes_path() { return notes_path; }
void pcache_init() {
    prepare_calls++;
    program_cache_load_some({}, [](Program& p) { if (p.compute) compute_callbacks++; });
    if (FILE* f = fopen(calls_path.c_str(), "ab")) fprintf(f, "%d %d\n", prepare_calls, compute_callbacks), fclose(f);
    if (interrupt_attempt) std::_Exit(23);   // Only this authored child; no signal, GPU or memory stress.
}
@PREFIX@
    (void)progress; (void)t0;
}
int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    if (argc < 3 || argc > 4) return 3;
    const fs::path root = fs::canonical(argv[2]);
    if (root.parent_path() != fs::canonical(fs::temp_directory_path()) ||
        root.filename().string().rfind("precompile-backoff-case-", 0) != 0) return 3;
    notes_path = (root / "authored-notes.bin").string();
    calls_path = (root / "calls.txt").string();
    if (argc == 4) {
        const std::string change = argv[3];
        if (change == "stamp") stamp++;
        else if (change == "vendor") dev.props.vendorID++;
        else if (change == "device") dev.props.deviceID++;
        else if (change == "driver") dev.props.driverVersion++;
        else if (change == "uuid") dev.props.pipelineCacheUUID[0]++;
        else return 3;
    }
    disk_records[42].push_back({1});
    auto p = std::make_unique<Program>();
    p->compute = true;
    programs[42].push_back(std::move(p));
    const std::string mode = argv[1], marker = notes_path + ".running";
    if (mode == "seed") {
#if HAS_BACKOFF
        const std::string identity = precompile_backoff_identity();
        if (!precompile_backoff_begin(marker, identity) || !precompile_backoff_begin(marker, identity)) return 4;
#else
        if (FILE* f = fopen(marker.c_str(), "wb")) fputs("2\n", f), fclose(f); else return 3;
#endif
        return 0;
    }
    if (mode == "complete") {
#if HAS_BACKOFF
        precompile_backoff_complete(marker, precompile_backoff_identity());
        return 0;
#else
        return 3;
#endif
    }
    if (mode != "run" && mode != "interrupt") return 3;
    interrupt_attempt = mode == "interrupt";
    precompile_run(nullptr);
    return 0;
}
'''


class PrecompileBackoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="precompile-backoff-build-")
        cls.addClassCleanup(cleanup, cls.temporary)
        cls.build = owned(cls.temporary)
        draw = (SOURCE / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
        shader = (SOURCE / "gpu/vk/vk_shader.cpp").read_text(encoding="utf-8")
        start = draw.index("void precompile_run(void (*progress)(size_t done, size_t total)) {")
        prefix_end = draw.index("    pcache_init();", start) + len("    pcache_init();")
        prefix = draw[start:prefix_end]
        start = shader.index("size_t program_cache_load_some(")
        selection = shader[start:shader.index("\n}\n", start) + 2]
        has_backoff = "struct PrecompileBackoff {" in draw
        helpers = draw[draw.index("struct PrecompileBackoff {"):draw.index("/// Every shader of the game, before its first start")] if has_backoff else ""
        fixture = FIXTURE.replace("@HELPERS@", helpers).replace("@SELECTION@", selection).replace("@PREFIX@", prefix)
        source, cls.exe = cls.build / "fixture.cpp", cls.build / "fixture.exe"
        source.write_text(fixture, encoding="utf-8", newline="\n")
        env = dict(os.environ)
        compiler = Path(shutil.which(CXX) or CXX).resolve()
        if os.name == "nt" and compiler.name.lower() not in ("cl.exe", "clang-cl.exe"):
            for key in ("INCLUDE", "LIB", "LIBPATH"):
                env.pop(key, None)
            env["PATH"] = str(compiler.parent) + os.pathsep + str(Path(env["SYSTEMROOT"]) / "System32")
        # Python's cached temp base and the C++ runtime's environment lookup may differ.
        # Keep subprocesses in the same verified base without relaxing the native ownership guard.
        for key in ("TMPDIR", "TEMP", "TMP"):
            env[key] = str(cls.build.parent)
        cls.runtime_env = env
        command = [str(compiler), "-std=c++20", "-Wall", "-Wextra", "-Werror", "-DHAS_BACKOFF=" + str(int(has_backoff)),
                   str(source), "-o", str(cls.exe)]
        result = invoke(command, env=env, cwd=cls.build)
        if result.returncode:
            raise AssertionError(result.stdout.decode("utf-8", "replace"))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="precompile-backoff-case-")
        self.addCleanup(cleanup, self.temporary)
        self.root = owned(self.temporary)
        self.notes = self.root / "authored-notes.bin"
        self.marker = self.root / "authored-notes.bin.running"
        self.calls = self.root / "calls.txt"

    def call(self, mode, change=None, noopt=None, expected=0):
        env = dict(self.runtime_env)
        env.pop("RX_VK_NOOPT", None)
        if noopt is not None:
            env["RX_VK_NOOPT"] = noopt
        command = [str(self.exe), mode, str(self.root), *([change] if change else [])]
        result = invoke(command, env=env, cwd=self.root)
        self.assertEqual(result.returncode, expected, result.stdout.decode("utf-8", "replace"))
        return result.stdout.decode("utf-8", "replace")

    def test_exhausted_state_skips_repeated_preparation_without_notes(self):
        self.call("seed")
        before = self.marker.read_bytes()
        self.call("run")
        self.call("run")
        self.assertFalse(self.calls.exists(), "exhausted state must not reach cached-compute preparation")
        self.assertEqual(self.marker.read_bytes(), before)

    def test_two_actual_interrupted_prefixes_suppress_later_compute_preparation(self):
        self.call("interrupt", expected=23)
        self.call("interrupt", expected=23)
        before = self.marker.read_bytes()
        self.call("run")
        self.call("run")
        self.assertEqual(self.calls.read_text().splitlines(), ["1 1", "1 1"])
        self.assertEqual(self.marker.read_bytes(), before)

    def test_success_resets_and_fresh_no_note_compute_is_still_prepared(self):
        self.call("run")
        self.assertFalse(self.marker.exists())
        self.call("run")
        self.assertFalse(self.marker.exists())
        self.assertEqual(self.calls.read_text().splitlines(), ["1 1", "1 1"])

    def test_each_cache_or_device_identity_change_allows_fresh_preparation(self):
        for index, change in enumerate(("stamp", "vendor", "device", "driver", "uuid"), 1):
            with self.subTest(change=change):
                self.call("seed")
                self.call("run", change=change)
                self.assertFalse(self.marker.exists())
                self.assertEqual(len(self.calls.read_text().splitlines()), index)

    def test_noopt_uses_presence_including_zero_not_numeric_truth(self):
        self.call("seed", noopt="0")
        before = self.marker.read_bytes()
        self.call("run", noopt="1")
        self.assertEqual(self.marker.read_bytes(), before)
        self.assertFalse(self.calls.exists())
        self.call("run")
        self.assertFalse(self.marker.exists())
        self.assertEqual(self.calls.read_text().splitlines(), ["1 1"])

    def test_legacy_exhausted_marker_allows_only_one_scoped_interrupted_attempt(self):
        self.marker.write_bytes(b"2\n")
        self.call("interrupt", expected=23)
        before = self.marker.read_bytes()
        self.assertTrue(before.startswith(b"HookshotPrecompileBackoff 1\n"))
        self.assertTrue(before.endswith(b" 2\n"))
        self.call("run")
        self.call("run")
        self.assertEqual(self.marker.read_bytes(), before)
        self.assertEqual(self.calls.read_text().splitlines(), ["1 1"])

    def test_legacy_success_clears_marker(self):
        self.marker.write_bytes(b"200\n")
        self.call("run")
        self.assertFalse(self.marker.exists())
        self.assertEqual(self.calls.read_text().splitlines(), ["1 1"])

    def test_empty_or_malformed_old_marker_is_retried_once(self):
        for value in (b"", b"not a count\n", b"HookshotPrecompileBackoff 1\ntruncated", b"9" * 80):
            with self.subTest(value=value):
                self.marker.write_bytes(value)
                self.call("interrupt", expected=23)
                self.assertTrue(self.marker.read_bytes().endswith(b" 2\n"))
                before = self.calls.read_bytes()
                self.call("run")
                self.assertEqual(self.calls.read_bytes(), before)

    def test_skipping_keeps_notes_bytes_and_attributes(self):
        self.notes.write_bytes(b"authored notes must stay")
        self.notes.chmod(0o444)
        self.addCleanup(self.notes.chmod, 0o666)
        before = (self.notes.read_bytes(), self.notes.stat().st_mode, getattr(self.notes.stat(), "st_file_attributes", None))
        self.call("seed")
        self.call("run")
        self.call("run")
        self.assertEqual((self.notes.read_bytes(), self.notes.stat().st_mode, getattr(self.notes.stat(), "st_file_attributes", None)), before)

    def test_unrecordable_state_warns_without_removing_directory_or_changing_permissions(self):
        self.marker.mkdir()
        before = self.marker.stat().st_mode
        output = self.call("run")
        self.assertIn("could not record failure backoff", output)
        self.assertTrue(self.marker.is_dir())
        self.assertEqual(self.marker.stat().st_mode, before)
        self.assertEqual(self.calls.read_text().splitlines(), ["1 1"])

    def test_completion_cannot_remove_a_different_identity_record(self):
        self.call("seed", change="driver")
        before = self.marker.read_bytes()
        self.call("complete")
        self.assertEqual(self.marker.read_bytes(), before)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--receipt", type=Path)
    args, remaining = parser.parse_known_args()
    CXX, SOURCE = args.cxx, args.source
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    try:
        program = unittest.main(argv=[__file__, *remaining], exit=False)
    finally:
        if args.receipt:
            args.receipt.write_text(json.dumps({"source": str(SOURCE), "commands": COMMANDS}, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if program.result.wasSuccessful() else 1)
