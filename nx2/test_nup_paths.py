"""Game-free extraction boundary regressions: python -I nx2/test_nup_paths.py."""

from contextlib import contextmanager, ExitStack
import io
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nup


class WriteAttempt(AssertionError):
    pass


@contextmanager
def guard_io(root):
    """Keep unsafe probes inside this fixture, without faking successful metadata reads."""
    roots = {os.path.normcase(os.path.abspath(root)), os.path.normcase(os.path.realpath(root))}
    attempts = []

    def metadata(function):
        def checked(path, *args, **kwargs):
            candidate = os.path.normcase(os.path.abspath(path))
            for owned in roots:
                try:
                    common = os.path.commonpath((candidate, owned))
                except ValueError:
                    continue
                if common in (candidate, owned):   # realpath may inspect the fixture's ancestors
                    return function(path, *args, **kwargs)
            raise AssertionError("metadata access outside fixture: %r" % path)
        return checked

    def writing(name):
        def blocked(*args, **kwargs):
            attempts.append(name)
            raise WriteAttempt(name)
        return blocked

    with ExitStack() as stack:
        for name in ("stat", "lstat"):
            stack.enter_context(patch.object(os, name, side_effect=metadata(getattr(os, name))))
        for name in ("makedirs", "mkdir", "open", "replace", "rename", "renames", "link", "symlink",
                     "unlink", "remove", "rmdir", "removedirs"):
            stack.enter_context(patch.object(os, name, side_effect=writing("os." + name)))
        stack.enter_context(patch("builtins.open", side_effect=writing("builtins.open")))
        stack.enter_context(patch.object(io, "open", side_effect=writing("io.open")))
        yield attempts


def snapshot(root):
    """Record directories and link targets too; never descend into a link or junction."""
    result = {}
    pending = [root]
    while pending:
        for path in pending.pop().iterdir():
            info = path.lstat()
            relative = path.relative_to(root)
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                result[relative] = ("link", os.readlink(path))
            elif stat.S_ISDIR(info.st_mode):
                result[relative] = ("directory",)
                pending.append(path)
            else:
                result[relative] = ("file", path.read_bytes())
    return result


class Package:
    def __init__(self, names, data=b"authored fixture"):
        self.names, self.data = names, data

    def walk(self, directory):
        for name in self.names:
            yield name, (0, 0, len(self.data), 0, len(self.data))

    def copy_to(self, record, output, name):
        output.write(self.data)


class ExtractionPaths(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.out = self.root / "out"
        self.out.mkdir()
        self.sentinel = self.root / "outside.bin"
        self.sentinel.write_bytes(b"keep outside data")

    def extract(self, names):
        return nup.extract(Package(names), {"romfs": 0}, str(self.out), only="romfs", quiet=True)

    def reject(self, names, message):
        before = snapshot(self.root)
        with self.assertRaisesRegex(SystemExit, "^" + re.escape(message)):
            self.extract(names)
        self.assertEqual(before, snapshot(self.root))

    def directory_link(self, link, target):
        try:
            os.symlink(target, link, target_is_directory=True)
        except OSError as error:
            if os.name != "nt":
                raise
            # Junctions need no Developer Mode or symbolic-link privilege on Windows.
            result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                    capture_output=True, text=True)
            if result.returncode:
                self.fail("Cannot create the link fixture: %s; %s" % (error, result.stderr))

    def test_nested_names_and_resume(self):
        names = ["Map/Forest/data.bin", "Text/Français.bin", "com10.txt", ".hidden", "conifer.txt"]
        self.assertEqual(self.extract(names), (len(names), len(names) * len(b"authored fixture")))
        for name in names:
            self.assertEqual((self.out / "romfs" / name).read_bytes(), b"authored fixture")
        self.assertEqual(self.extract(names), (0, 0))

    def test_paths_cannot_leave_the_output(self):
        names = ["", ".", "..", "../../outside.bin", "nested/../../outside.bin", "nested/../file.bin",
                 "/rooted.bin", "//server/share/file.bin", r"\rooted.bin", r"\\server\share\file.bin",
                 r"C:\outside.bin", "C:/outside.bin", "C:outside.bin", r"nested\..\outside.bin",
                 "nested//file.bin", "nested/.. /outside.bin", "file:stream", "trailing.", "trailing ",
                 "control\x00.bin", "CON", "aux.txt", "NUL.bin", "COM1", "LPT9.txt", "COM¹.txt"]
        for name in names:
            with self.subTest(name=name):
                # Even a broken validator must not let the fixture touch an absolute path or UNC share.
                before = snapshot(self.root)
                with guard_io(self.root) as attempts:
                    with self.assertRaisesRegex(SystemExit, "^" + re.escape("unsafe path in the package: %r" % name) + "$"):
                        self.extract([name])
                    self.assertEqual(attempts, [])
                    # In this exact context, a valid name must pass validation and reach a write.
                    with self.assertRaisesRegex(WriteAttempt, "^os\\.makedirs$"):
                        self.extract(["valid.bin"])
                    self.assertEqual(attempts, ["os.makedirs"])
                self.assertEqual(before, snapshot(self.root))

    def test_write_guards_block_alternative_io(self):
        before = snapshot(self.root)
        operations = (("builtins.open", lambda: open(self.sentinel, "wb")),
                      ("io.open", lambda: io.open(self.sentinel, "wb")),
                      ("os.open", lambda: os.open(self.sentinel, os.O_WRONLY | os.O_TRUNC)),
                      ("os.mkdir", lambda: os.mkdir(self.root / "new")),
                      ("os.rename", lambda: os.rename(self.sentinel, self.root / "renamed")),
                      ("os.link", lambda: os.link(self.sentinel, self.root / "linked")))
        with guard_io(self.root) as attempts:
            for name, operation in operations:
                with self.subTest(operation=name), self.assertRaisesRegex(WriteAttempt, "^" + re.escape(name) + "$"):
                    result = operation()
                    if name == "os.open":
                        os.close(result)
                    elif hasattr(result, "close"):
                        result.close()
            self.assertEqual(attempts, [name for name, _ in operations])
        self.assertEqual(before, snapshot(self.root))

    def test_parent_path_cannot_overwrite_a_sibling(self):
        # This also safely demonstrates the original failure: both files belong to this fixture.
        self.reject(["../../outside.bin"], "unsafe path in the package: '../../outside.bin'")

    def test_ordinary_partial_file_is_replaced(self):
        (self.out / "romfs").mkdir()
        partial = self.out / "romfs" / "file.bin.part"
        partial.write_bytes(b"an interrupted extraction")
        self.extract(["file.bin"])
        self.assertEqual((self.out / "romfs" / "file.bin").read_bytes(), b"authored fixture")
        self.assertFalse(partial.exists())

    def test_part_directory_link_cannot_redirect_extraction(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.directory_link(self.out / "romfs", outside)
        self.reject(["new.bin"], "extraction path is a link or reparse point: ")

    def test_nested_directory_link_cannot_redirect_extraction(self):
        (self.out / "romfs").mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        self.directory_link(self.out / "romfs" / "nested", outside)
        self.reject(["nested/new.bin"], "extraction path is a link or reparse point: ")

    def test_directory_reparse_point_cannot_redirect_extraction(self):
        outside = self.root / "outside"
        outside.mkdir()
        link = self.out / "romfs"
        if os.name == "nt":
            result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            os.symlink(outside, link, target_is_directory=True)
        self.reject(["new.bin"], "extraction path is a link or reparse point: ")

    def test_destination_link_is_rejected_before_resume(self):
        (self.out / "romfs").mkdir()
        try:
            os.symlink(self.sentinel, self.out / "romfs" / "file.bin")
        except OSError as error:
            self.fail("Cannot create the file-link fixture: %s" % error)
        self.reject(["file.bin"], "extraction path is a link or reparse point: ")

    def test_temporary_link_cannot_truncate_an_outside_file(self):
        (self.out / "romfs").mkdir()
        try:
            os.symlink(self.sentinel, self.out / "romfs" / "file.bin.part")
        except OSError as error:
            self.fail("Cannot create the file-link fixture: %s" % error)
        self.reject(["file.bin"], "extraction path is a link or reparse point: ")

    def test_temporary_hard_link_cannot_truncate_an_outside_file(self):
        (self.out / "romfs").mkdir()
        os.link(self.sentinel, self.out / "romfs" / "file.bin.part")
        self.reject(["file.bin"], "extraction temporary file has other hard links: ")


if __name__ == "__main__":
    unittest.main()
