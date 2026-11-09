# Layout creation failures

`ensure_layout` publishes a program's layout and descriptor metadata only after both
Vulkan layout creations succeed. If descriptor-set creation fails, no dependent API
is called. If pipeline-layout creation fails, the newly created descriptor-set
layout is destroyed. The program remains eligible for a later retry.

Draw, dispatch, preparation, cached precompilation and prediction stop the affected
pipeline work when layout creation fails. Each creation operation logs its first
eight failures with the program key and `VkResult`, followed by one suppression
message. This does not diagnose a particular driver crash.

Run the game-free regression with a C++20 Clang or GCC and Vulkan headers:

```sh
python -I -B tests/test_layout_failures.py --cxx clang++ --headers /path/to/Vulkan-Headers/include
```

The normal `tests/run_all.py --strict` runner also discovers this check. It compiles
the actual layout helper and selected caller blocks with authored program metadata
and API spies. The eight cases cover transactional failure/retry, successful
arguments and the existing-layout fast path, draw/dispatch guards, compute warming,
graphics seed cleanup, cached compute/graphics notes, prediction borrow cleanup,
and bounded result logging. The fixture disables only the missing-field-initializer
warning used by the production Vulkan `sType` aggregate initialization; other
enabled warnings are errors. GCC maps the shared C header's `_Static_assert` spelling
to `static_assert` in this fixture.

For a baseline comparison, save the original `gpu/vk/vk_draw.cpp` outside the source
tree and pass its path with `--baseline-source`. The check compares successful API
arguments for authored vertex/fragment descriptors and verifies that the original
helper calls a dependent API after a failed descriptor-set allocation.
The success case also checks that mesh-fallback buffer addresses require push
constants even when render-area and global-memory push users are absent.

No Vulkan driver, game data or GPU is used. The draw/dispatch wrappers observe the
actual early-return blocks, not the rest of the renderer. Cached workers execute
serially; concurrent cache ownership, real allocation exhaustion and driver behavior
need separate validation. These checks do not establish the cause of issue #803.
