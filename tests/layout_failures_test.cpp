// Actual production blocks, authored programs and Vulkan API spies; no driver is loaded.
#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan.h>
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/shader_info.h"
// The shared C header uses the C spelling; Clang accepts it as an extension.
#ifndef __clang__
#define _Static_assert static_assert
#endif
#include "vk_state.h"
#ifndef __clang__
#undef _Static_assert
#endif

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x); std::exit(1); } } while (0)
#define PROF(x) ((void)0)
namespace mkwvk {
struct Texture;
#include "layout_types.inc"
template<class T> T handle(uintptr_t n) {
    if constexpr (std::is_pointer_v<T>) return reinterpret_cast<T>(n);
    else return static_cast<T>(n);
}
template<class T> uintptr_t id(T h) {
    if constexpr (std::is_pointer_v<T>) return reinterpret_cast<uintptr_t>(h);
    else return static_cast<uintptr_t>(h);
}
struct Counts { int sets=0, layouts=0, destroys=0, computes=0, graphics=0, libs=0, made=0, invalid=0; } counts;
struct Device { VkDevice device=handle<VkDevice>(1); bool push_descriptor=true; uint32_t max_push_descriptors=64; } dev;
bool vkCmdPushDescriptorSetKHR = true;
int failure = 0, after_guard = 0;
uintptr_t next_handle = 100;
std::set<uintptr_t> sets, layouts, modules, pipes;
std::vector<std::string> logs;
std::vector<uint64_t> arguments;
void logf(const char* fmt, ...) {
    char out[512]; va_list args; va_start(args, fmt); std::vsnprintf(out, sizeof out, fmt, args); va_end(args);
    logs.emplace_back(out);
}
VkResult vkCreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo* ci,
                                    const VkAllocationCallbacks* allocator, VkDescriptorSetLayout* out) {
    CHECK(device == dev.device && !allocator && ci->sType == VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
    ++counts.sets;
    arguments.insert(arguments.end(), {1, ci->flags, ci->bindingCount});
    for (uint32_t i=0; i<ci->bindingCount; ++i) {
        const auto& b=ci->pBindings[i];
        CHECK(!b.pImmutableSamplers);
        arguments.insert(arguments.end(), {b.binding, static_cast<uint64_t>(b.descriptorType), b.descriptorCount, b.stageFlags});
    }
    if (failure == 1) return VK_ERROR_OUT_OF_HOST_MEMORY;
    *out=handle<VkDescriptorSetLayout>(++next_handle); sets.insert(id(*out));
    return VK_SUCCESS;
}
VkResult vkCreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo* ci,
                               const VkAllocationCallbacks* allocator, VkPipelineLayout* out) {
    CHECK(device == dev.device && !allocator && ci->sType == VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
    ++counts.layouts;
    CHECK(ci->setLayoutCount == 1 && ci->pushConstantRangeCount == 1);
    if (!sets.count(id(ci->pSetLayouts[0]))) ++counts.invalid;
    arguments.insert(arguments.end(), {2, ci->flags, ci->setLayoutCount, id(ci->pSetLayouts[0]),
        ci->pushConstantRangeCount, ci->pPushConstantRanges[0].stageFlags,
        ci->pPushConstantRanges[0].offset, ci->pPushConstantRanges[0].size});
    if (failure == 2 || counts.invalid) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    *out=handle<VkPipelineLayout>(++next_handle); layouts.insert(id(*out));
    return VK_SUCCESS;
}
void vkDestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout set, const VkAllocationCallbacks* allocator) {
    CHECK(device == dev.device && !allocator && sets.erase(id(set)) == 1); ++counts.destroys;
}
void vkDestroyPipelineLayout(VkDevice device, VkPipelineLayout layout, const VkAllocationCallbacks* allocator) {
    CHECK(device == dev.device && !allocator && layouts.erase(id(layout)) == 1);
}
void vkDestroyShaderModule(VkDevice device, VkShaderModule module, const VkAllocationCallbacks* allocator) {
    CHECK(device == dev.device && !allocator && modules.erase(id(module)) == 1);
}
void vkDestroyPipeline(VkDevice device, VkPipeline pipe, const VkAllocationCallbacks* allocator) {
    CHECK(device == dev.device && !allocator && pipes.erase(id(pipe)) == 1);
}
VkPipeline new_pipeline() { const auto p=handle<VkPipeline>(++next_handle); pipes.insert(id(p)); return p; }
VkResult vkCreateComputePipelines(VkDevice device, VkPipelineCache, uint32_t count,
                                 const VkComputePipelineCreateInfo* ci, const VkAllocationCallbacks*, VkPipeline* out) {
    CHECK(device == dev.device && count == 1);
    ++counts.computes;
    CHECK(ci->stage.stage == VK_SHADER_STAGE_COMPUTE_BIT && ci->stage.module && !ci->stage.pSpecializationInfo);
    CHECK(layouts.count(id(ci->layout)));
    *out=new_pipeline(); return VK_SUCCESS;
}
VkResult guarded_compute(VkPipelineCache cache, const VkComputePipelineCreateInfo* ci, VkPipeline* out) {
    return vkCreateComputePipelines(dev.device, cache, 1, ci, nullptr, out);
}
void pin_subgroup(VkPipelineShaderStageCreateInfo&, uint32_t) {}
bool taau_key(uint64_t) { return false; }
VkPipelineCache pcache=VK_NULL_HANDLE;
std::atomic<int64_t> pcache_built_since_save{0};
int held_locks=0;
struct ParLock { ParLock() { ++held_locks; } ~ParLock() { --held_locks; } };
struct SkipSite { std::atomic<int64_t> n{0}; } skip_sites[1];
int64_t st_skipped=0;
int skip_slot(int) { return 0; }
std::mutex note_mu;
std::unordered_map<uint64_t, std::unique_ptr<PipeEntry>> pipelines;
VkPipeline build_graphics(const Program& p, const NvnState&, const Targets&, VkPrimitiveTopology, int = -1) {
    ++counts.graphics; CHECK(layouts.count(id(p.layout))); return new_pipeline();
}
PipeEntry* graphics_pipeline(const Program& p, const NvnState&, const Targets&, VkPrimitiveTopology) {
    ++counts.graphics; CHECK(layouts.count(id(p.layout))); static PipeEntry entry; return &entry;
}
Targets note_targets(const PipeNote&) { return {}; }
std::vector<Program*> cached;
std::function<std::unique_ptr<Program>()> factory;
std::unique_ptr<Program> seed_program(size_t, int) { return factory(); }
size_t program_cache_load_some(const std::unordered_set<uint64_t>&, const std::function<void(Program&)>& fn) {
    for (Program* p : cached) fn(*p);
    return cached.size();
}
const Program* program_find(uint64_t key, uint64_t variant) {
    for (Program* p : cached) if (p->key == key && p->variant == variant) return p;
    return nullptr;
}
void program_cache_load(uint64_t, std::vector<std::unique_ptr<Program>>& variants, bool keep) {
    CHECK(keep); variants.push_back(factory());
}

#include "layout_helper.inc"
#include "layout_baseline.inc"
#include "layout_callers.inc"

std::unique_ptr<Program> program(bool compute) {
    auto p=std::make_unique<Program>(); p->key=7; p->variant=9; p->compute=compute;
    Stage st; st.vk_stage=compute ? VK_SHADER_STAGE_COMPUTE_BIT : VK_SHADER_STAGE_VERTEX_BIT;
    st.module=handle<VkShaderModule>(++next_handle); modules.insert(id(st.module));
    auto add=[](auto& ds) { typename std::decay_t<decltype(ds)>::value_type d{}; d.count=2; ds.push_back(d); };
    add(st.info.constant_buffer_descriptors); add(st.info.storage_buffers_descriptors);
    add(st.info.texture_buffer_descriptors); add(st.info.image_buffer_descriptors);
    add(st.info.texture_descriptors); add(st.info.image_descriptors);
    st.info.uses_global_memory=true;
    st.info.uses_render_area=true;
    p->stages.push_back(std::move(st));
    if (!compute) {
        Stage fragment; fragment.vk_stage=VK_SHADER_STAGE_FRAGMENT_BIT;
        fragment.module=handle<VkShaderModule>(++next_handle); modules.insert(id(fragment.module));
        add(fragment.info.texture_descriptors);
        p->stages.push_back(std::move(fragment));
    }
    return p;
}
void cleanup(Program& p) {
    if (p.compute_pipeline) vkDestroyPipeline(dev.device, p.compute_pipeline, nullptr), p.compute_pipeline=VK_NULL_HANDLE;
    if (p.layout) vkDestroyPipelineLayout(dev.device, p.layout, nullptr), p.layout=VK_NULL_HANDLE;
    if (p.set_layout) vkDestroyDescriptorSetLayout(dev.device, p.set_layout, nullptr), p.set_layout=VK_NULL_HANDLE;
    for (Stage& st : p.stages) if (st.module) vkDestroyShaderModule(dev.device, st.module, nullptr), st.module=VK_NULL_HANDLE;
}
void clean_map() {
    for (auto& [key, entry] : pipelines) { (void)key; if (entry->pipeline) vkDestroyPipeline(dev.device, entry->pipeline, nullptr); }
    pipelines.clear();
}
void unchanged(const Program& p) {
    CHECK(!p.layout && !p.set_layout && !p.push && p.push_stages == 0);
    CHECK(!p.uses_render_area && p.uses_push && p.binding_types.empty());
    CHECK(!p.failed && !p.ready && p.compute_state == 0);
}
void check_transaction() {
    auto p=program(true);
    failure=1; CHECK(!ensure_layout(*p)); unchanged(*p);
    CHECK(counts.sets == 1 && counts.layouts == 0 && counts.destroys == 0);
    failure=2; CHECK(!ensure_layout(*p)); unchanged(*p);
    CHECK(counts.layouts == 1 && counts.destroys == 1 && sets.empty());
    failure=0; CHECK(ensure_layout(*p));
    CHECK(p->binding_types.size() == 6 && p->layout && p->set_layout && p->push);
    CHECK(p->uses_render_area && p->uses_push && p->push_stages == VK_SHADER_STAGE_COMPUTE_BIT);
    const int calls=counts.sets; CHECK(ensure_layout(*p)); CHECK(counts.sets == calls);
    CHECK(counts.invalid == 0); cleanup(*p);
}
void check_success() {
    auto p=program(false); CHECK(ensure_layout(*p));
    CHECK(p->binding_types.size() == 7 && counts.sets == 1 && counts.layouts == 1 && counts.invalid == 0);
#ifdef HAVE_ORIGINAL_HELPER
    const auto fixed=arguments; cleanup(*p);
    CHECK(sets.empty() && layouts.empty() && modules.empty());
    arguments.clear(); next_handle=100;
    p=program(false); ensure_layout_original(*p);
    CHECK(arguments == fixed); CHECK(p->binding_types.size() == 7);
#endif
    cleanup(*p);
    // A fallback needs its buffer addresses even without render-area/global-memory push users.
    for (bool fallback : {false, true}) {
        auto make = [&] {
            auto q=program(false); q->fallback=fallback;
            for (auto& st : q->stages) st.info.uses_render_area=st.info.uses_global_memory=false;
            return q;
        };
        arguments.clear(); next_handle=100;
        p=make(); CHECK(ensure_layout(*p));
        CHECK(!p->uses_render_area && p->uses_push == fallback);
        CHECK(p->binding_types.size() == 7 && p->push);
#ifdef HAVE_ORIGINAL_HELPER
        const auto fallback_args=arguments; cleanup(*p);
        CHECK(sets.empty() && layouts.empty() && modules.empty());
        arguments.clear(); next_handle=100;
        p=make(); ensure_layout_original(*p);
        CHECK(arguments == fallback_args && p->uses_push == fallback);
#endif
        cleanup(*p);
    }
}
void check_draw_dispatch() {
    for (int fail : {1,2}) {
        failure=fail; auto p=program(true);
        call_draw(*p); call_dispatch(p.get());
        CHECK(after_guard == 0 && counts.graphics == 0 && counts.computes == 0);
        unchanged(*p); failure=0;
        call_draw(*p); call_dispatch(p.get());
        CHECK(after_guard == 2 && counts.graphics == 1 && p->compute_state == 0);
        after_guard=0; counts.graphics=0; cleanup(*p);
    }
}
void check_warm() {
    for (int fail : {1,2,0}) {
        failure=fail; auto p=program(true); const int before=counts.computes;
        call_warm(*p); CHECK(counts.computes == before + (fail == 0));
        CHECK(!p->layout && !p->set_layout && !p->failed && p->compute_state == 0);
        if (fail) CHECK(p->binding_types.empty());
        cleanup(*p);
    }
    CHECK(counts.libs == 1 && pipes.empty());
}
void check_seed() {
    factory=[] { return program(false); };
    for (int fail : {1,2,0}) {
        failure=fail; const int before=counts.graphics;
        call_seed(); CHECK(counts.graphics == before + (fail == 0 ? 2 : 0));
        CHECK(sets.empty() && layouts.empty() && modules.empty() && pipes.empty());
    }
    CHECK(counts.libs == 2 && counts.made == 3);
}
void check_cached() {
    for (bool compute : {false,true}) for (int fail : {1,2,0}) {
        failure=fail; auto p=program(compute); cached={p.get()};
        std::vector<std::unique_ptr<PipeNote>> notes;
        if (!compute) { auto n=std::make_unique<PipeNote>(); n->key=22; n->prog=p->key; n->variant=p->variant; notes.push_back(std::move(n)); }
        const int before=compute ? counts.computes : counts.graphics;
        call_precompile(std::move(notes));
        CHECK((compute ? counts.computes : counts.graphics) == before + (fail == 0));
        if (fail) unchanged(*p);
        CHECK(p->compute_state == 0 && !p->failed); clean_map(); cleanup(*p); cached.clear();
    }
}
void check_predict() {
    factory=[] { return program(false); };
    PipeNote note{}; note.variant=9;
    const std::vector<const PipeNote*> list{&note};
    for (int fail : {1,2,0}) {
        failure=fail; const int before=counts.graphics;
        call_predict(list); CHECK(counts.graphics == before + (fail == 0));
        CHECK(sets.empty() && layouts.empty() && modules.empty() && pipes.empty());
    }
    CHECK(counts.made == 1);
}
void check_logging() {
    for (int fail : {1,2}) for (int i=0; i<12; ++i) {
        failure=fail; auto p=program(true); CHECK(!ensure_layout(*p)); cleanup(*p);
    }
    CHECK(logs.size() == 18);
    CHECK(std::count_if(logs.begin(), logs.end(), [](const auto& s) {return s.find("(-1)") != s.npos;}) == 8);
    CHECK(std::count_if(logs.begin(), logs.end(), [](const auto& s) {return s.find("(-2)") != s.npos;}) == 8);
    CHECK(logs[8].find("further vkCreateDescriptorSetLayout failures") != std::string::npos);
    CHECK(logs[17].find("further vkCreatePipelineLayout failures") != std::string::npos);
}
} // namespace mkwvk
int main(int argc, char** argv) {
    using namespace mkwvk; CHECK(argc == 2); const std::string scenario=argv[1];
    if (scenario == "transaction") check_transaction();
    else if (scenario == "success") check_success();
    else if (scenario == "draw-dispatch") check_draw_dispatch();
    else if (scenario == "warm") check_warm();
    else if (scenario == "seed") check_seed();
    else if (scenario == "cached") check_cached();
    else if (scenario == "predict") check_predict();
    else if (scenario == "logging") check_logging();
#ifdef HAVE_ORIGINAL_HELPER
    else if (scenario == "original-negative") {
        auto p=program(true); failure=1; ensure_layout_original(*p);
        if (counts.layouts || counts.invalid) {
            std::fputs("original helper called a dependent API after failure\n", stderr); return 1;
        }
        CHECK(false);
    }
#endif
    else CHECK(false);
    CHECK(sets.empty() && layouts.empty() && modules.empty() && pipes.empty() && held_locks == 0);
    std::printf("PASS: %s (sets=%d, layouts=%d, compute=%d, graphics=%d)\n", scenario.c_str(), counts.sets, counts.layouts, counts.computes, counts.graphics);
}
