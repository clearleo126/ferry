// 真实模型 miss trace 读取器（L1；对应 实验设计.txt【19】/19.6 "仍缺：真实 trace"）
//
// 输入：tools/capture_miss_trace.py 产出的 .bin（格式见该脚本头部注释）。
// 语义：page = 16 token，block = 4 token；每层每步选 sel_pages 个 page，
//       展开成 sel_blocks 个 4-token block。
// id：抓取端用**链式内容哈希**（同组共享前缀 → 同 id；不同上下文不误合并）。
//     这里按首次出现顺序重映射为稠密 id（全局一致，运行时与反事实逐位相同），
//     并设 spec.slab_blocks = 单请求最大 block 数（slab 只装载荷，按此取模寻址）。
//
// 纪律：所有参数必须与 MissSpec 逐项一致，否则直接 abort——宁可失败，
//       不可静默退回合成流（那会把真实 trace 的结果变成假阴性）。
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "workloads/missstream.cuh"

namespace ferry {

struct MissTraceInfo {
  std::string path;
  int requests = 0, steps = 0, layers = 0;
  int page_tokens = 0, block_tokens = 0, sel_pages = 0;
  int kv_heads = 0, head_dim = 0;
  uint64_t unique_blocks = 0;
  uint64_t max_blocks_per_req = 0;
};

// 环境变量入口：FERRY_MISS_TRACE=<path.bin>
inline bool ferry_miss_trace_path(std::string* out) {
  const char* p = std::getenv("FERRY_MISS_TRACE");
  if (!p || !*p) return false;
  *out = p;
  return true;
}

inline uint64_t fnv1a64_file(const std::string& path, uint64_t* size_out = nullptr) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return 0;
  uint64_t h = 1469598103934665603ull;
  uint64_t n = 0;
  std::vector<char> buf(1 << 20);
  size_t r;
  while ((r = std::fread(buf.data(), 1, buf.size(), f)) > 0) {
    for (size_t i = 0; i < r; ++i) {
      h ^= (unsigned char)buf[i];
      h *= 1099511628211ull;
    }
    n += r;
  }
  std::fclose(f);
  if (size_out) *size_out = n;
  return h;
}

[[noreturn]] inline void miss_trace_die(const std::string& path, const char* why) {
  std::fprintf(stderr,
               "[miss_trace] %s: %s\n"
               "[miss_trace] 真实 trace 参数与本次运行不一致，或文件损坏；"
               "拒绝退回合成流。\n",
               path.c_str(), why);
  std::exit(1);
}

// 读 trace → MissTrace（事件顺序 (r, s, l, k)，与 synthetic 一致）。
// spec 会被写入 slab_blocks（真实 trace 专用），其余字段只做校验。
inline MissTrace load_miss_trace(const std::string& path, MissSpec& spec,
                                 MissTraceInfo* info = nullptr) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) miss_trace_die(path, "无法打开");
  char magic[4];
  uint32_t ver = 0, R = 0, S = 0, L = 0, PT = 0, BT = 0, KV = 0, D = 0, SEL = 0;
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "FMT1", 4) != 0)
    miss_trace_die(path, "magic 不是 FMT1");
  auto rd_u32 = [&](uint32_t* v) {
    if (std::fread(v, 4, 1, f) != 1) miss_trace_die(path, "头部截断");
  };
  rd_u32(&ver); rd_u32(&R); rd_u32(&S); rd_u32(&L);
  rd_u32(&PT); rd_u32(&BT); rd_u32(&KV); rd_u32(&D); rd_u32(&SEL);
  if (ver != 1) miss_trace_die(path, "版本不是 1");

  char why[256];
  auto expect = [&](bool ok, const char* what) {
    if (!ok) miss_trace_die(path, what);
  };
  expect((int)R == spec.num_requests, "num_requests 不一致");
  expect((int)S == spec.steps, "steps 不一致");
  expect((int)L == spec.num_layers, "layers 不一致");
  expect((int)KV == spec.kv_heads, "kv_heads 不一致（c4 字节数会错）");
  expect((int)D == spec.head_dim, "head_dim 不一致（c4 字节数会错）");
  expect((int)BT == 4, "block_tokens != 4（本仿真器 c4 语义固定 4 token）");
  expect(PT % BT == 0, "page_tokens 不是 block_tokens 的整数倍");
  const int blocks_per_page = (int)(PT / BT);
  expect((int)SEL * blocks_per_page == spec.sel_blocks,
         "sel_pages×blocks_per_page != sel_blocks（检查 --sel/--hot 与 trace 头）");

  const size_t total_ev = (size_t)R * S * L * (size_t)spec.sel_blocks;
  MissTrace tr;
  tr.events.reserve(total_ev);

  // 与 replay_miss_stream 完全相同的 per-(req, layer) LRU 语义
  std::vector<std::deque<uint64_t>> dq((size_t)R * L);
  std::vector<std::unordered_set<uint64_t>> st((size_t)R * L);
  std::unordered_map<uint32_t, uint64_t> idmap;
  std::vector<std::unordered_set<uint32_t>> per_req_raw(R);

  std::vector<uint32_t> ids(spec.sel_blocks);
  uint64_t max_per_req = 0;
  for (uint32_t r = 0; r < R; ++r) {
    for (uint32_t s = 0; s < S; ++s) {
      for (uint32_t l = 0; l < L; ++l) {
        uint16_t cnt = 0;
        if (std::fread(&cnt, 2, 1, f) != 1)
          miss_trace_die(path, "payload 截断（事件计数）");
        if ((int)cnt != spec.sel_blocks) {
          std::snprintf(why, sizeof why,
                        "事件 (r=%u,s=%u,l=%u) 选中 %u block，期望 %d"
                        "（sel_pages×4）。尾部不完整 page 会造成此错。",
                        r, s, l, (unsigned)cnt, spec.sel_blocks);
          miss_trace_die(path, why);
        }
        if (std::fread(ids.data(), 4, cnt, f) != cnt)
          miss_trace_die(path, "payload 截断（block id）");

        auto& d = dq[(size_t)r * L + l];
        auto& set = st[(size_t)r * L + l];
        for (int k = 0; k < (int)cnt; ++k) {
          const uint32_t raw = ids[k];
          per_req_raw[r].insert(raw);
          auto it = idmap.find(raw);
          uint64_t blk;
          if (it == idmap.end()) {
            blk = idmap.size();
            idmap.emplace(raw, blk);
          } else {
            blk = it->second;
          }
          ++tr.total_selects;
          const bool hit = set.count(blk) > 0;
          tr.events.push_back({(int)r, (int)l, (int)s, blk, !hit});
          if (hit) {
            for (auto dit = d.begin(); dit != d.end(); ++dit) {
              if (*dit == blk) { d.erase(dit); break; }
            }
            d.push_back(blk);
          } else {
            ++tr.total_misses;
            tr.miss_bytes += spec.c4_bytes();
            if ((int)d.size() >= spec.hot_blocks) {
              set.erase(d.front());
              d.pop_front();
            }
            d.push_back(blk);
            set.insert(blk);
          }
        }
      }
    }
    if (per_req_raw[r].size() > max_per_req) max_per_req = per_req_raw[r].size();
  }
  std::fclose(f);

  tr.hit_rate = tr.total_selects
                    ? 1.0 - (double)tr.total_misses / (double)tr.total_selects
                    : 0.0;
  spec.slab_blocks = (int)max_per_req;  // slab 容量 = 单请求最大 block 数
  if (info) {
    info->path = path;
    info->requests = (int)R; info->steps = (int)S; info->layers = (int)L;
    info->page_tokens = (int)PT; info->block_tokens = (int)BT;
    info->sel_pages = (int)SEL;
    info->kv_heads = (int)KV; info->head_dim = (int)D;
    info->unique_blocks = idmap.size();
    info->max_blocks_per_req = max_per_req;
  }
  return tr;
}

}  // namespace ferry
