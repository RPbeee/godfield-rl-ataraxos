#pragma once
#include "selection.hpp"
#include <string_view>

namespace gfrl {
enum class Controller { external, random, proposal, greedy };
Controller controller(std::string_view name);
struct Random { static gf::Command act(const gf::Match&, gf::PlayerId, std::uint64_t&); };
struct Proposal { static gf::Command act(const gf::Match&, gf::PlayerId, std::uint64_t&); };
struct Greedy { static gf::Command act(const gf::Match&, gf::PlayerId, std::uint64_t&); };
gf::Command bot_action(Controller, const gf::Match&, gf::PlayerId, std::uint64_t&);
std::uint64_t mix_seed(std::uint64_t);
struct EvalResult {
    std::uint64_t g = 0, pair = 0;
    int agent_seat = 0, result = 0;
    bool truncated = false;
    std::uint64_t decisions = 0;
    std::array<std::int64_t, 2> hp{};
    bool operator==(const EvalResult&) const = default;
};
// Shared identical paired initialization; odd games switch the agent's seat.
void reset_evaluation(gf::Match&, std::uint64_t eval_seed, std::uint64_t g,
                      std::uint64_t& chance_rng, std::array<std::uint64_t,2>& policy_rng);
EvalResult evaluation_result(const gf::Match&, std::uint64_t g);
std::vector<EvalResult> evaluate_bots(Controller a, Controller b, std::size_t pairs,
                                     std::size_t threads, std::uint64_t eval_seed);
}
