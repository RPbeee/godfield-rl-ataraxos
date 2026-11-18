#pragma once
#include <godfield/official_match.hpp>
#include <algorithm>
#include <array>
#include <random>
#include <memory>
#include <type_traits>

namespace gfrl {
namespace gf = godfield::official;
enum class Stage : std::int8_t { kind, card, target, alloc_hp, alloc_mp };
using Mask = std::array<bool, 21>;
static_assert(sizeof(Mask) == 21 * sizeof(bool), "numpy mask rows must be contiguous");
int bucket_value(int bucket);
// Policy restriction, using displayed faces just like other selection rules.
bool forbidden_pair(const gf::MatchSnapshot&, std::span<const gf::ItemToken>);
template<class M> gf::SelectionNode policy_extensions(const M& match, gf::CommandKind kind,
        std::span<const gf::ItemToken> prefix, bool canonical) {
    if (forbidden_pair(match.snapshot(), prefix)) return {};
    auto node = match.legal_extensions(kind, prefix, {canonical});
    std::vector<gf::ItemToken> extended(prefix.begin(), prefix.end());
    extended.push_back(0);
    std::erase_if(node.next, [&](auto token) {
        extended.back() = token;
        return forbidden_pair(match.snapshot(), extended);
    });
    return node;
}
struct SelectionExclusions {
    std::vector<gf::ItemToken> tokens;
    std::vector<int> kinds;
    bool enemy_default_self = false;
    // Evaluation removes branches from the original canonical policy tree.
    // The advisor may instead need equivalent, noncanonical item slots.
    bool preserve_canonical = false;
};
// A pure selection cursor over one immutable Match. Does not execute Commands.
class Selection {
public:
    Stage stage = Stage::kind;
    Mask mask{};
    gf::Command command;
    std::array<std::int8_t, 18> cards{};
    int actor_seat = 0, count = 0, substep = 0;
    bool complete = false;
    bool kind_chosen = false;
    SelectionExclusions exclusions;
    // Immutable public oracle for the first decision only; copied into worlds.
    std::shared_ptr<const gf::PublicSelection> public_root;
    explicit Selection(const gf::Match& match, bool canonical = true,
                       SelectionExclusions excluded = {},
                       std::shared_ptr<const gf::PublicSelection> public_check = {});
    explicit Selection(const gf::PublicSelection& match, bool canonical = true, SelectionExclusions excluded = {});
    void choose(const gf::Match& match, int choice, bool canonical);
    void choose(const gf::PublicSelection& match, int choice, bool canonical);
private:
    std::int64_t total_ = 0;
    template<class M> bool completion(const M&, gf::CommandKind, std::vector<gf::ItemToken>, bool) const;
    template<class M> void prune(const M&, bool);
    template<class M> void init(const M& match, bool canonical);
    template<class M> void refresh(const M& match, bool canonical);
    template<class M> void select(const M& match, int choice, bool canonical);
};
std::vector<std::int32_t> random_choices(std::span<const Mask> mask,
    std::span<const std::int8_t> stage, std::mt19937_64& rng);
}
