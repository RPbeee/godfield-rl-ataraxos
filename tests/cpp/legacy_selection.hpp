// Test-only snapshot of dac9be8; retain the recursive oracle for equivalence checks.
#pragma once
#include "gfrl_core/selection.hpp"
namespace gfrl {
class LegacySelection {
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
    explicit LegacySelection(const gf::Match& match, bool canonical = true,
                       SelectionExclusions excluded = {},
                       std::shared_ptr<const gf::PublicSelection> public_check = {});
    explicit LegacySelection(const gf::PublicSelection& match, bool canonical = true, SelectionExclusions excluded = {});
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
}
