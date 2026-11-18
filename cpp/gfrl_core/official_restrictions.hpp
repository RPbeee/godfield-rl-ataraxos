#pragma once
#include "selection.hpp"

namespace gfrl {
// Per-decision indicators, aggregated over A only. Categories independently
// compare their legal set to the unrestricted tree, before any relaxation.
using RestrictionCounts = std::array<std::uint64_t, 8>;
struct OfficialRestrictionPlan {
    SelectionExclusions exclusions;
    RestrictionCounts counts{}; // decisions, restricted, kind, dream, ambiguous,
                               // self_target, dream_last_resort, unrestricted_last_resort
};
struct OfficialRestrictionCategories {
    bool kind = true, dream = true, self_target = true;
};
// Only the viewer's public hand/faces and public legality enter this boundary.
OfficialRestrictionPlan official_restriction_plan(const gf::MatchView&, bool canonical,
                                                 const std::vector<int>& excluded_kinds,
                                                 OfficialRestrictionCategories active = {});
}
