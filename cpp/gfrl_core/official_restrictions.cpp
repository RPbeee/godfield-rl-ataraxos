#include "official_restrictions.hpp"

namespace gfrl {
namespace {
bool removes_branch(const gf::PublicSelection& position, const Selection& original,
                    const Selection& restricted, bool canonical) {
    if(original.complete || original.stage>=Stage::alloc_hp)return false;
    for(int c=0;c<21;++c)if(original.mask[c] && !restricted.mask[c])return true;
    const auto& x=restricted.exclusions;
    if(x.tokens.empty() && !x.enemy_default_self)return false; // Kinds differ only at the root.
    if(x.enemy_default_self && original.stage==Stage::card && !original.command.items.empty()) {
        const auto& own=position.snapshot().state.players[original.actor_seat];
        const auto& base=*std::find_if(own.items.begin(),own.items.end(),
            [&](const auto& h){return h.token==original.command.items.front();});
        if(!gf::targets_enemy_by_default(base.fake_model.value_or(base.model)))return false;
    }
    for(int c=0;c<21;++c)if(original.mask[c]) {
        if(x.enemy_default_self && original.stage==Stage::kind && c!=0 && c!=5 && c!=6)continue;
        auto a=original,b=restricted;
        a.choose(position,c,canonical);b.choose(position,c,canonical);
        if(removes_branch(position,a,b,canonical))return true;
    }
    return false;
}
bool changes_set(const gf::PublicSelection& position, bool canonical, const SelectionExclusions& x) {
    if(x.tokens.empty() && x.kinds.empty() && !x.enemy_default_self)return false;
    try {
        return removes_branch(position,Selection(position,canonical),Selection(position,canonical,x),canonical);
    } catch(const std::invalid_argument&) {
        return true; // The restriction removes every completion.
    }
}
bool has_completion(const gf::PublicSelection& position, bool canonical, const SelectionExclusions& x) {
    try {Selection selection(position,canonical,x);return true;}
    catch(const std::invalid_argument&) {return false;}
}
}
OfficialRestrictionPlan official_restriction_plan(const gf::MatchView& view, bool canonical,
                                                 const std::vector<int>& excluded_kinds) {
    const gf::PublicSelection position(view);
    OfficialRestrictionPlan plan;
    plan.counts[0]=1;
    std::array<SelectionExclusions,4> categories;
    for(auto& x:categories)x.preserve_canonical=true;
    categories[0].kinds=excluded_kinds;
    categories[3].enemy_default_self=true;
    std::vector<gf::CardId> disguised_faces;
    for(const auto& h:view.board.hand)if(h.disguised)disguised_faces.push_back(h.face);
    for(const auto& h:view.board.hand) {
        if(h.disguised)categories[1].tokens.push_back(h.token);
        else if(std::find(disguised_faces.begin(),disguised_faces.end(),h.face)!=disguised_faces.end())
            categories[2].tokens.push_back(h.token);
    }
    for(std::size_t i=0;i<categories.size();++i) {
        plan.counts[i+2]=changes_set(position,canonical,categories[i]);
        plan.counts[1]|=plan.counts[i+2];
    }
    auto& x=plan.exclusions;
    x.preserve_canonical=true;x.kinds=excluded_kinds;x.enemy_default_self=true;
    x.tokens=categories[1].tokens;
    x.tokens.insert(x.tokens.end(),categories[2].tokens.begin(),categories[2].tokens.end());
    if(!has_completion(position,canonical,x)) {
        plan.counts[6]=!x.tokens.empty();
        x.tokens.clear(); // Restore Dream and ambiguous items together, keeping hard gaps.
        if(!has_completion(position,canonical,x)) {
            plan.counts[7]=1;
            x={}; // Simulation must progress even when no verified action exists.
        }
    }
    return plan;
}
}
