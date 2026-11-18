#include "official_restrictions.hpp"

namespace gfrl {
namespace {
bool has_completion(const gf::PublicSelection& position, bool canonical, const SelectionExclusions& x) {
    try {Selection selection(position,canonical,x);return true;}
    catch(const std::invalid_argument&) {return false;}
}

// Category diagnostics need existence witnesses, not command enumeration.
// Every selectable face has a witness of at most three cards: a base, an
// optional enabler (FREE, a miracle before FREE, or a defense filter), and the
// item. Removing unrelated material only reduces cost/canonical constraints.
// Categories include ALL equivalent slots of a face/disguise class, so the
// first canonical duplicate suffices. Self targets need only base + repair
// (or sale commodity); adding material can only remove explicit targets.
// Inspect root/one-card/two-card nodes once for all active categories. This is
// O(k*h^2) oracle queries, independent of the size of the full command tree.
void category_witnesses(const gf::PublicSelection& position, bool canonical,
                       const std::array<SelectionExclusions,4>& categories,
                       RestrictionCounts& counts) {
    const auto kinds=position.legal_kinds();
    for(auto k:kinds)if(std::find(categories[0].kinds.begin(),categories[0].kinds.end(),int(k))!=categories[0].kinds.end())
        counts[2]=1;
    const auto& snap=position.snapshot();
    const auto& own=*std::find_if(snap.state.players.begin(),snap.state.players.end(),
        [&](const auto& p){return p.id==*snap.actor;});
    auto needed=[&] {
        return (!counts[3] && !categories[1].tokens.empty())
            || (!counts[4] && !categories[2].tokens.empty())
            || (!counts[5] && categories[3].enemy_default_self);
    };
    std::vector<gf::ItemToken> prefix;
    auto visit=[&](auto&& self,gf::CommandKind kind)->void {
        if(!needed())return;
        const auto node=policy_extensions(position,kind,prefix,canonical);
        for(int i=1;i<=2;++i)if(!counts[i+2])
            for(auto token:node.next)if(std::find(categories[i].tokens.begin(),categories[i].tokens.end(),token)!=categories[i].tokens.end()) {
                counts[i+2]=1;break;
            }
        if(!counts[5] && categories[3].enemy_default_self && !prefix.empty()
                && (kind==gf::CommandKind::play || kind==gf::CommandKind::sell || kind==gf::CommandKind::buy)
                && node.can_finish && std::find(node.targets.begin(),node.targets.end(),snap.actor)!=node.targets.end()) {
            const auto base=std::find_if(own.items.begin(),own.items.end(),[&](const auto& h){return h.token==prefix.front();});
            counts[5]=gf::targets_enemy_by_default(base->fake_model.value_or(base->model));
        }
        if(prefix.size()==2)return;
        for(auto token:node.next) {
            prefix.push_back(token);self(self,kind);prefix.pop_back();
            if(!needed())break;
        }
    };
    for(auto k:kinds)visit(visit,k);
    for(int i=2;i<=5;++i)counts[1]|=counts[i];
}
}
OfficialRestrictionPlan official_restriction_plan(const gf::MatchView& view, bool canonical,
                                                 const std::vector<int>& excluded_kinds,
                                                 OfficialRestrictionCategories active) {
    const gf::PublicSelection position(view);
    OfficialRestrictionPlan plan;
    plan.counts[0]=1;
    std::array<SelectionExclusions,4> categories;
    if(active.kind)categories[0].kinds=excluded_kinds;
    categories[3].enemy_default_self=active.self_target;
    std::vector<gf::CardId> disguised_faces;
    if(active.dream) {
        for(const auto& h:view.board.hand)if(h.disguised)disguised_faces.push_back(h.face);
        for(const auto& h:view.board.hand) {
            if(h.disguised)categories[1].tokens.push_back(h.token);
            else if(std::find(disguised_faces.begin(),disguised_faces.end(),h.face)!=disguised_faces.end())
                categories[2].tokens.push_back(h.token);
        }
    }
    category_witnesses(position,canonical,categories,plan.counts);
    auto& x=plan.exclusions;
    x.preserve_canonical=true;x.kinds=categories[0].kinds;x.enemy_default_self=active.self_target;
    x.tokens=categories[1].tokens;
    x.tokens.insert(x.tokens.end(),categories[2].tokens.begin(),categories[2].tokens.end());
    if(!has_completion(position,canonical,x)) {
        plan.counts[6]=!x.tokens.empty();
        x.tokens.clear(); // Restore Dream and ambiguous items together, keeping active hard gaps.
        if(!has_completion(position,canonical,x)) {
            plan.counts[7]=1;
            x={}; // Simulation must progress even when no verified action exists.
        }
    }
    return plan;
}
}
