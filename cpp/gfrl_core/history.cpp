#include "history.hpp"
#include <algorithm>
#include <cmath>
namespace gfrl {
void History::push(Token token, std::uint64_t decision) {
    card_overflow+=static_cast<std::uint64_t>(token.num[CountOverflow]);
    if(!capacity_) { ++dropped; return; }
    if(events_.size()==capacity_) {events_.pop_front(); ++dropped;}
    events_.push_back({std::move(token),decision});
}
void History::append(const gf::MatchView& b,const gf::MatchView& a,const gf::PublicDecision& d,gf::PlayerId viewer) {
    Token t; t.type=Decision; t.cat[Phase]=int(d.phase)+1; t.cat[Source]=int(d.source)+1;
    if(d.kind) t.cat[Kind]=int(*d.kind)+1;
    if(d.actor) t.cat[Actor]=relative(*d.actor,viewer);
    if(d.target) t.cat[Target]=relative(*d.target,viewer);
    cards(t,d.faces); t.num[FacesHidden]=d.faces_hidden;
    if(d.outcome_model) t.aux=card_code(*d.outcome_model);
    t.num[HasAllocation]=bool(d.allocation);
    if(d.allocation) {t.num[AllocationHp]=d.allocation->hp/99.f;t.num[AllocationMp]=d.allocation->mp/99.f;t.num[AllocationCp]=d.allocation->cp/99.f;}
    for(const auto& p:a.board.players) {
        auto it=std::find_if(b.board.players.begin(),b.board.players.end(),[&](const auto& q){return q.id==p.id;});
        if(it==b.board.players.end()) throw std::runtime_error("roster changed during decision");
        int k=Delta0+(relative(p.id,viewer)-1)*18;
        t.num[k+4]=int(p.dead)-int(it->dead); t.num[k+5]=int(p.fogged)-int(it->fogged);
        t.num[k+3]=!p.status || !it->status;
        if(p.status && it->status) {const auto& x=*it->status;const auto& y=*p.status;
            t.num[k]=(y.hp-x.hp)/99.f;t.num[k+1]=(y.mp-x.mp)/99.f;t.num[k+2]=(y.cp-x.cp)/99.f;
            t.num[k+6]=int(y.confused)-int(x.confused);t.num[k+7]=int(x.guardian)/10.f;t.num[k+8]=int(y.guardian)/10.f;
            for(auto c:x.curses) t.num[k+9+int(c)]-=1;
            for(auto c:y.curses) t.num[k+9+int(c)]+=1;
        }
    }
    push(t,a.decisions);
    // clear_events executes at a main launch (also external launches); response/purchase/grant retain the journal.
    const bool fresh=d.phase==gf::MatchPhase::main;
    auto base=[&](Type type){Token x;x.type=type;return x;};
    auto vectors=[&](const auto& before,const auto& after,auto emit) {
        std::size_t start=0;
        if(!fresh) {while(start<before.size() && start<after.size() && before[start]==after[start]) ++start;}
        for(;start<after.size();++start) emit(after[start]);
    };
    vectors(b.last_reveals,a.last_reveals,[&](const auto& e){auto x=base(Reveal);x.cat[Actor]=relative(e.owner,viewer);x.cards[0]=card_code(e.face);x.aux=card_code(e.model);push(x,a.decisions);});
    vectors(b.last_guardian_changes,a.last_guardian_changes,[&](const auto& e){auto x=base(GuardianChange);x.cat[Actor]=relative(e.player,viewer);x.cat[GuardianBefore]=int(e.before)+1;x.cat[GuardianAfter]=int(e.after)+1;x.cat[Reason]=int(e.reason)+1;push(x,a.decisions);});
    vectors(b.last_redraws,a.last_redraws,[&](auto e){auto x=base(Redraw);x.cat[Actor]=relative(e,viewer);push(x,a.decisions);});
    if(a.last_removal && (fresh || a.last_removal!=b.last_removal)) {const auto& e=*a.last_removal;auto x=base(Removal);x.cat[Origin]=relative(e.origin,viewer);x.cat[SourceSeat]=relative(e.source,viewer);x.cat[Target]=relative(e.target,viewer);x.aux=card_code(e.model);cards(x,e.removed_models);push(x,a.decisions);}
    if(a.last_sale && (fresh || a.last_sale!=b.last_sale)) {const auto& e=*a.last_sale;auto x=base(Sale);x.cat[Origin]=relative(e.origin,viewer);x.cat[Actor]=relative(e.source,viewer);x.cat[Target]=relative(e.target,viewer);x.aux=card_code(e.model);if(e.overflow_model)x.cards[0]=card_code(*e.overflow_model);x.num[Price]=e.price/99.f;push(x,a.decisions);}
    if(a.last_purchase && (fresh || a.last_purchase!=b.last_purchase)) {const auto& e=*a.last_purchase;auto x=base(Purchase);x.cat[Origin]=relative(e.origin,viewer);x.cat[Actor]=relative(e.buyer,viewer);x.cat[Target]=relative(e.seller,viewer);x.cat[Reason]=int(e.reason)+1;x.aux=card_code(e.model);if(e.overflow_model)x.cards[0]=card_code(*e.overflow_model);x.num[Price]=e.price/99.f;push(x,a.decisions);}
    if(a.last_danger && (fresh || a.last_danger!=b.last_danger)) {const auto& e=*a.last_danger;auto x=base(Danger);x.cat[Origin]=relative(e.origin,viewer);x.cat[Target]=relative(e.target,viewer);cards(x,e.models);x.num[Power]=std::log1p(double(e.power));x.num[Mortar]=e.mortar;if(e.element)x.cat[Element]=int(*e.element)+1;push(x,a.decisions);}
    auto devil=[&](const auto& e,Type type){auto x=base(type);x.cat[Target]=relative(e.target,viewer);x.aux=card_code(e.model);cards(x,e.removed_models);push(x,a.decisions);};
    if(a.last_devil && (fresh || a.last_devil!=b.last_devil)) devil(*a.last_devil,Devil);
    if(a.last_grant && (fresh || a.last_grant!=b.last_grant || d.source==gf::DecisionSource::grant || d.source==gf::DecisionSource::artifact_grant)) devil(*a.last_grant,Grant);
    if(a.last_fate && (fresh || a.last_fate!=b.last_fate)) {auto x=base(Fate);x.cat[Actor]=relative(a.last_fate->actor,viewer);x.aux=card_code(a.last_fate->phenomenon);push(x,a.decisions);}
    if(a.last_phenomenon && (fresh || a.last_phenomenon!=b.last_phenomenon)) {auto x=base(Phenomenon);x.aux=card_code(a.last_phenomenon->model);if(a.last_phenomenon->recipient)x.cat[Target]=relative(*a.last_phenomenon->recipient,viewer);push(x,a.decisions);}
}
}
