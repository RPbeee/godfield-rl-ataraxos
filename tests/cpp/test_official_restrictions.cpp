#include "gfrl_core/official_restrictions.hpp"
#include <iostream>
#include <set>
#include <sstream>

using namespace gfrl;
void check(bool value, const char* message) {if(!value)throw std::runtime_error(message);}
gf::Match scene(std::vector<gf::ItemInstance> hand, bool apocalypse=false) {
    gf::Match match;
    auto state=match.snapshot().state;
    for(auto& p:state.players) {p.items.clear();p.curses.clear();p.guardian=gf::Guardian::none;}
    state.players[0].id=42;state.players[1].id=77;
    state.players[0].items=std::move(hand);
    if(std::any_of(state.players[0].items.begin(),state.players[0].items.end(),
        [](const auto& h){return h.fake_model.has_value();}))state.players[0].curses={gf::Curse::dream};
    if(apocalypse) {state.gf=10;state.tiebreak_gf=10;}
    match.load_scene(state,42,123);
    return match;
}
std::string key(const gf::Command& c) {
    std::ostringstream out;out<<int(c.kind)<<':';
    for(auto token:c.items)out<<token<<',';
    out<<':';if(c.target)out<<*c.target;
    if(c.allocation)out<<':'<<c.allocation->hp<<','<<c.allocation->mp<<','<<c.allocation->cp;
    return out.str();
}
template<class M> void leaves(const M& position, Selection selection, std::vector<gf::Command>& out) {
    if(selection.complete) {out.push_back(selection.command);return;}
    for(int c=0;c<21;++c)if(selection.mask[c]) {
        auto next=selection;next.choose(position,c,true);leaves(position,next,out);
    }
}
std::vector<gf::Command> commands(const gf::MatchView& view, SelectionExclusions x={}) {
    gf::PublicSelection position(view);std::vector<gf::Command> out;
    leaves(position,Selection(position,true,x),out);return out;
}
std::vector<gf::Command> commands(const gf::Match& match, SelectionExclusions x) {
    std::vector<gf::Command> out;leaves(match,Selection(match,true,x),out);
    for(const auto& c:out)check(match.legal(c),"restricted simulator completion must be legal");
    return out;
}
bool unsupported(const gf::Command& c, const gf::MatchView& view) {
    if(c.kind==gf::CommandKind::discard || c.kind==gf::CommandKind::sacrifice)return true;
    for(auto token:c.items)if(token==100 || token==101)return true;
    if(!c.items.empty() && c.target==42 && (c.kind==gf::CommandKind::play
        || c.kind==gf::CommandKind::sell || c.kind==gf::CommandKind::buy)) {
        for(const auto& h:view.board.hand)if(h.token==c.items.front())
            return gf::targets_enemy_by_default(h.face);
    }
    return false;
}
void categories() {
    for(bool apocalypse:{false,true}) {
        const auto match=scene({{100,7,6,false},{101,6,{},false},{102,7,{},false},
            {103,198,{},false},{104,4,{},false},{105,5,{},false},{106,3,{},false},
            {107,130,{},false},{108,7,{},false}},apocalypse);
        const auto view=match.view(42);
        const auto plan=official_restriction_plan(view,true,{2,9});
        check(plan.counts==RestrictionCounts{1,1,1,1,1,1,0,0},"category counts");
        std::set<std::string> expected,actual;
        const auto original=commands(view);
        for(const auto& c:original)if(!unsupported(c,view))expected.insert(key(c));
        for(const auto& c:commands(view,plan.exclusions))actual.insert(key(c));
        check(!actual.empty() && actual==expected,"restrictions must remove exactly unsupported completions");
        std::set<std::string> simulated;
        for(const auto& c:commands(match,plan.exclusions))simulated.insert(key(c));
        check(simulated==actual,"simulator and public restriction trees must agree");
        check(actual.contains("0:102,:77"),"ordinary enemy attack survives");
        check(actual.contains("0:103,:42"),"self-default card keeps self target");
        check(actual.contains("6:105,:77"),"buy keeps enemy target");
        check(!actual.contains("6:105,:42"),"buy loses self target");
        // Canonical duplicate slots must not appear when unrelated cards are excluded.
        check(!actual.contains("0:108,:77"),"restriction must not add noncanonical completions");
    }
}
void relaxations() {
    for(bool ambiguous:{false,true}) {
        auto hand=std::vector<gf::ItemInstance>{{100,7,6,false}};
        if(ambiguous)hand.push_back({101,6,{},false});
        const auto view=scene(hand).view(42);
        const auto plan=official_restriction_plan(view,true,{2,9});
        check(plan.counts[6]==1 && plan.counts[7]==0,"restore Dream before hard restrictions");
        for(const auto& c:commands(view,plan.exclusions)) {
            check(c.kind==gf::CommandKind::play,"Dream recovery must not restore discard");
            check(c.target==77,"Dream recovery must not restore self target");
        }
    }
    const auto view=scene({{100,6,{},false}}).view(42);
    const auto plan=official_restriction_plan(view,true,{0,2,9});
    check(plan.counts[6]==0 && plan.counts[7]==1,"hard-only exhaustion uses unrestricted fallback");
    check(commands(view,plan.exclusions).size()==commands(view).size(),"fallback restores original tree");
    const auto dream_view=scene({{100,7,6,false}}).view(42);
    const auto both=official_restriction_plan(dream_view,true,{0,2,9});
    check(both.counts[6]==1 && both.counts[7]==1,"both relaxation steps recorded");
    const auto shield=scene({{100,130,130,false}}).view(42);
    const auto safe=official_restriction_plan(shield,true,{2,9});
    check(!safe.counts[6] && !safe.counts[7],"cardless prayer avoids Dream relaxation");
    for(const auto& c:commands(shield,safe.exclusions))
        check(c.kind==gf::CommandKind::pray && c.items.empty(),"safe cardless move survives");
}
void public_only() {
    const auto a=scene({{100,7,6,false},{101,6,{},false},{102,7,{},false}}).view(42);
    const auto b=scene({{100,8,6,false},{101,6,{},false},{102,7,{},false}}).view(42);
    const auto x=official_restriction_plan(a,true,{2,9}),y=official_restriction_plan(b,true,{2,9});
    check(x.counts==y.counts && x.exclusions.tokens==y.exclusions.tokens,"hidden Dream identity changed restrictions");
    std::set<std::string> ax,bx;
    for(const auto& c:commands(a,x.exclusions))ax.insert(key(c));
    for(const auto& c:commands(b,y.exclusions))bx.insert(key(c));
    check(ax==bx,"hidden Dream identity changed allowed completions");
    const auto used=scene({}).view(42);
    const auto plan=official_restriction_plan(used,true,{2,9});
    check(plan.counts==RestrictionCounts{1,0,0,0,0,0,0,0},"no legal restricted completion means no count");
    // Dream and ambiguous weapons are visible, but cannot be selected while
    // defending. Their mere presence must not increment either category.
    auto state=scene({{100,7,6,false},{101,6,{},false},{102,120,{},false}}).snapshot().state;
    state.players[1].items={{200,6,{},false}};
    gf::Match response;response.load_scene(state,77,123);
    response.step({77,gf::CommandKind::play,{200},42});
    const auto defending=official_restriction_plan(response.view(42),true,{2,9});
    check(defending.counts==RestrictionCounts{1,0,0,0,0,0,0,0},"unselectable Dream/ambiguous cards must not count");
}
int main() {
    try {categories();relaxations();public_only();}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
