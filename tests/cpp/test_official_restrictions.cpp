#include "gfrl_core/official_restrictions.hpp"
#include "gfrl_core/bots.hpp"
#include "legacy_selection.hpp"
#include <chrono>
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
// The legacy declaration is in gfrl, alongside the production function.
namespace gfrl { OfficialRestrictionPlan legacy_restriction_plan(const gf::MatchView&,bool,const std::vector<int>&); }

bool same(const SelectionExclusions& a,const SelectionExclusions& b) {
    return a.tokens==b.tokens && a.kinds==b.kinds && a.enemy_default_self==b.enemy_default_self
        && a.preserve_canonical==b.preserve_canonical;
}
template<class M> void compare_path(const M& position,bool canonical,SelectionExclusions x,std::mt19937_64& rng) {
    std::optional<Selection> current;
    std::optional<LegacySelection> old;
    try {current.emplace(position,canonical,x);}catch(const std::invalid_argument&) {}
    try {old.emplace(position,canonical,x);}catch(const std::invalid_argument&) {}
    check(bool(current)==bool(old),"legacy construction exhaustion differs");
    if(!current)return;
    for(int steps=0;!current->complete;++steps) {
        check(steps<24,"selection did not terminate");
        check(current->mask==old->mask && current->stage==old->stage,"legacy mask differs");
        std::vector<int> choices;
        for(int i=0;i<21;++i)if(current->mask[i])choices.push_back(i);
        const int choice=choices[rng()%choices.size()];
        current->choose(position,choice,canonical);old->choose(position,choice,canonical);
    }
    check(key(current->command)==key(old->command),"legacy command differs");
}
void compare_position(const gf::Match& match,std::mt19937_64& rng) {
    const auto view=match.view(*match.snapshot().actor);
    const gf::PublicSelection position(view);
    for(bool canonical:{true,false}) {
        const auto a=official_restriction_plan(view,canonical,{2,9});
        const auto b=gfrl::legacy_restriction_plan(view,canonical,{2,9});
        check(a.counts==b.counts && same(a.exclusions,b.exclusions),"legacy plan differs");
        compare_path(position,canonical,a.exclusions,rng);
        compare_path(match,canonical,a.exclusions,rng);
        // OfficialDecision uses arbitrary token exclusions and reopens equivalent
        // noncanonical slots. Exercise it separately from evaluation's categories.
        SelectionExclusions bot;
        bot.enemy_default_self=true;
        for(const auto& h:view.board.hand)if(rng()%2)bot.tokens.push_back(h.token);
        compare_path(position,canonical,bot,rng);
        compare_path(match,canonical,bot,rng);
    }
}

template<class M> void compare_masks(const M& position,Selection current,LegacySelection old,bool canonical) {
    check(current.mask==old.mask && current.complete==old.complete && current.stage==old.stage,
          "exhaustive legacy mask differs");
    if(current.complete) {
        check(key(current.command)==key(old.command),"exhaustive legacy command differs");return;
    }
    for(int i=0;i<21;++i)if(current.mask[i]) {
        auto a=current;auto b=old;
        a.choose(position,i,canonical);b.choose(position,i,canonical);
        compare_masks(position,std::move(a),std::move(b),canonical);
    }
}
void repair_regressions() {
    // Ordered FREE repair, filter -> reaction -> FREE, duplicated FREE slots,
    // and exchange allocations. Enumerate EVERY mask in these small trees.
    for(bool response:{false,true})for(bool canonical:{false,true}) {
        auto state=scene({{100,6,{},false},{101,46,{},false},{102,204,{},false},
            {103,189,{},false},{104,233,{},false},{105,3,{},false},{106,218,{},false}}).snapshot().state;
        state.players[0].mp=0;
        state.players[1].items={{200,59,{},false}};
        gf::Match match;match.load_scene(state,response?77:42,123);
        if(response)match.step({77,gf::CommandKind::play,{200},42});
        const gf::PublicSelection position(match.view(42));
        for(bool preserve:{true,false})for(auto excluded:{gf::ItemToken(101),gf::ItemToken(102),gf::ItemToken(103)}) {
            SelectionExclusions x;x.tokens={excluded};x.enemy_default_self=true;x.preserve_canonical=preserve;
            compare_masks(position,Selection(position,canonical,x),LegacySelection(position,canonical,x),canonical);
            compare_masks(match,Selection(match,canonical,x),LegacySelection(match,canonical,x),canonical);
        }
    }
}

void subsets() {
    const auto view=scene({{100,7,6,false},{101,6,{},false},{102,7,{},false},{103,4,{},false},{104,3,{},false}}).view(42);
    for(int bits=0;bits<8;++bits) {
        const OfficialRestrictionCategories active{bool(bits&1),bool(bits&2),bool(bits&4)};
        const auto plan=official_restriction_plan(view,true,{2,9},active);
        check(bool(plan.counts[2])==active.kind,"inactive kind counted");
        check(bool(plan.counts[3])==active.dream && bool(plan.counts[4])==active.dream,"inactive Dream counted");
        check(bool(plan.counts[5])==active.self_target,"inactive self target counted");
        check(plan.exclusions.kinds.empty()!=active.kind,"subset kind exclusions");
        check(plan.exclusions.tokens.empty()!=active.dream,"subset Dream exclusions");
        check(plan.exclusions.enemy_default_self==active.self_target,"subset self exclusions");
    }
    const auto dream=scene({{100,7,6,false}}).view(42);
    const auto only_dream=official_restriction_plan(dream,true,{2,9},{false,true,false});
    check(only_dream.counts[6] && !only_dream.counts[7] && only_dream.exclusions.kinds.empty()
        && !only_dream.exclusions.enemy_default_self,"Dream-only fallback must restore tokens without adding hard gaps");
    const auto exhausted=official_restriction_plan(dream,true,{0,2,9},{true,false,false});
    check(!exhausted.counts[6] && exhausted.counts[7],"inactive Dream must not be relaxed");
}

struct Timings {
    double plan=0,selection=0;
    int positions=0,large=0,dream=0,response=0;
    template<class F> void measure(double& maximum,F f) {
        const auto start=std::chrono::steady_clock::now();f();
        const auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        maximum=std::max(maximum,elapsed);
        check(elapsed<100,"restriction/selection exceeded 100ms per-call budget");
    }
    void inspect(const gf::Match& match,std::mt19937_64& rng) {
        const auto view=match.view(*match.snapshot().actor);
        const gf::PublicSelection position(view);
        ++positions;large+=view.board.hand.size()>=16;
        dream+=std::any_of(view.board.hand.begin(),view.board.hand.end(),[](const auto& h){return h.disguised;});
        response+=view.phase==gf::MatchPhase::response;
        OfficialRestrictionPlan plan;
        measure(this->plan,[&]{plan=official_restriction_plan(view,true,{2,9});});
        std::optional<Selection> selection;
        measure(this->selection,[&]{selection.emplace(position,true,plan.exclusions);});
        // Time all pruning stages, including allocation, through completion.
        while(!selection->complete) {
            std::vector<int> choices;
            for(int i=0;i<21;++i)if(selection->mask[i])choices.push_back(i);
            measure(this->selection,[&]{selection->choose(position,choices[rng()%choices.size()],true);});
        }
        SelectionExclusions bot=plan.exclusions;bot.preserve_canonical=false;
        for(const auto& h:view.board.hand)if(rng()%3==0)bot.tokens.push_back(h.token);
        measure(this->selection,[&]{
            try {Selection cursor(position,true,bot);}catch(const std::invalid_argument&) {}
            try {Selection cursor(match,true,bot,std::make_shared<gf::PublicSelection>(position));}
            catch(const std::invalid_argument&) {}
        });
    }
};

void regression_and_stress() {
    std::mt19937_64 rng(0xdecaf);
    Timings timings;
    int regression=0;
    // Random and greedy games provide main, response, purchase and allocation
    // decisions. Keep the exponential legacy oracle on ordinary small hands.
    for(int game=0;timings.positions<5000 || regression<1000;++game) {
        gf::Match match;
        std::uint64_t chance;std::array<std::uint64_t,2> policy;
        reset_evaluation(match,11,game,chance,policy);
        while(!match.done()) {
            const auto& snap=match.snapshot();
            if(snap.phase==gf::MatchPhase::grant) {match.grant_step(match.sample_grant(chance));continue;}
            const auto actor=*snap.actor;
            const auto own=std::find_if(snap.state.players.begin(),snap.state.players.end(),[&](const auto& p){return p.id==actor;});
            if(snap.phase==gf::MatchPhase::main && own->confusion_count) {match.confusion_step(match.sample_confusion(chance));continue;}
            timings.inspect(match,rng);
            if(own->items.size()<=5 && regression<1000) {compare_position(match,rng);++regression;}
            const auto c=(game%2?Greedy::act(match,actor,policy[0]):Random::act(match,actor,policy[0]));
            match.step(c);
        }
    }
    // Full hands with Dream, FREE, plus attacks, filters, reactions and exchange.
    const std::array<int,27> pool{3,4,5,6,9,12,46,94,113,114,120,130,143,146,153,163,165,180,189,193,198,204,218,228,232,233,236};
    for(int i=0;i<1000;++i) {
        std::vector<gf::ItemInstance> hand;
        const int n=i<300?5:18;
        for(int j=0;j<n;++j) {
            const auto face=gf::CardId(pool[rng()%pool.size()]);
            hand.push_back({gf::ItemToken(100+j),face,j%5==0?std::optional<gf::CardId>(face):std::nullopt,false});
        }
        auto match=scene(hand,i%3==0);
        auto state=match.snapshot().state;state.players[0].mp=i%3?0:99;
        match.load_scene(state,42,i+123);
        if(i%2) {
            state.players[1].items={{200,6,{},false}};
            match.load_scene(state,77,i+123);match.step({77,gf::CommandKind::play,{200},42});
        }
        timings.inspect(match,rng);
        if(n==5) {compare_position(match,rng);++regression;}
    }
    // Reproducer: a Dream weapon cannot defend, while 16 distinct defense
    // barriers admit vast numbers of ordered subsets. Old diagnostic traversal
    // enumerates them all to conclude Dream changes nothing.
    std::vector<gf::ItemInstance> hand{{100,7,6,false},{101,6,{},false}};
    for(const auto& card:gf::catalog())if(card.category==gf::Category::armor && card.defense>0 && hand.size()<18)
        hand.push_back({gf::ItemToken(100+hand.size()),card.id,{},false});
    auto state=scene(hand).snapshot().state;state.players[1].items={{200,6,{},false}};
    gf::Match response;response.load_scene(state,77,123);response.step({77,gf::CommandKind::play,{200},42});
    timings.inspect(response,rng);
    const auto plan=official_restriction_plan(response.view(42),true,{2,9});
    check(plan.counts==RestrictionCounts{1,0,0,0,0,0,0,0},"large defense phantom restriction");
    check(timings.large>=700 && timings.dream>=1000 && timings.response>1000,"stress coverage missing");
    std::cout<<"legacy equivalence positions="<<regression<<"; stress positions="<<timings.positions
             <<" large="<<timings.large<<" dream="<<timings.dream<<" response="<<timings.response
             <<"; max plan="<<timings.plan<<" ms, selection="<<timings.selection<<" ms (budget 100 ms each)\n";
}

int main() {
    try {categories();relaxations();public_only();subsets();repair_regressions();regression_and_stress();}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
