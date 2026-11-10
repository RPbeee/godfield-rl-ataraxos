// Reuse the upstream native replays, including externally supplied receipts.
// This emits public observations only; no received or native hidden inventories.
#include "gfrl_core/history.hpp"
#include <iostream>
#include <set>
#include <sstream>
void official_observe(const godfield::official::Match&);
#define GF_WIRE_OBSERVER official_observe
#define main(...) upstream_wire_main(__VA_ARGS__)
#include "../../third_party/goodfield/tests/official_wire_trace_test.cpp"
#undef main
#undef GF_WIRE_OBSERVER
using namespace gfrl;
void official_observe(const gf::Match& match) {
    static std::set<std::string> seen;
    static std::optional<gf::MatchSnapshot> previous;
    static std::array<History,2> history{History(64),History(64)};
    static std::uint64_t policy_decisions=0;
    const auto& snap=match.snapshot();
    if(!previous || snap.decisions<previous->decisions || snap.decisions==0) {
        history={History(64),History(64)};policy_decisions=0;previous=snap;
    } else if(snap.decisions>previous->decisions) {
        gf::Match before;before.restore(*previous);
        const auto d=match.last_decision(snap.state.players[0].id);
        if(d && d->source!=gf::DecisionSource::grant && d->source!=gf::DecisionSource::artifact_grant) {
            ++policy_decisions;
            for(int seat=0;seat<2;++seat) {
                const auto id=snap.state.players[seat].id;
                auto a=match.view(id),b=before.view(id);auto record=*match.last_decision(id);
                a.decisions=policy_decisions;b.decisions=policy_decisions-1;record.decision=policy_decisions;
                history[seat].append(b,a,record,id);
            }
        }
        previous=snap;
    }
    for(const auto& player:match.snapshot().state.players) {
        const auto seat=player.id==snap.state.players[0].id?0:1;
        const auto v=match.view(player.id);std::vector<gf::ItemToken> order;
        for(const auto& item:player.items)order.push_back(item.token);
        auto tokens=encode(v,player.id,order);
        tokens.resize(history_offset+64);std::size_t offset=history_offset;
        for(const auto& e:history[seat].events()) {tokens[offset]=e.token;tokens[offset++].num[Age]=float(policy_decisions-e.decision)/64;}
        std::ostringstream out;
        out<<"{\"viewer\":"<<player.id<<",\"policy_decisions\":"<<policy_decisions<<",\"view\":"<<match.observe(player.id)<<",\"tokens\":[";
        for(std::size_t i=0;i<tokens.size();++i) {
            if(i)out<<',';const auto& t=tokens[i];out<<'['<<int(t.type)<<','<<t.aux<<",[";
            for(std::size_t j=0;j<t.cards.size();++j){if(j)out<<',';out<<t.cards[j];}out<<"],[";
            for(std::size_t j=0;j<t.cat.size();++j){if(j)out<<',';out<<int(t.cat[j]);}out<<"],[";
            out.precision(9);for(std::size_t j=0;j<t.num.size();++j){if(j)out<<',';out<<t.num[j];}out<<"]]";
        }
        out<<"],\"prefixes\":[";
        if(v.actor==player.id && v.phase!=gf::MatchPhase::grant && v.phase!=gf::MatchPhase::finished) {
            Selection s(match);bool first=true;
            while(!s.complete) {
                const auto n=std::count(s.mask.begin(),s.mask.end(),true);
                if(n>1) {
                    if(!first)out<<',';first=false;
                    out<<"{\"stage\":"<<int(s.stage)<<",\"mask\":[";
                    for(std::size_t i=0;i<21;++i){if(i)out<<',';out<<(s.mask[i]?"true":"false");}
                    out<<"],\"choice\":"<<(std::find(s.mask.begin(),s.mask.end(),true)-s.mask.begin())<<'}';
                }
                s.choose(match,std::find(s.mask.begin(),s.mask.end(),true)-s.mask.begin(),true);
            }
        }
        out<<"]}";
        // Decisions vary across external-grant proofs; preserve them for tests.
        auto line=out.str();if(seen.insert(line).second)std::cout<<line<<'\n';
    }
}
int main(int argc, char** argv) {
    if(argc==2 && std::string_view(argv[1])=="--guardian-decisions") {
        const auto compare=[](const gf::Match& match) {
            auto v=match.view(*match.snapshot().actor);
            const auto& own=match.snapshot().state.players[*v.actor==0?0:1];
            const auto hand=v.board.hand;v.board.hand.clear();
            for(const auto& item:own.items)v.board.hand.push_back(*std::find_if(hand.begin(),hand.end(),[&](auto h){return h.token==item.token;}));
            gf::PublicSelection public_selection(v);
            if(match.legal_kinds()!=public_selection.legal_kinds())throw std::runtime_error("guardian kinds mismatch");
            for(auto kind:match.legal_kinds()) {
                std::vector<gf::ItemToken> prefix;
                const auto visit=[&](auto&& self)->void {
                    auto a=match.legal_extensions(kind,prefix,{true});
                    if(a!=public_selection.legal_extensions(kind,prefix,{true}))throw std::runtime_error("guardian prefix mismatch");
                    for(auto token:a.next) {prefix.push_back(token);self(self);prefix.pop_back();}
                };visit(visit);
            }
            official_observe(match);
        };
        for(auto guardian:{gf::Guardian::none,gf::Guardian::earth,gf::Guardian::moon,gf::Guardian::mars}) {
            gf::SnapshotState state;
            gf::PlayerState a;a.id=0;a.guardian=guardian;a.items={{0,6,{},false},{1,130,{},false},{2,235,{},false}};
            gf::PlayerState b;b.id=7;b.items={{0,6,{},false},{1,142,{},false},{2,190,{},false},{3,223,{},false}};
            state.players={a,b};gf::Match match;match.load_scene(state,7,0);compare(match);
            match.step({7,gf::CommandKind::play,{0},0});compare(match);
        }
        for(auto guardian:{gf::Guardian::earth,gf::Guardian::moon}) {
            std::vector<std::vector<gf::CardId>> payloads;
            for(const auto& card:gf::catalog())if(gf::match_guardian_payload_supported(guardian,card.id))payloads.push_back({card.id});
            if(guardian==gf::Guardian::earth)payloads.push_back({6,203});
            for(const auto& models:payloads) {
                gf::SnapshotState state;
                gf::PlayerState a;a.id=0;a.guardian=guardian;
                gf::PlayerState b;b.id=7;b.mp=99;b.items={{0,130,{},false},{1,142,{},false},{2,190,{},false},{3,223,{},false}};
                state.players={a,b};gf::Match match;match.load_scene(state,7,0);
                match.guardian_payload_step({0,guardian,models,7});
                if(match.snapshot().phase==gf::MatchPhase::response)compare(match);
            }
        }
        return 0;
    }
    if(argc==3 && std::string_view(argv[1])=="--public-events") {
        const auto scenario=std::string_view(argv[2]);
        gf::SnapshotState state;state.gf=1;
        gf::PlayerState source;source.id=0;
        gf::PlayerState target;target.id=7;
        source.items={{0,scenario=="cp"?gf::CardId{236}:scenario=="sale"?gf::CardId{4}:gf::CardId{130},{},false}};
        if(scenario=="sale") {
            source.items.push_back({1,6,{},false});
            for(unsigned i=0;i<18;++i)target.items.push_back({i,235,{},true});
        } else target.items={{0,130,{},false}};
        if(scenario=="disease")source.curses={gf::Curse::cold};
        state.players={source,target};
        unsigned seed=0;
        if(scenario=="disease")for(;;++seed) {
            gf::Match trial;trial.load_scene(state,0,seed);
            trial.step({0,gf::CommandKind::pray,{},{}});
            if(trial.snapshot().last_tick && trial.snapshot().last_tick->worsened)break;
        }
        gf::Match match;match.load_scene(state,0,seed);official_observe(match);
        if(scenario=="sale") {
            match.step({0,gf::CommandKind::sell,{0,1},7});official_observe(match);
            match.step({7,gf::CommandKind::defend,{},{}});
        } else if(scenario=="cp")match.step({0,gf::CommandKind::play,{0},0});
        else match.step({0,gf::CommandKind::pray,{},{}});
        official_observe(match);return 0;
    }
    if(argc==2 && std::string_view(argv[1])=="--removals") {
        for(auto operation: {gf::CardId{205},gf::CardId{206}}) {
            gf::SnapshotState state;
            gf::PlayerState source;source.id=0;source.items={{0,operation,{},false}};
            gf::PlayerState target;target.id=7;
            for(auto model: {gf::CardId{235},gf::CardId{235},gf::CardId{223},gf::CardId{6}})
                target.items.push_back({target.items.size(),model,{},operation==206 && model!=6});
            state.players={source,target};
            gf::Match match;match.load_scene(state,0,42);
            official_observe(match);
            match.step(gf::Command{0,gf::CommandKind::play,{0},7});
            official_observe(match);
            match.step(gf::Command{7,gf::CommandKind::defend,{}, {}});
            official_observe(match);
        }
        return 0;
    }
    if(argc==2 && std::string_view(argv[1])=="--trades") {
        for(auto operation: {gf::CardId{4},gf::CardId{5}}) {
            gf::SnapshotState state;
            gf::PlayerState source;source.id=0;source.items={{0,operation,{},false},{1,48,{},false}};
            gf::PlayerState target;target.id=7;
            for(auto model: {gf::CardId{142},gf::CardId{180},gf::CardId{189},gf::CardId{190}})
                target.items.push_back({target.items.size(),model,{},false});
            state.players={source,target};
            gf::Match match;match.load_scene(state,0,42);
            gf::Command command{0,operation==4?gf::CommandKind::sell:gf::CommandKind::buy,{0},7};
            if(operation==4)command.items.push_back(1);
            match.step(command);
            official_observe(match);
        }
        return 0;
    }
    if(argc==2 && std::string_view(argv[1])=="--counter-values") {
        for(gf::CardId model=181;model<=188;++model) for(unsigned seed=0;seed<16;++seed) {
            gf::SnapshotState state;
            gf::PlayerState source;source.id=0;
            source.items={{0,48,{},false},{1,130,{},false},{2,190,{},false}};
            gf::PlayerState target;target.id=7;target.items={{0,model,{},false}};
            state.players={source,target};
            gf::Match match;match.load_scene(state,0,seed);
            match.step({0,gf::CommandKind::play,{0},7});
            match.step({7,gf::CommandKind::defend,{0},{}});
            official_observe(match);
        }
        return 0;
    }
    if(argc==2 && std::string_view(argv[1])=="--attack-values") {
        // Launch every single-card attack from a real Match, plus ordered
        // dynamic/ordinary modifier cases. No synthetic pending injection.
        std::vector<std::vector<gf::CardId>> cases;
        for(const auto& card:gf::catalog())
            if(gf::match_supported(card.id) && ((card.atk>0 && !card.is_plus_atk) || card.id==58))
                cases.push_back({card.id});
        for(auto base:{gf::CardId{6},gf::CardId{58},gf::CardId{94}})
            for(auto extra: {std::vector<gf::CardId>{231},std::vector<gf::CardId>{231,204},
                             std::vector<gf::CardId>{9},std::vector<gf::CardId>{9,231}}) {
                extra.insert(extra.begin(),base);cases.push_back(extra);
            }
        for(const auto& models:cases) for(unsigned seed=0;seed<16;++seed) {
            gf::SnapshotState state;
            gf::PlayerState source;source.id=0;source.mp=30;
            for(auto model:models) source.items.push_back({source.items.size(),model,{},false});
            gf::PlayerState target;target.id=7;target.mp=99;
            for(auto model:{gf::CardId{130},gf::CardId{142},gf::CardId{180},gf::CardId{184},
                           gf::CardId{189},gf::CardId{190},gf::CardId{223},gf::CardId{235}})
                target.items.push_back({target.items.size(),model,{},false});
            state.players={source,target};
            gf::Match match;match.load_scene(state,0,seed);
            gf::Command command{0,gf::CommandKind::play,{},7};
            for(std::size_t i=0;i<models.size();++i)command.items.push_back(i);
            if(!match.legal(command)) {command.target.reset();if(!match.legal(command))continue;}
            match.step(command);official_observe(match);
        }
        return 0;
    }
    if(argc==2 && std::string_view(argv[1])=="--dynamic") {
        for(auto model: {gf::CardId{202},gf::CardId{58},gf::CardId{94}}) {
            for(unsigned variant=0;variant<16;++variant) {
                gf::SnapshotState state;
                gf::PlayerState source;source.id=0;source.mp=variant%2?10:20;
                source.items={{0,model,{},false}};
                gf::PlayerState target;target.id=7;
                target.items={{0,130,{},false},{1,190,{},false}};
                if(model==94 && variant%2) target.items.push_back({2,209,{},false});
                state.players={source,target};
                gf::Match match;match.load_scene(state,0,variant);
                match.step(gf::Command{0,gf::CommandKind::play,{0},model==94?std::optional<gf::PlayerId>{}:7});
                official_observe(match);
            }
        }
        return 0;
    }
    return upstream_wire_main();
}
