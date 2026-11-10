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
int main() { return upstream_wire_main(); }
