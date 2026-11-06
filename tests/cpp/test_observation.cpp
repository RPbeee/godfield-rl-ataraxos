#include "gfrl_core/history.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
using namespace gfrl;
void check(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
gf::Match scene(std::vector<gf::CardId> own,std::vector<gf::CardId> other,std::uint64_t seed=123) {
    gf::Match m;auto s=m.snapshot().state;
    for(int i=0;i<2;++i) {s.players[i].items.clear();s.players[i].guardian=gf::Guardian::none;s.players[i].curses.clear();}
    gf::ItemToken token=100;
    for(auto id:own)s.players[0].items.push_back({token++,id,{},false});
    token=200;for(auto id:other)s.players[1].items.push_back({token++,id,{},false});
    m.load_scene(s,s.players[0].id,seed);return m;
}
std::vector<Token> observation(const gf::Match& m,const History& h,int seat=0) {
    auto id=m.snapshot().state.players[seat].id;std::vector<gf::ItemToken> order;
    for(const auto& item:m.snapshot().state.players[seat].items)order.push_back(item.token);
    auto v=m.view(id);auto ts=encode(v,id,order);
    for(const auto& e:h.events()){auto t=e.token;t.num[Age]=float(v.decisions-e.decision)/64;ts.push_back(t);}return ts;
}
void bits(const std::vector<Token>& a,const std::vector<Token>& b) {
    check(a.size()==b.size(),"token count mismatch");
    for(std::size_t i=0;i<a.size();++i) {
        check(a[i].type==b[i].type && a[i].cards==b[i].cards && a[i].aux==b[i].aux && a[i].cat==b[i].cat,"categorical observation leaked");
        check(std::memcmp(a[i].num.data(),b[i].num.data(),F*sizeof(float))==0,"numeric observation leaked");
    }
}
void accept(gf::Match& m,History& h,const gf::Command& c,int viewer=0) {
    auto id=m.snapshot().state.players[viewer].id;auto before=m.view(id);
    if(!m.legal(c)) throw std::runtime_error("fixture illegal kind="+std::to_string(int(c.kind))+" phase="+std::to_string(int(m.view(id).phase)));m.step(c);
    h.append(before,m.view(id),*m.last_decision(id),id);
}
void leaks() {
    std::size_t comparisons=0;
    for(int variant=0;variant<3;++variant) {
        std::vector<gf::CardId> full(18,120);auto a=scene(full,full),b=a;
        auto initial=a.snapshot().state; initial.players[1].curses={gf::Curse::dream};a.load_scene(initial,0,123);b=a;
        auto state=b.snapshot().state;
        if(variant==0) {state.players[1].items.pop_back();state.players[1].items.back().model=7;state.players[1].items.back().fake_model=6;}
        if(variant==1) {
            auto s=a.snapshot().state;s.players[0].curses={gf::Curse::dream};s.players[0].items.back().model=19;s.players[0].items.back().fake_model=27;
            a.load_scene(s,0,123);state=s;state.players[0].items.back().model=27;
        }
        b.load_scene(state,0,variant==2?999:123);
        History ha,hb;bits(observation(a,ha),observation(b,hb));
        // 32 deterministic discards leave all differing cards untouched. No gifts, draws, damage or guardian rolls.
        for(int round=0;round<16;++round)for(int seat=0;seat<2;++seat) {
            gf::Command c;c.actor=seat;c.kind=gf::CommandKind::discard;c.items={gf::ItemToken((seat?200:100)+round)};
            accept(a,ha,c);accept(b,hb,c);bits(observation(a,ha),observation(b,hb));++comparisons;
        }
        check(ha.events().size()==32,"history decision count");
        if(variant==0)check(a.snapshot().state.players[1].items.size()!=b.snapshot().state.players[1].items.size(),"hidden counts converged");
    }
    std::cout<<"leak pairs=3 public decisions=32/pair bitwise comparisons="<<comparisons<<"\n";
}
void slots() {
    auto m=scene({7,19,6},{});auto s=m.snapshot().state;s.players[0].curses={gf::Curse::dream};s.players[0].items[1].fake_model=27;m.load_scene(s,0,1);
    auto v=m.view(0);History h;auto ts=observation(m,h);
    for(std::size_t j=0;j<s.players[0].items.size();++j) {auto token=s.players[0].items[j].token;
        auto it=std::find_if(v.board.hand.begin(),v.board.hand.end(),[&](const auto& x){return x.token==token;});
        check(ts[3+j].cards[0]==card_code(it->face),"slot face mismatch");
    }
    check(ts[4].cards[0]==card_code(27) && ts[4].cards[0]!=card_code(19),"Dream exposed true model");
    check(ts[4].num[Disguised]==1,"missing disguise");
    std::cout<<"own slot order and Dream face verified\n";
}
void decision(const History& h,gf::CommandKind kind,int actor,int target,std::initializer_list<int> faces,float dhp=0,int damaged=1) {
    auto it=std::find_if(h.events().rbegin(),h.events().rend(),[](const auto& e){return e.token.type==Decision;});check(it!=h.events().rend(),"no decision");const auto& t=it->token;
    check(t.cat[Kind]==int(kind)+1 && t.cat[Actor]==actor && t.cat[Target]==target,"decision metadata mismatch");int i=0;
    for(auto id:faces)check(t.cards[i++]==card_code(id),"decision faces mismatch");
    check(t.num[Delta0+(damaged-1)*18]==dhp,"HP delta mismatch");
}
void contents() {
    {auto m=scene({13},{120});History h;accept(m,h,{0,gf::CommandKind::play,{100},1});decision(h,gf::CommandKind::play,1,2,{13});
        accept(m,h,{1,gf::CommandKind::defend,{200}, {}});decision(h,gf::CommandKind::defend,2,0,{120},-1/99.f,2);}
    {auto m=scene({120},{120});History h;accept(m,h,{0,gf::CommandKind::discard,{100},{}});decision(h,gf::CommandKind::discard,1,0,{120});
        accept(m,h,{1,gf::CommandKind::pray,{}, {}});decision(h,gf::CommandKind::pray,2,0,{});}
    {auto m=scene({3},{});History h;accept(m,h,{0,gf::CommandKind::exchange,{100},{},gf::ResourceAllocation{30,20,20}});decision(h,gf::CommandKind::exchange,1,1,{3},-10/99.f);
        check(h.events()[0].token.num[HasAllocation]==1,"allocation missing");}
    for(auto kind:{gf::CommandKind::sell,gf::CommandKind::buy}) {
        auto m=scene({gf::CardId(kind==gf::CommandKind::sell?4:5),6},{7});History h;
        gf::Command c{0,kind,{100},1};if(kind==gf::CommandKind::sell)c.items.push_back(101);
        accept(m,h,c);decision(h,kind,1,2,kind==gf::CommandKind::sell?std::initializer_list<int>{4,6}:std::initializer_list<int>{5});
        if(m.view(0).phase==gf::MatchPhase::response)accept(m,h,{1,gf::CommandKind::defend,{}, {}});
        if(kind==gf::CommandKind::buy) {check(m.view(0).phase==gf::MatchPhase::purchase,"missing purchase");accept(m,h,{0,gf::CommandKind::accept,{}, {}});decision(h,gf::CommandKind::accept,1,0,{});}
        check(std::any_of(h.events().begin(),h.events().end(),[&](const auto& e){return e.token.type==(kind==gf::CommandKind::buy?Purchase:Sale);}),"trade companion missing");
    }
    {auto m=scene({120},{120});auto s=m.snapshot().state;s.players[0].curses={gf::Curse::fog};m.load_scene(s,1,123);History h;
        accept(m,h,{1,gf::CommandKind::discard,{200},{}});const auto& t=h.events().front().token;
        check(t.num[FacesHidden] && t.cards[0]==0,"fog discard leaked");check(t.num[Delta0+18+3]==1,"hidden status flag missing");}
    std::cout<<"play defend discard pray exchange sell buy purchase fog verified\n";
}
void ring() {
    auto m=scene(std::vector<gf::CardId>(18,120),std::vector<gf::CardId>(18,120));History h(3);
    for(int round=0;round<5;++round)for(int seat=0;seat<2;++seat)accept(m,h,{gf::PlayerId(seat),gf::CommandKind::discard,{gf::ItemToken((seat?200:100)+round)}, {}});
    check(h.events().size()==3 && h.dropped==7,"ring overflow wrong");check(h.events().front().decision<h.events().back().decision,"history order wrong");h.clear();check(h.events().empty(),"history clear failed");
    // A repeated same-valued grant is new even though optional equality cannot detect it.
    gf::MatchView b,a;b.last_grant=gf::DevilEvent{0,0,{}};a=b;a.decisions=1;gf::PublicDecision d;d.phase=gf::MatchPhase::grant;d.source=gf::DecisionSource::grant;
    h.append(b,a,d,0);check(h.events().size()==2 && h.events().back().token.type==Grant,"identical grant suppressed");
    auto retained=a;retained.decisions=2;d.phase=gf::MatchPhase::response;d.source=gf::DecisionSource::player;h.append(a,retained,d,0);
    check(h.events().back().token.type==Decision,"stale grant duplicated");
    History all;
    gf::MatchView initial, notices; notices.decisions=1;
    notices.last_reveals={{0,19,27}};notices.last_removal=gf::RemovalEvent{0,0,1,205,{}};
    notices.last_sale=gf::SaleEvent{0,0,1,6,1,{}};
    notices.last_purchase=gf::PurchaseEvent{0,0,1,7,10,gf::PurchaseReason::bought,{}};
    notices.last_danger=gf::DangerEvent{0,1,{94},false,99,{}};
    notices.last_devil=gf::DevilEvent{1,240,{}};notices.last_grant=gf::DevilEvent{1,0,{}};
    notices.last_fate=gf::FateEvent{0,288};notices.last_phenomenon=gf::PhenomenonView{288,{}};
    notices.last_guardian_changes={{0,gf::Guardian::none,gf::Guardian::mars,gf::GuardianChangeReason::summon}};
    notices.last_redraws={1};d.phase=gf::MatchPhase::main;all.append(initial,notices,d,0);
    check(all.events().size()==12,"missing companion kind");
    d.phase=gf::MatchPhase::response;auto next=notices;next.decisions=2;all.append(notices,next,d,0);
    check(all.events().size()==13,"retained notices duplicated");
    next.last_reveals.push_back({0,19,27});next.last_redraws.push_back(1);next.last_guardian_changes.push_back(next.last_guardian_changes[0]);
    all.append(notices,next,d,0);check(all.events().size()==17,"identical vector suffix lost");
    d.phase=gf::MatchPhase::main;all.append(notices,notices,d,0);check(all.events().size()==29,"identical events after reset lost");
    auto visible=scene({},{}).view(0);visible.board.players[1].used_models.assign(18,199);
    auto encoded=encode(visible,0,{});check(encoded.size()==55 && encoded[2].num[UsedOverflow]==2,"used overflow wrong");
    check(std::count_if(encoded.begin(),encoded.end(),[](const auto& t){return t.type==Used;})==16,"used limit wrong");
    visible.pending=gf::PendingView{};visible.pending->models.assign(10,6);
    encoded=encode(visible,0,{});check(encoded[21].num[CountOverflow]==2,"card overflow wrong");
    check(encoded[21].num[ForcedHidden]==1,"optional visibility flag wrong");
    std::cout<<"ring eviction=7; identical grants and retained notices verified\n";
}
int main(int argc,char** argv) {try {check(argc==2,"case required");std::string c=argv[1];if(c=="leak")leaks();else if(c=="slots")slots();else if(c=="history")contents();else if(c=="ring")ring();else throw std::runtime_error("unknown");}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
