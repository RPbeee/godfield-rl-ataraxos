#pragma once
#include "gfrl_core/vec_env.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace gfrl {
// Exercise the production history recording and tensor publication, including
// padding, Age, eviction and physical own-slot ordering. Never drive/reset here.
struct TestAccess {
    static void inject(VecEnv& env, const gf::Match& match) {
        auto& e=env.envs_[0]; e.match=match;
        for(auto& h:e.history) h.clear();
    }
    static void before(VecEnv& env) { env.before_decision(0); }
    static void record(VecEnv& env) { env.record_decision(0,false); }
    static void encode(VecEnv& env,int viewer) {
        auto& e=env.envs_[0];
        // Only actor_seat is read by encode_observation; this also permits
        // inspecting a nonacting viewer and a terminal Match.
        e.selection.emplace(gf::Match{}); e.selection->actor_seat=viewer;
        env.encode_observation(0);
    }
    static gf::Match& match(VecEnv& env) { return env.envs_[0].match; }
};
}

namespace observation_rollout {
using namespace gfrl;
inline void require(bool ok,const std::string& message) { if(!ok)throw std::runtime_error(message); }
inline std::vector<std::string> differences(const gf::Match& a,const gf::Match& b,int viewer) {
    auto x=a.view(viewer), y=b.view(viewer);
    std::vector<std::string> out;
#define FIELD(f) if(x.f!=y.f)out.push_back("view." #f)
    FIELD(phase); FIELD(gift_pool); FIELD(local_guardian_payloads); FIELD(external_artifacts);
    FIELD(apocalypse_active); FIELD(owner); FIELD(actor); FIELD(terminated); FIELD(truncated);
    FIELD(decisions); FIELD(pending_grant); FIELD(last_grant); FIELD(last_devil); FIELD(last_fate);
    FIELD(last_phenomenon); FIELD(last_guardian_payload); FIELD(last_auto_guardian_payloads);
    FIELD(last_auto_guardians); FIELD(last_guardian_changes); FIELD(last_guardian);
    FIELD(last_redraws); FIELD(last_reveals); FIELD(last_danger); FIELD(last_removal);
    FIELD(last_sale); FIELD(purchase); FIELD(last_purchase); FIELD(pending);
    FIELD(board.viewer); FIELD(board.hand); FIELD(board.gf); FIELD(board.tiebreak_gf); FIELD(board.is_over);
    if(x.board.players.size()!=y.board.players.size())out.push_back("view.board.players.size");
    for(std::size_t i=0;i<std::min(x.board.players.size(),y.board.players.size());++i) {
        const auto& p=x.board.players[i];const auto& q=y.board.players[i];
#define PLAYER(f) if(p.f!=q.f)out.push_back("view.board.players["+std::to_string(i)+"]." #f)
        PLAYER(id); PLAYER(display_name); PLAYER(team); PLAYER(dead); PLAYER(fogged); PLAYER(used_models);
        if(p.status.has_value()!=q.status.has_value())out.push_back("view.board.players["+std::to_string(i)+"].status.presence");
        if(p.status && q.status) {
#define STATUS(f) if(p.status->f!=q.status->f)out.push_back("view.board.players["+std::to_string(i)+"].status." #f)
            STATUS(hp); STATUS(mp); STATUS(cp); STATUS(curses); STATUS(guardian); STATUS(confused);
#undef STATUS
        }
#undef PLAYER
    }
#undef FIELD
    auto dx=a.last_decision(viewer),dy=b.last_decision(viewer);
    if(dx.has_value()!=dy.has_value())out.push_back("last_decision.presence");
    if(dx && dy) {
#define FIELD(f) if(dx->f!=dy->f)out.push_back("last_decision." #f)
        FIELD(decision); FIELD(gf); FIELD(phase); FIELD(source); FIELD(actor); FIELD(kind);
        FIELD(faces); FIELD(faces_hidden); FIELD(target); FIELD(allocation); FIELD(outcome_model);
#undef FIELD
    }
    require((x==y && dx==dy)==out.empty(),"difference reporter missed a public field");
    return out;
}
template<class T> void buffer_bits(const Buffer<T>& a,const Buffer<T>& b,std::size_t width,
                                   const std::string& name,const std::string& context) {
    require(a.size==b.size,"buffer shape mismatch");
    for(std::size_t j=0;j<a.size;++j)if(std::memcmp(a.data()+j,b.data()+j,sizeof(T))) {
        std::ostringstream s;
        s<<context<<" field="<<name<<" token="<<j/width<<" component="<<j%width
         <<" A="<<+a[j]<<" B="<<+b[j];
        throw std::runtime_error(s.str());
    }
}
inline void tensors(VecEnv& a,VecEnv& b,int viewer,const std::string& context) {
    TestAccess::encode(a,viewer); TestAccess::encode(b,viewer);
#define BUFFER(f,w) buffer_bits(a.f,b.f,w,#f,context)
    BUFFER(obs_type,1); BUFFER(obs_cards,8); BUFFER(obs_aux_card,1);
    BUFFER(obs_cat,C); BUFFER(obs_num,F); BUFFER(obs_valid,1);
#undef BUFFER
}
struct Counts {
    std::size_t decisions=0, comparisons=0, damage=0, defense=0, refill=0, prayer=0, fog=0, dream=0, trade=0;
    bool sufficient() const {
        return decisions>=10000 && damage>=50 && defense>=50 && refill>=50 && prayer>=50
            && fog>=50 && dream>=50 && trade>=50;
    }
};
inline bool contains_fog(const gf::MatchView& v) {
    return std::any_of(v.board.players.begin(),v.board.players.end(),[](const auto& p){return p.fogged;});
}
inline void events(Counts& n,const gf::Match& before,const gf::Match& after) {
    ++n.decisions;
    const auto& s=before.snapshot(); const auto& t=after.snapshot();
    const auto d=after.last_decision(0);
    n.defense+=d && d->kind==gf::CommandKind::defend && !d->faces.empty();
    n.prayer+=d && d->kind==gf::CommandKind::pray;
    n.trade+=d && (d->kind==gf::CommandKind::sell || d->kind==gf::CommandKind::buy);
    n.fog+=contains_fog(before.view(0)) || contains_fog(before.view(1));
    // A positive incoming attack followed by HP loss, excluding prayer,
    // exchange and owner disease ticks in unrelated main decisions.
    bool damage=false,refill=false;
    for(int seat=0;seat<2;++seat) {
        damage|=s.phase==gf::MatchPhase::response && before.view(seat).pending
            && before.view(seat).pending->power>0
            && d && d->kind==gf::CommandKind::defend && d->faces.empty()
            && !(t.last_tick && t.last_tick!=s.last_tick && t.last_tick->owner==t.state.players[seat].id)
            && t.state.players[seat].hp<s.state.players[seat].hp
            && before.view(seat).pending->target==t.state.players[seat].id;
        refill|=t.state.players[seat].items.size()>s.state.players[seat].items.size();
    }
    n.damage+=damage; n.refill+=refill;
    // Reveal vectors persist until the next main; count newly emitted suffixes.
    const auto x=before.view(0),y=after.view(0);
    const bool fresh=s.phase==gf::MatchPhase::main;
    n.dream+=!y.last_reveals.empty() && (fresh || y.last_reveals!=x.last_reveals);
}
inline bool blocked(const gf::Command& c,const std::set<gf::ItemToken>& hidden) {
    return std::any_of(c.items.begin(),c.items.end(),[&](auto t){return hidden.contains(t);});
}
inline void run() {
    const auto start=std::chrono::steady_clock::now();
    VecEnv va(1,1,1),vb(1,1,1);
    Counts total; std::array<std::size_t,3> pairs{},steps{};
    std::map<std::string,std::size_t> reasons,ends,primary_stops;
    std::size_t attempted=0,resampled=0,rejected=0,grants=0,confusions=0,max_length=0,existing_dream=0,created_dream=0;
    // Fixed seeds make failures reproducible. Increase the corpus in complete
    // a/b/c batches until both the length and event coverage contracts hold.
    for(std::size_t index=0;index<1200;++index) {
        if(index>=360 && index%3==0 && total.sufficient())break;
        ++attempted;
        const int variant=index%3,viewer=(index/3)%2;
        const std::uint64_t seed=0x8d97a42b0000ULL+index*7919;
        std::uint64_t policy=seed^0x123456789ULL,chance=seed^0xabcdefULL;
        gf::Match warm;
        const std::array<gf::PlayerSetup,2> players{{{0,gf::Team::solo,"A"},{1,gf::Team::solo,"B"}}};
        warm.reset(players,seed);
        const auto warm_length=3+index%17;
        for(std::size_t j=0;j<warm_length && !warm.done();++j)warm.rollout_step(policy,chance);
        // load_scene accepts quiescent boards, so finish any in-flight action.
        while(!warm.done() && warm.snapshot().phase!=gf::MatchPhase::main)warm.rollout_step(policy,chance);
        if(warm.done()) {++ends["warmup_finished"];continue;}
        auto sa=warm.snapshot().state,sb=sa;
        std::set<gf::ItemToken> hidden;
        if(variant==0) {
            for(auto& item:sb.players[1-viewer].items)if(!item.used && hidden.size()<2) {
                hidden.insert(item.token); item.model=item.model==6?7:6; item.fake_model.reset();
            }
        } else if(variant==1) {
            auto& p=sa.players[viewer];
            auto it=std::find_if(p.items.begin(),p.items.end(),[](const auto& i){
                return !i.used && i.fake_model && i.model!=*i.fake_model;
            });
            if(it!=p.items.end()) {
                ++existing_dream;
                hidden.insert(it->token); sb=sa;
                // Identity disguises are supported received Dream cards.
                sb.players[viewer].items[it-p.items.begin()].model=*it->fake_model;
            } else {
                it=std::find_if(p.items.begin(),p.items.end(),[](const auto& i){return !i.used;});
                if(it!=p.items.end()) {
                    ++created_dream;
                    if(std::find(p.curses.begin(),p.curses.end(),gf::Curse::dream)==p.curses.end())p.curses.push_back(gf::Curse::dream);
                    std::sort(p.curses.begin(),p.curses.end());
                    hidden.insert(it->token);it->model=19;it->fake_model=27;
                    sb=sa; sb.players[viewer].items[it-p.items.begin()].model=27;
                }
            }
        }
        if(variant!=2 && hidden.empty()) {++ends["no_unused_card"];continue;}
        gf::Match a,b;
        a.load_scene(sa,warm.snapshot().owner,seed^0xdeadbeefULL);
        b.load_scene(sb,warm.snapshot().owner,(seed^0xdeadbeefULL)+(variant==2?31337:0));
        require(a.snapshot().next_token==b.snapshot().next_token,"fixture token streams differ");
        TestAccess::inject(va,a); TestAccess::inject(vb,b);
        auto& ma=TestAccess::match(va);auto& mb=TestAccess::match(vb);
        ++pairs[variant];
        std::array<bool,2> active{true,true};
        auto compare=[&](std::size_t decision) {
            for(int v=0;v<2;++v)if(active[v]) {
                auto diff=differences(ma,mb,v);
                if(!diff.empty()) {
                    active[v]=false;
                    if(v==viewer && decision) {
                        // One exclusive primary-viewer stop reason per pair;
                        // the full per-field report below may overlap.
                        auto cause=diff.front();
                        for(const auto& preferred:{"view.last_reveals","view.last_removal","view.last_purchase","view.purchase"})
                            if(std::find(diff.begin(),diff.end(),preferred)!=diff.end()) {cause=preferred;break;}
                        ++primary_stops[std::string(1,char('a'+variant))+"/"+cause];
                    }
                    for(const auto& f:diff)++reasons[std::string(decision?"after/":"initial/")+char('a'+variant)+"/"+f];
                } else {
                    tensors(va,vb,v,"pair="+std::to_string(index)+" variant="+char('a'+variant)
                        +" seed="+std::to_string(seed)+" decision="+std::to_string(decision)+" viewer="+std::to_string(v));
                    ++total.comparisons;
                }
            }
        };
        compare(0);require(active[viewer],"designated viewer sees initial hidden mutation");
        std::size_t length=0;
        for(;length<192 && !ma.done() && active[viewer];) {
            gf::Command command;bool grant=ma.snapshot().phase==gf::MatchPhase::grant,confusion=false,found=grant;
            if(!grant) {
                auto actor=*ma.snapshot().actor;
                confusion=ma.snapshot().phase==gf::MatchPhase::main && ma.snapshot().state.players[actor].confusion_count>0;
                for(int retry=0;retry<256;++retry) {
                    command=confusion?ma.sample_confusion(chance):ma.sample_action(policy);
                    if(blocked(command,hidden)) {++resampled;continue;}
                    if(!(confusion?mb.confusion_legal(command):mb.legal(command))) {++rejected;continue;}
                    found=true;break;
                }
            }
            if(!found) {++ends["command_sampling_exhausted"];break;}
            const auto before=ma;
            TestAccess::before(va); TestAccess::before(vb);
            if(grant) {auto action=ma.sample_grant(chance);ma.grant_step(action);mb.grant_step(action);}
            else {
                require(confusion?mb.confusion_legal(command):mb.legal(command),"same command rejected: pair="+std::to_string(index)+" decision="+std::to_string(length+1));
                if(confusion) {ma.confusion_step(command);mb.confusion_step(command);}
                else {ma.step(command);mb.step(command);}
            }
            TestAccess::record(va); TestAccess::record(vb);
            ++length;
            compare(length);
            // Only accepted transitions with equal before AND after public
            // information contribute to event/decision coverage.
            if(active[viewer]) {events(total,before,ma);++steps[variant];grants+=grant;confusions+=confusion;}
        }
        max_length=std::max(max_length,length);
        if(!active[viewer])++ends[std::string("public_divergence/")+char('a'+variant)];
        else if(ma.done())++ends["finished"];
        else if(length==192)++ends["horizon"];
    }
    std::cout<<"rollout leak pairs a="<<pairs[0]<<" b="<<pairs[1]<<" c="<<pairs[2]<<" attempted="<<attempted
        <<" compared decisions="<<total.decisions<<" (a="<<steps[0]<<" b="<<steps[1]<<" c="<<steps[2]
        <<") viewer tensor comparisons="<<total.comparisons<<" max_length="<<max_length<<'\n';
    std::cout<<"events damage="<<total.damage<<" defense="<<total.defense<<" refill="<<total.refill
        <<" prayer="<<total.prayer<<" fog="<<total.fog<<" dream="<<total.dream<<" trade="<<total.trade
        <<" grants="<<grants<<" confusions="<<confusions<<" protected_resamples="<<resampled<<" B_illegal_resamples="<<rejected<<'\n';
    std::cout<<"Dream fixtures existing="<<existing_dream<<" created="<<created_dream<<'\n';
    for(const auto& [reason,count]:primary_stops)std::cout<<"primary stop "<<reason<<"="<<count<<'\n';
    for(const auto& [reason,count]:ends)std::cout<<"end "<<reason<<"="<<count<<'\n';
    // Field counts overlap when multiple public fields diverge in one decision.
    for(const auto& [reason,count]:reasons)std::cout<<"divergence "<<reason<<"="<<count<<'\n';
    std::cout<<"rollout seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<std::endl;
    require(total.sufficient(),"rollout coverage insufficient after 1200 pairs; see event counts");
}
}
inline void rollout_leaks() { observation_rollout::run(); }
