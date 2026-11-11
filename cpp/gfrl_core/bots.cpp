#include "bots.hpp"
#include "vec_env.hpp"
#include <algorithm>
#include <functional>
#include <tuple>

namespace gfrl {
std::uint64_t mix_seed(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
namespace {
std::uint64_t draw(std::uint64_t& rng) { rng += 0x9e3779b97f4a7c15ULL; return mix_seed(rng); }
std::size_t uniform(std::uint64_t& rng, std::size_t n) {
    const auto bound = std::uint64_t(n), threshold = -bound % bound;
    for (;;) { const auto x=draw(rng); if(x>=threshold)return x%bound; }
}
void actor_check(const gf::Match& m, gf::PlayerId id) {
    if(m.view(id).actor != id)throw std::invalid_argument("bot seat is not current actor");
}
}
Controller controller(std::string_view s) {
    if(s=="external")return Controller::external;
    if(s=="random")return Controller::random;
    if(s=="proposal")return Controller::proposal;
    if(s=="greedy")return Controller::greedy;
    throw std::invalid_argument("unknown controller: "+std::string(s));
}
gf::Command Random::act(const gf::Match& m, gf::PlayerId id, std::uint64_t& rng) {
    actor_check(m,id);
    auto kinds=m.legal_kinds(); gf::Command c; c.actor=id; c.kind=kinds[uniform(rng,kinds.size())];
    for (;;) {
        const auto node=policy_extensions(m,c.kind,c.items,true);
        const auto choice=uniform(rng,node.next.size()+node.can_finish);
        if(choice<node.next.size()) {c.items.push_back(node.next[choice]);continue;}
        c.target=node.targets[uniform(rng,node.targets.size())];
        if(node.exchange) {
            std::vector<int> hp,mp;
            for(int b=0;b<=20;++b)if(node.exchange->total-bucket_value(b)>=0 && node.exchange->total-bucket_value(b)<=198)hp.push_back(bucket_value(b));
            const int h=hp[uniform(rng,hp.size())];
            for(int b=0;b<=20;++b)if(node.exchange->total-h-bucket_value(b)>=0 && node.exchange->total-h-bucket_value(b)<=99)mp.push_back(bucket_value(b));
            const int p=mp[uniform(rng,mp.size())];
            c.allocation=gf::ResourceAllocation{h,p,int(node.exchange->total-h-p)};
        }
        return c;
    }
}
gf::Command Proposal::act(const gf::Match& m,gf::PlayerId id,std::uint64_t& rng) {actor_check(m,id);auto c=m.sample_action(rng);return forbidden_pair(m.snapshot(),c.items)?Random::act(m,id,rng):c;}
gf::Command Greedy::act(const gf::Match& m,gf::PlayerId id,std::uint64_t& rng) {
    actor_check(m,id); const auto v=m.view(id);
    const auto own=std::find_if(v.board.players.begin(),v.board.players.end(),[&](const auto& p){return p.id==id;});
    const auto& status=own->status;
    if(v.phase==gf::MatchPhase::purchase) {
        const bool accept=v.purchase && status && v.board.hand.size()<18 && status->cp>=v.purchase->price && v.purchase->price<=status->cp/2;
        gf::Command c{id,accept?gf::CommandKind::accept:gf::CommandKind::decline,{},{}};
        if(m.legal(c))return c;
        return Random::act(m,id,rng);
    }
    std::optional<gf::Command> attack,heal,defend,pray;
    std::tuple<double,std::int64_t,std::size_t> attack_score;
    std::tuple<std::int64_t,std::size_t> defense_score;
    int healing=-1;
    std::vector<gf::ItemToken> attack_tokens;
    auto consider=[&](const gf::Command& c) {
        std::vector<gf::CardId> faces;
        for(auto t:c.items) {
            auto it=std::find_if(v.board.hand.begin(),v.board.hand.end(),[&](const auto& x){return x.token==t;});
            faces.push_back(it->face); // Never read reference or a Dream's true model.
        }
        if(c.kind==gf::CommandKind::pray)pray=c;
        if(c.kind==gf::CommandKind::defend && v.pending) {
            std::int64_t defense=0;bool reaction=false;
            auto element=v.pending->element;
            for(auto f:faces) {
                const auto& card=*gf::find_card(f);
                if(card.ability==gf::Ability::filter_atk_element)element.reset();
                defense+=card.defense;
                reaction |= gf::reaction_for(f,v.pending->category,element)!=gf::Reaction::none;
            }
            auto score=std::tuple{reaction?std::int64_t(0):std::max<std::int64_t>(0,v.pending->power-defense),faces.size()};
            if(!defend || score<defense_score){defend=c;defense_score=score;}
        }
        if(c.kind!=gf::CommandKind::play || faces.empty())return;
        const auto& base=*gf::find_card(faces[0]);
        const bool opponent=c.target && *c.target!=id;
        // Untargeted chance/wide attacks can hit the opponent; ignore redirect/self-hit risk.
        const bool wide=std::any_of(faces.begin(),faces.end(),[](auto f){return gf::find_card(f)->ability==gf::Ability::attack_every_enemy;});
        const bool enemy=opponent || wide || (!c.target && base.hit_rate>0);
        const auto cost=gf::magic_cost(faces);
        const auto power=gf::attack_power(faces,base.ability==gf::Ability::atk_by2x_mp && status ? 2*(status->mp-cost):0);
        if(enemy && (base.category==gf::Category::weapons || base.category==gf::Category::miracles) && power>0) {
            auto mp_used=base.ability==gf::Ability::atk_by2x_mp && status?status->mp:cost;
            auto score=std::tuple{-double(power)*(base.hit_rate?base.hit_rate/100.0:1.0),mp_used,faces.size()};
            if(!attack || score<attack_score){attack=c;attack_score=score;}
            attack_tokens.insert(attack_tokens.end(),c.items.begin(),c.items.end());
        }
        if(status && status->hp<=20 && c.target==id && (base.category==gf::Category::miracles || base.category==gf::Category::sundries) && base.ability==gf::Ability::boost_hp && base.ability_value>healing) {
            healing=base.ability_value;heal=c;
        }
    };
    gf::Command c;c.actor=id;
    std::function<void()> walk=[&] {
        const auto node=policy_extensions(m,c.kind,c.items,true);
        if(node.can_finish)for(auto target:node.targets){c.target=target;consider(c);}
        if(c.items.size()==4)return;
        for(auto t:node.next){c.items.push_back(t);walk();c.items.pop_back();}
    };
    for(auto kind:m.legal_kinds())if(kind==gf::CommandKind::play || kind==gf::CommandKind::defend || kind==gf::CommandKind::pray){c.kind=kind;walk();}
    if(v.phase==gf::MatchPhase::response && defend)return *defend;
    if(attack)return *attack;
    if(heal)return *heal;
    if(pray)return *pray;
    // Own physical inventory order is permitted; no foreign inventory or hidden model is read.
    const auto seat=std::size_t(own-v.board.players.begin());
    for(const auto& item:m.snapshot().state.players[seat].items) {
        if(std::find(attack_tokens.begin(),attack_tokens.end(),item.token)!=attack_tokens.end())continue;
        gf::Command discard{id,gf::CommandKind::discard,{item.token},{}};
        if(m.legal(discard))return discard;
    }
    return Random::act(m,id,rng);
}
gf::Command bot_action(Controller b,const gf::Match& m,gf::PlayerId id,std::uint64_t& rng) {
    switch(b){case Controller::random:return Random::act(m,id,rng);case Controller::proposal:return Proposal::act(m,id,rng);case Controller::greedy:return Greedy::act(m,id,rng);default:throw std::invalid_argument("external is not a bot");}
}
void reset_evaluation(gf::Match& m,std::uint64_t seed,std::uint64_t g,std::uint64_t& chance,std::array<std::uint64_t,2>& policy) {
    const auto root=mix_seed(seed ^ mix_seed(g/2) ^ 0x4556414c50414952ULL);
    const std::array<gf::PlayerSetup,2> players{{{0,gf::Team::solo,"A"},{1,gf::Team::solo,"B"}}};
    m.reset(players,mix_seed(root^0x13198a2e03707344ULL));
    chance=mix_seed(root^0xa4093822299f31d0ULL);
    for(int seat=0;seat<2;++seat)policy[seat]=mix_seed(root ^ (seat==int(g%2)?0x4147454e54ULL:0x424f54ULL));
}
EvalResult evaluation_result(const gf::Match& m,std::uint64_t g) {
    const auto& s=m.snapshot();return {g,g/2,int(g%2),s.truncated?0:int(s.rewards.at(g%2)),s.truncated,s.decisions,{s.state.players[0].hp,s.state.players[1].hp}};
}
std::vector<EvalResult> evaluate_bots(Controller a,Controller b,std::size_t pairs,std::size_t threads,std::uint64_t seed) {
    if(a==Controller::external || b==Controller::external || !threads || threads>1000000 || pairs>10000000)throw std::invalid_argument("invalid bot evaluation arguments");
    std::vector<EvalResult> results(2*pairs);Workers workers(threads);
    workers.run([&](std::size_t worker){for(std::size_t g=worker;g<results.size();g+=threads){
        gf::Match m;std::uint64_t chance;std::array<std::uint64_t,2> policy;reset_evaluation(m,seed,g,chance,policy);
        while(!m.done()) {
            const auto v=m.view({});
            if(v.phase==gf::MatchPhase::grant){m.grant_step(m.sample_grant(chance));continue;}
            const auto seat=std::size_t(*v.actor);
            const auto own=m.view(*v.actor);
            if(v.phase==gf::MatchPhase::main && own.board.players[seat].status->confused)m.confusion_step(m.sample_confusion(chance));
            else {auto c=bot_action(seat==g%2?a:b,m,*v.actor,policy[seat]);if(!m.legal(c))throw std::runtime_error("bot produced illegal Command");m.step(c);}
        }
        results[g]=evaluation_result(m,g);
    }});return results;
}
}
