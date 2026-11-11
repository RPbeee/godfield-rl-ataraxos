#pragma once
// This boundary accepts only the allowlisted public representation, never a wire game.
namespace {
template<class E, class Name> E official_enum(const std::string& name,int count,Name get) {
    for(int i=0;i<count;++i)if((i==0 && name=="none") || name==get(static_cast<E>(i)))return static_cast<E>(i);
    throw std::invalid_argument("unknown public enum");
}
template<class T> T field(nb::dict d,const char* key,T fallback) {
    return d.contains(key)?nb::cast<T>(d[key]):fallback;
}
gf::MatchView official_view(nb::dict d) {
    gf::MatchView v;const auto viewer=nb::cast<gf::PlayerId>(d["viewer"]);
    v.board.viewer=viewer;v.owner=nb::cast<gf::PlayerId>(d["owner"]);
    v.actor=nb::cast<std::optional<gf::PlayerId>>(d["actor"]);
    v.phase=static_cast<gf::MatchPhase>(nb::cast<int>(d["phase"]));
    v.decisions=field<std::uint64_t>(d,"decisions",0);v.board.gf=nb::cast<std::int64_t>(d["gf"]);
    v.board.tiebreak_gf=nb::cast<std::optional<std::int64_t>>(d["tiebreak_gf"]);
    v.board.is_over=field<bool>(d,"is_over",false);v.terminated=v.board.is_over;
    v.apocalypse_active=v.board.tiebreak_gf && v.board.gf>=*v.board.tiebreak_gf;
    for(auto obj:nb::cast<nb::list>(d["players"])) {
        auto p=nb::cast<nb::dict>(obj);gf::PlayerView x;x.id=nb::cast<gf::PlayerId>(p["id"]);
        x.dead=nb::cast<bool>(p["dead"]);x.fogged=nb::cast<bool>(p["fogged"]);
        x.used_models=nb::cast<std::vector<gf::CardId>>(p["used"]);
        if(!p["status"].is_none()) {
            auto a=nb::cast<nb::dict>(p["status"]);gf::StatusPane status;
            status.hp=nb::cast<std::int64_t>(a["hp"]);status.mp=nb::cast<std::int64_t>(a["mp"]);status.cp=nb::cast<std::int64_t>(a["cp"]);
            status.guardian=official_enum<gf::Guardian>(nb::cast<std::string>(a["guardian"]),11,gf::guardian_name);
            status.confused=nb::cast<bool>(a["confused"]);
            for(auto c:nb::cast<std::vector<std::string>>(a["curses"]))status.curses.push_back(official_enum<gf::Curse>(c,9,gf::curse_name));
            x.status=status;
        }
        v.board.players.push_back(x);
    }
    for(auto obj:nb::cast<nb::list>(d["hand"])) {
        auto a=nb::cast<nb::dict>(obj);gf::HandItemView h;
        h.token=v.board.hand.size()+1;h.face=nb::cast<gf::CardId>(a["face"]);h.reference=h.face;
        if(!gf::find_card(h.face))throw std::invalid_argument("unknown public card");
        h.used=nb::cast<bool>(a["used"]);h.disguised=nb::cast<bool>(a["disguised"]);v.board.hand.push_back(h);
    }
    if(d.contains("pending") && !d["pending"].is_none()) {
        auto a=nb::cast<nb::dict>(d["pending"]);gf::PendingView p;
        p.models=nb::cast<std::vector<gf::CardId>>(a["models"]);
        if(p.models.empty())throw std::invalid_argument("empty public attack");
        for(auto id:p.models)if(!gf::find_card(id))throw std::invalid_argument("unknown public attack");
        const auto& base=*gf::find_card(p.models.front());
        p.origin=nb::cast<std::optional<gf::PlayerId>>(a["origin"]);p.source=nb::cast<std::optional<gf::PlayerId>>(a["source"]);
        p.target=nb::cast<gf::PlayerId>(a["target"]);p.category=base.category;p.element=gf::attack_element(p.models);
        p.ability=base.ability;p.amount=field<std::int64_t>(a,"amount",base.ability_value);
        if(base.ability==gf::Ability::sell && p.models.size()>1)p.amount=gf::find_card(p.models[1])->price;
        if(base.ability==gf::Ability::remove_items)p.amount=3;
        if(base.ability==gf::Ability::remove_used_miracles)p.amount=2;
        p.power=a.contains("power")?nb::cast<std::int64_t>(a["power"]):gf::attack_power(p.models,field<std::int64_t>(a,"base_override",0));
        p.hit_rate=base.hit_rate;p.counter=field<bool>(a,"counter",false);p.death=field<bool>(a,"death",false);
        p.reflections=field<std::size_t>(a,"reflections",0);p.bounces=field<std::size_t>(a,"bounces",0);
        p.environmental=!p.source;p.guardian=official_enum<gf::Guardian>(field<std::string>(a,"guardian","none"),11,gf::guardian_name);
        p.forced_hit=nb::cast<std::optional<bool>>(a["forced_hit"]);v.pending=p;
    }
    if(d.contains("purchase") && !d["purchase"].is_none()) {
        auto a=nb::cast<nb::dict>(d["purchase"]);auto model=nb::cast<gf::CardId>(a["model"]);
        if(!gf::find_card(model))throw std::invalid_argument("unknown public offer");
        v.purchase=gf::PurchaseOfferView{nb::cast<gf::PlayerId>(a["origin"]),nb::cast<gf::PlayerId>(a["buyer"]),nb::cast<gf::PlayerId>(a["seller"]),model,gf::find_card(model)->price};
    }
    if(d.contains("removal")) {
        auto a=nb::cast<nb::dict>(d["removal"]);
        v.last_removal=gf::RemovalEvent{nb::cast<gf::PlayerId>(a["origin"]),nb::cast<gf::PlayerId>(a["source"]),
            nb::cast<gf::PlayerId>(a["target"]),nb::cast<gf::CardId>(a["model"]),
            nb::cast<std::vector<gf::CardId>>(a["models"])};
    }
    if(d.contains("sale")) {
        auto a=nb::cast<nb::dict>(d["sale"]);
        v.last_sale=gf::SaleEvent{nb::cast<gf::PlayerId>(a["origin"]),nb::cast<gf::PlayerId>(a["source"]),
            nb::cast<gf::PlayerId>(a["target"]),nb::cast<gf::CardId>(a["model"]),
            nb::cast<int>(a["price"]),nb::cast<std::optional<gf::CardId>>(a["overflow_model"])};
    }
    if(d.contains("guardian_changes")) {
        for(auto obj:nb::cast<nb::list>(d["guardian_changes"])) {
            auto e=nb::cast<nb::dict>(obj);
            if(nb::cast<std::string>(e["reason"])!="departure")throw std::invalid_argument("unknown guardian change reason");
            v.last_guardian_changes.push_back({nb::cast<gf::PlayerId>(e["player"]),
                official_enum<gf::Guardian>(nb::cast<std::string>(e["before"]),11,gf::guardian_name),
                official_enum<gf::Guardian>(nb::cast<std::string>(e["after"]),11,gf::guardian_name),gf::GuardianChangeReason::departure});
        }
    }
    return v;
}
gf::PublicDecision official_decision(nb::dict a) {
    gf::PublicDecision d;d.decision=nb::cast<std::uint64_t>(a["decision"]);d.gf=nb::cast<std::int64_t>(a["gf"]);
    d.phase=static_cast<gf::MatchPhase>(nb::cast<int>(a["phase"]));d.source=static_cast<gf::DecisionSource>(field<int>(a,"source",0));
    d.actor=nb::cast<std::optional<gf::PlayerId>>(a["actor"]);d.target=nb::cast<std::optional<gf::PlayerId>>(a["target"]);
    if(!a["kind"].is_none())d.kind=static_cast<gf::CommandKind>(nb::cast<int>(a["kind"]));
    if(a.contains("outcome_model"))d.outcome_model=nb::cast<std::optional<gf::CardId>>(a["outcome_model"]);
    d.faces=nb::cast<std::vector<gf::CardId>>(a["faces"]);d.faces_hidden=field<bool>(a,"faces_hidden",false);
    if(a.contains("allocation") && !a["allocation"].is_none()) {auto x=nb::cast<nb::dict>(a["allocation"]);d.allocation=gf::ResourceAllocation{nb::cast<int>(x["hp"]),nb::cast<int>(x["mp"]),nb::cast<int>(x["cp"])};}
    return d;
}
History official_history(nb::list events,std::size_t k) {
    History h(k);
    for(auto obj:events) {auto e=nb::cast<nb::dict>(obj);auto b=official_view(nb::cast<nb::dict>(e["before"]));auto a=official_view(nb::cast<nb::dict>(e["after"]));
        h.append(b,a,official_decision(nb::cast<nb::dict>(e["decision"])),*a.board.viewer);}
    return h;
}
nb::dict official_observation(nb::dict d,nb::list events,std::size_t k) {
    const auto v=official_view(d);std::vector<gf::ItemToken> order;
    for(const auto& h:v.board.hand)order.push_back(h.token);
    auto ts=encode(v,*v.board.viewer,order);ts.resize(history_offset+k);auto h=official_history(events,k);
    std::size_t i=history_offset;for(const auto& e:h.events()) {ts[i]=e.token;ts[i++].num[Age]=k?float(v.decisions-e.decision)/k:0;}
    nb::list type,aux,valid,cards,cat,num;
    for(const auto& t:ts) {type.append(t.type);aux.append(t.aux);valid.append(t.type!=0);cards.append(nb::cast(t.cards));cat.append(nb::cast(t.cat));num.append(nb::cast(t.num));}
    nb::dict out;out["obs_type"]=type;out["obs_aux_card"]=aux;out["obs_valid"]=valid;out["obs_cards"]=cards;out["obs_cat"]=cat;out["obs_num"]=num;return out;
}
struct OfficialCursor {
    gf::PublicSelection position;Selection selection;
    static SelectionExclusions restrictions(nb::dict d) {
        SelectionExclusions x;
        x.kinds=field<std::vector<int>>(d,"excluded_kinds",{});
        x.enemy_default_self=field<bool>(d,"exclude_enemy_default_self",false);
        for(auto slot:field<std::vector<int>>(d,"excluded_slots",{}))
            x.tokens.push_back(static_cast<gf::ItemToken>(slot+1));
        return x;
    }
    explicit OfficialCursor(nb::dict d):position(official_view(d)),selection(position,true,restrictions(d)) {advance();}
    void advance() {
        while(!selection.complete && std::count(selection.mask.begin(),selection.mask.end(),true)==1)
            selection.choose(position,std::find(selection.mask.begin(),selection.mask.end(),true)-selection.mask.begin(),true);
    }
    void choose(int choice) {++selection.substep;selection.choose(position,choice,true);advance();}
    nb::dict state() const {
        const auto& s=selection;nb::dict d;
        d["complete"]=s.complete;d["stage"]=s.complete?-1:int(s.stage);d["mask"]=nb::cast(s.mask);d["substep"]=s.substep;
        d["actor_seat"]=s.actor_seat;d["chosen_kind"]=s.kind_chosen?int(s.command.kind):-1;d["chosen_cards"]=nb::cast(s.cards);d["chosen_count"]=s.count;
        d["chosen_target"]=s.stage>=Stage::alloc_hp?(s.command.target?(*s.command.target==s.command.actor?1:2):0):-1;
        d["chosen_alloc_hp"]=s.stage==Stage::alloc_mp?(s.command.allocation->hp==99?20:s.command.allocation->hp/5):-1;
        return d;
    }
    nb::dict command() const {
        if(!selection.complete)throw std::invalid_argument("unfinished public selection");
        const auto& c=selection.command;nb::dict d;d["kind"]=int(c.kind);d["target"]=nb::cast(c.target);
        nb::list slots;for(auto token:c.items)slots.append(token-1);d["slots"]=slots;d["allocation"]=nb::none();
        if(c.allocation){nb::dict a;a["hp"]=c.allocation->hp;a["mp"]=c.allocation->mp;a["cp"]=c.allocation->cp;d["allocation"]=a;}return d;
    }
};
}

namespace gfrl {
struct OfficialAccess {
    static std::unique_ptr<VecEnv> root(nb::dict public_view,nb::list events,std::size_t k,nb::dict restrictions) {
        auto v=official_view(public_view);
        if(!v.board.viewer || v.actor!=v.board.viewer)
            throw std::invalid_argument("search_requires_viewer_root");
        gf::PublicSelection selection(v);auto state=selection.snapshot().state;
        auto out=std::make_unique<VecEnv>(1,1,0,true,gf::MatchConfig{},k,
            std::array<Controller,2>{Controller::external,Controller::external},std::nullopt,Controller::external,0,true);
        auto& e=out->envs_[0];
        if(v.phase==gf::MatchPhase::main) {
            e.match.load_scene(std::move(state),v.owner,0);
        } else {
            if(v.phase!=gf::MatchPhase::response || !public_view.contains("_search_launch"))
                throw std::invalid_argument("search_requires_public_launch_journal");
            const auto proof=nb::cast<nb::dict>(public_view["_search_launch"]);
            const auto before=official_view(nb::cast<nb::dict>(proof["before"]));
            const auto models=nb::cast<std::vector<gf::CardId>>(proof["models"]);
            if(before.phase!=gf::MatchPhase::main || before.owner!=v.owner || before.owner==*v.board.viewer
                    || before.actor!=before.owner || models.empty() || !v.pending
                    || v.pending->models!=models || v.pending->counter || v.pending->death
                    || v.pending->bounces || v.pending->guardian!=gf::Guardian::none)
                throw std::invalid_argument("unsupported_public_launch");
            state.gf=before.board.gf;
            const auto responses=field<nb::list>(proof,"responses",nb::list());
            gf::ItemToken next=100;
            gf::Command attack;attack.actor=v.owner;attack.kind=gf::CommandKind::play;
            attack.target=nb::cast<gf::PlayerId>(proof["target"]);
            // Percentage commands select their sole enemy internally. The wire
            // publishes that resolved recipient, not a manual command target.
            if(gf::find_card(models.front())->hit_rate)attack.target.reset();
            for(auto& player:state.players) {
                const auto old=std::find_if(before.board.players.begin(),before.board.players.end(),[&](const auto& p){return p.id==player.id;});
                if(old==before.board.players.end() || !old->status || old->fogged || old->status->confused)
                    throw std::invalid_argument("hidden_launch_status");
                const auto& pane=*old->status;
                player.hp=pane.hp;player.mp=pane.mp;player.cp=pane.cp;player.curses=pane.curses;
                player.guardian=pane.guardian;player.dead=old->dead;
                if(player.id==*v.board.viewer) {
                    player.items.clear();
                    for(const auto& h:before.board.hand)
                        player.items.push_back({h.token,h.face,h.disguised?std::optional<gf::CardId>{h.face}:std::nullopt,h.used});
                } else if(player.id==v.owner) {
                    player.items.clear();
                    for(auto model:old->used_models)player.items.push_back({next++,model,{},true});
                    for(auto model:models) {
                        // Reusable public miracles may already be used at launch.
                        auto used=std::find_if(player.items.begin(),player.items.end(),[&](const auto& h){
                            return h.model==model && std::find(attack.items.begin(),attack.items.end(),h.token)==attack.items.end();});
                        if(used!=player.items.end())attack.items.push_back(used->token);
                        else {attack.items.push_back(next);player.items.push_back({next++,model,{},false});}
                    }
                }
            }
            // Opponent response cards are now public too. Add only those
            // selected copies to the launch inventory, never unused wire truth.
            auto& opponent=*std::find_if(state.players.begin(),state.players.end(),[&](const auto& p){return p.id==v.owner;});
            for(auto obj:responses) {
                const auto response=nb::cast<nb::dict>(obj);
                if(nb::cast<gf::PlayerId>(response["actor"])==v.owner)
                    for(auto model:nb::cast<std::vector<gf::CardId>>(response["models"]))
                        opponent.items.push_back({next++,model,{},false});
            }
            bool found=false;
            // Pending existence proves a percentage hit. HP gamble also exposes
            // its sign. Select a fresh synthetic stream on these public facts;
            // never transplant an unvalidated batch or the server's private RNG.
            for(std::uint64_t seed=0;seed<256 && !found;++seed) {
                e.match.load_scene(state,v.owner,seed);
                auto initial=e.match.snapshot();initial.decisions=before.decisions;e.match.restore(initial);
                e.match.step(attack);
                if(e.match.snapshot().phase!=gf::MatchPhase::response)continue;
                for(auto obj:responses) {
                    const auto response=nb::cast<nb::dict>(obj);
                    gf::Command command;command.actor=nb::cast<gf::PlayerId>(response["actor"]);
                    command.kind=gf::CommandKind::defend;
                    const auto& current=e.match.snapshot().state.players;
                    const auto player=std::find_if(current.begin(),current.end(),[&](const auto& p){return p.id==command.actor;});
                    if(player==current.end())throw std::invalid_argument("unknown_public_responder");
                    for(auto model:nb::cast<std::vector<gf::CardId>>(response["models"])) {
                        const auto item=std::find_if(player->items.begin(),player->items.end(),[&](const auto& h){
                            return h.fake_model.value_or(h.model)==model
                                && std::find(command.items.begin(),command.items.end(),h.token)==command.items.end();});
                        if(item==player->items.end())throw std::invalid_argument("missing_public_response_card");
                        command.items.push_back(item->token);
                    }
                    e.match.step(command);
                }
                auto replay=e.match.view(*v.board.viewer);
                found=replay.phase==gf::MatchPhase::response && replay.actor==v.actor && replay.pending
                    && replay.pending->models==v.pending->models && replay.pending->target==v.pending->target
                    && replay.pending->power==v.pending->power && replay.pending->amount==v.pending->amount;
            }
            if(!found)throw std::invalid_argument("public_launch_replay_mismatch");
        }
        auto snap=e.match.snapshot();snap.decisions=v.decisions;
        // The response journal's counter must agree with its launch anchor.
        if(v.phase==gf::MatchPhase::main)e.match.restore(snap);
        else if(e.match.snapshot().decisions!=v.decisions)throw std::invalid_argument("public_launch_decision_mismatch");
        for(int seat=0;seat<2;++seat) {
            e.history[seat]=History(k);
            if(snap.state.players[seat].id==*v.board.viewer)e.history[seat]=official_history(events,k);
        }
        e.selection.emplace(e.match,true,OfficialCursor::restrictions(restrictions),
            std::make_shared<const gf::PublicSelection>(v));e.encoded=false;e.waiting=false;
        // A fully forced command must not execute while constructing a root.
        while(!e.selection->complete && std::count(e.selection->mask.begin(),e.selection->mask.end(),true)==1)
            e.selection->choose(e.match,std::find(e.selection->mask.begin(),e.selection->mask.end(),true)-e.selection->mask.begin(),true);
        if(e.selection->complete)throw std::invalid_argument("search_forced_root");
        out->publish(0);return out;
    }
};
}
