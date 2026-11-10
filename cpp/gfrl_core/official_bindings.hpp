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
        p.ability=base.ability;p.amount=base.ability_value;
        if(base.ability==gf::Ability::sell && p.models.size()>1)p.amount=gf::find_card(p.models[1])->price;
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
    explicit OfficialCursor(nb::dict d):position(official_view(d)),selection(position) {advance();}
    void advance() {while(!selection.complete && std::count(selection.mask.begin(),selection.mask.end(),true)==1)
        selection.choose(position,std::find(selection.mask.begin(),selection.mask.end(),true)-selection.mask.begin(),true);}
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
    static std::unique_ptr<VecEnv> root(nb::dict public_view,nb::list events,std::size_t k) {
        auto v=official_view(public_view);
        if(v.phase!=gf::MatchPhase::main || !v.board.viewer || v.actor!=v.board.viewer)
            throw std::invalid_argument("search_requires_main_root");
        gf::PublicSelection selection(v);auto state=selection.snapshot().state;
        auto out=std::make_unique<VecEnv>(1,1,0,true,gf::MatchConfig{},k,
            std::array<Controller,2>{Controller::external,Controller::external},std::nullopt,Controller::external,0,true);
        auto& e=out->envs_[0];e.match.load_scene(std::move(state),v.owner,0);
        auto snap=e.match.snapshot();snap.decisions=v.decisions;e.match.restore(snap);
        for(int seat=0;seat<2;++seat) {
            e.history[seat]=History(k);
            if(snap.state.players[seat].id==*v.board.viewer)e.history[seat]=official_history(events,k);
        }
        e.selection.emplace(e.match,true);e.encoded=false;e.waiting=false;
        // A fully forced command must not execute while constructing a root.
        while(!e.selection->complete && std::count(e.selection->mask.begin(),e.selection->mask.end(),true)==1)
            e.selection->choose(e.match,std::find(e.selection->mask.begin(),e.selection->mask.end(),true)-e.selection->mask.begin(),true);
        if(e.selection->complete)throw std::invalid_argument("search_forced_root");
        out->publish(0);return out;
    }
};
}
