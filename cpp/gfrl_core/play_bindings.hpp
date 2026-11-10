#pragma once
// Terminal presentation allowlist. No snapshots, RNGs or item tokens cross it.
#include <nanobind/stl/vector.h>
namespace {
nb::dict play_item(const gf::HandItemView& h) {
    nb::dict d;d["face"]=h.face;d["used"]=h.used;d["disguised"]=h.disguised;return d;
}
nb::dict play_board(const gf::MatchView& v) {
    nb::dict d;
    const char* phases[]={"main","response","finished","purchase","grant"};
    d["phase"]=phases[int(v.phase)];d["actor"]=nb::cast(v.actor);
    d["decisions"]=v.decisions;d["gf"]=v.board.gf;
    d["terminated"]=v.terminated;d["truncated"]=v.truncated;
    nb::list players,hand;
    for(const auto& p:v.board.players) {
        nb::dict x;x["id"]=p.id;x["dead"]=p.dead;x["fogged"]=p.fogged;
        x["used"]=nb::cast(p.used_models);
        x["status"]=nb::none();
        if(p.status) {
            const auto& s=*p.status;nb::dict status;
            status["hp"]=s.hp;status["mp"]=s.mp;status["cp"]=s.cp;
            status["guardian"]=gf::guardian_name(s.guardian).data();
            status["confused"]=s.confused;nb::list curses;
            for(auto c:s.curses)curses.append(nb::str(gf::curse_name(c).data()));
            status["curses"]=curses;x["status"]=status;
        }
        players.append(x);
    }
    for(const auto& h:v.board.hand)hand.append(play_item(h));
    d["players"]=players;d["hand"]=hand;d["pending"]=nb::none();d["purchase"]=nb::none();
    if(v.pending) {
        const auto& p=*v.pending;nb::dict x;
        x["origin"]=nb::cast(p.origin);x["source"]=nb::cast(p.source);x["target"]=p.target;
        x["models"]=nb::cast(p.models);x["power"]=p.power;x["amount"]=p.amount;
        x["category"]=gf::category_name(p.category).data();
        x["element"]=p.element ? nb::cast(gf::element_name(*p.element).data()) : nb::none();
        x["ability"]=gf::ability_name(p.ability).data();x["hit_rate"]=p.hit_rate;
        x["reflections"]=p.reflections;x["bounces"]=p.bounces;d["pending"]=x;
    }
    if(v.purchase) {
        const auto& p=*v.purchase;nb::dict x;
        x["buyer"]=p.buyer;x["seller"]=p.seller;x["model"]=p.model;x["price"]=p.price;d["purchase"]=x;
    }
    nb::list reveals,guardians;
    for(const auto& r:v.last_reveals) {nb::dict x;x["owner"]=r.owner;x["face"]=r.face;x["model"]=r.model;reveals.append(x);}
    for(const auto& g:v.last_guardian_changes) {nb::dict x;x["player"]=g.player;x["before"]=gf::guardian_name(g.before).data();x["after"]=gf::guardian_name(g.after).data();guardians.append(x);}
    d["removal"]=nb::none();d["sale"]=nb::none();d["purchase_result"]=nb::none();
    if(v.last_removal) {
        const auto& r=*v.last_removal;nb::dict x;x["target"]=r.target;x["models"]=nb::cast(r.removed_models);d["removal"]=x;
    }
    if(v.last_sale) {
        const auto& r=*v.last_sale;nb::dict x;x["source"]=r.source;x["target"]=r.target;
        x["model"]=r.model;x["price"]=r.price;x["overflow_model"]=nb::cast(r.overflow_model);d["sale"]=x;
    }
    if(v.last_purchase) {
        const auto& r=*v.last_purchase;nb::dict x;x["buyer"]=r.buyer;x["seller"]=r.seller;
        x["model"]=r.model;x["price"]=r.price;x["reason"]=int(r.reason);
        x["overflow_model"]=nb::cast(r.overflow_model);d["purchase_result"]=x;
    }
    d["reveals"]=reveals;d["guardian_changes"]=guardians;
    return d;
}
nb::dict play_decision(const gf::PublicDecision& v) {
    nb::dict d;d["decision"]=v.decision;d["actor"]=nb::cast(v.actor);
    d["kind"]=v.kind ? nb::cast(int(*v.kind)) : nb::none();
    d["source"]=int(v.source);d["faces"]=nb::cast(v.faces);d["faces_hidden"]=v.faces_hidden;
    d["target"]=nb::cast(v.target);d["outcome_model"]=nb::cast(v.outcome_model);
    d["allocation"]=nb::none();
    if(v.allocation) {nb::dict a;a["hp"]=v.allocation->hp;a["mp"]=v.allocation->mp;a["cp"]=v.allocation->cp;d["allocation"]=a;}
    return d;
}
}
