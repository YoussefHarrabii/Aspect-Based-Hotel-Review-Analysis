"""Example category rules applied after the classifier thresholds.

These encode label conventions the classifier does not learn. They are examples:
a user replacing the bucket set should also review this ruleset.

  R1 topic rescue    a room noun is present, Room sits just under its bar, and
                     nothing else was predicted -> accept Room.
  R2 housekeeping    a room noun + room-condition vocabulary while Staff is
                     predicted -> accept Room.
  water -> Room      the sentence says "water", Food is predicted, and the
                     sentence is close to a bathroom vocabulary -> Food off,
                     Room on.

    python app/pipeline/cat_rules.py        # self-check
"""
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
CATS = json.load(open(os.path.join(ROOT, "config", "taxonomy_v5.json"), encoding="utf-8"))["categories"]
RI = CATS.index("Room & Property")
FI = CATS.index("Food & Drink")
SI = CATS.index("Staff & Service")

ROOM_NOUN = re.compile(
    r"\b(room|rooms|bedroom|bedrooms|bathroom|bathrooms|shower|toilet|bed|balcony|balconies|"
    r"air condition(?:ing|er)|a/c\b|minibar|mini bar|wardrobe|television|\btv\b|kettle|hairdryer|"
    r"socket|mattress|bath\b|washbasin|gym|tennis court|suite)\b", re.I)
NOT_ROOM = re.compile(r"room service", re.I)
HOUSEKEEP = re.compile(
    r"\b(clean|cleaned|cleaning|cleaner|cleaners|housekeep\w*|maid|upgrade[ds]?|upgrading|"
    r"renovat\w+|refurbish\w+|redecorat\w+|decor|decors|spacious|dirty|dirt|filth|stain\w*|"
    r"mould|mold|rust|cockroach|ants|smell|smelly|noisy|leak\w*|drip\w*|broken|repair\w*|"
    r"maintenance|tired|worn)\b", re.I)

TOPIC_FLOOR = 0.05
HOUSE_FLOOR = 0.02
BATH_SIM = 0.945
BATHROOM = [
    "the shower", "the bathroom", "the toilet", "the sink", "the tap", "the bathtub",
    "shower water pressure", "hot water in the shower", "the drain", "the plumbing is blocked",
    "water leaking from the ceiling", "the shower is not draining", "flooding in the bathroom",
    "shampoo and shower gel", "the water temperature in the shower", "a wet bathroom floor",
]


def encode_bathroom(enc, instr):
    """Embed BATHROOM with an already-loaded SentenceTransformer (no extra model load)."""
    v = enc.encode([f"{instr}{t}" for t in BATHROOM], normalize_embeddings=True,
                   show_progress_bar=False)
    return np.asarray(v, dtype=np.float32)


def has_room_noun(text):
    return bool(ROOM_NOUN.search(text)) and not NOT_ROOM.search(text)


# Off-property Staff veto. Generic travel-operator vocabulary.
OPERATOR = re.compile(
    r"tour operator|travel agent|package holiday|airline|cruise line|excursion company", re.I)
OFFPROP = re.compile(
    r"\breps?\b|\bdrivers?\b|\btaxis?\b|\bairport\b|\btransfers?\b|\btouts?\b|"
    r"shopkeep|haggl|hassl|excursion|\bcoach\b", re.I)
STAFF_NOUN = re.compile(
    r"reception|receptionist|wait(er|ress)|maid|housekeep|cleaner|cleaning staff|manager|"
    r"chef|bartender|barman|animation|lifeguard|chambermaid|concierge|bellboy|front desk", re.I)


def staff_veto(text):
    """True if Staff & Service should be suppressed: an off-property person, not hotel staff."""
    if OPERATOR.search(text):
        return True
    return bool(OFFPROP.search(text)) and not STAFF_NOUN.search(text)


def veto_staff(pred, texts):
    """q = pred with Staff removed on off-property sentences."""
    q = pred.copy()
    for i, t in enumerate(texts):
        if q[i, SI] and staff_veto(t):
            q[i, SI] = False
    return q


def apply_rules(pred, probs, texts):
    """pred: (N, len(CATS)) bool, probs: (N, len(CATS)) float, texts: [str]. Returns a copy.

    Only ever turns Room & Property on. No other category is added, removed or reordered.
    """
    q = pred.copy()
    for i, t in enumerate(texts):
        if pred[i, RI] or not has_room_noun(t):
            continue
        other = pred[i].copy()
        other[RI] = False
        if not other.any() and probs[i, RI] >= TOPIC_FLOOR:
            q[i, RI] = True
        elif pred[i, SI] and probs[i, RI] >= HOUSE_FLOOR and HOUSEKEEP.search(t):
            q[i, RI] = True
    return q


def water_to_room(pred, unit_emb, texts, bath_emb, sim_floor=BATH_SIM):
    """'water' + Food predicted + close to a bathroom vocabulary -> Food off, Room on."""
    q = pred.copy()
    if not len(texts):
        return q
    sims = (bath_emb @ np.asarray(unit_emb, dtype=np.float32).T).max(0)
    fire = pred[:, FI] & (sims >= sim_floor)
    fire &= np.array(["water" in t.lower() for t in texts])
    q[fire, FI] = False
    q[fire, RI] = True
    return q


MANAGER_RE = re.compile(r"\b(manager|management|manageress)\b", re.I)


def manager_rule(facet, order, cat, text):
    """Staff & Service: a named manager outranks Staff Attitude.

    manager present AND top1 == Staff Attitude AND top2 == Management & Policies -> M&P
    """
    out = list(facet)
    for i, (c, text_i) in enumerate(zip(cat, text)):
        if c != "Staff & Service" or i >= len(order):
            continue
        t1, t2 = order[i]
        if t1 == "Staff Attitude" and t2 == "Management & Policies" and MANAGER_RE.search(text_i):
            out[i] = "Management & Policies"
    return out


def _selftest():
    assert has_room_noun("the girls who clean the rooms") is True
    assert has_room_noun("no air conditioning in the room") is True
    assert has_room_noun("we ordered room service") is False
    assert has_room_noun("a sunbed and blue towels by the pool") is False
    assert bool(HOUSEKEEP.search("they should upgrade rooms")) is True
    assert bool(HOUSEKEEP.search("check the shower and phone")) is False

    assert staff_veto("the rep was too busy") is True
    assert staff_veto("the tour operator rep was rude") is True
    assert staff_veto("a taxi driver yelled at me") is True
    assert staff_veto("the porter was very helpful") is False
    assert staff_veto("reception staff were helpful") is False
    n = len(CATS)
    sv = np.zeros((3, n), dtype=bool)
    sv[:, SI] = True
    sv2 = veto_staff(sv, ["the rep never called back", "the porter carried our bags",
                           "reception sorted it out"])
    assert not sv2[0, SI] and sv2[1, SI] and sv2[2, SI]
    assert np.array_equal(sv[:, SI], np.ones(3, dtype=bool)), "input must not be mutated"

    pred = np.zeros((4, n), dtype=bool)
    pred[0, SI] = True
    pred[1, SI] = True
    pred[2, 1] = True
    pred[3, SI] = True
    probs = np.zeros((4, n))
    probs[:, RI] = 0.30
    q = apply_rules(pred, probs, ["they should upgrade the rooms",
                                  "reception accused us of stealing towels from the room",
                                  "a wide range of dishes at the buffet",
                                  "reception never answered the phone"])
    assert q[0, RI], "R2 should accept Room"
    assert not q[1, RI], "adjunct room noun must not trigger"
    assert not q[2, RI], "R1 must not fire when another category is predicted"
    assert not q[3, RI], "no room noun, no change"
    assert q.sum() == pred.sum() + 1

    e = np.zeros((3, 4), dtype=np.float32)
    e[0] = [1, 0, 0, 0]
    e[1] = [0, 1, 0, 0]
    e[2] = [0.9, 0.1, 0, 0]
    b = np.array([[1, 0, 0, 0]], dtype=np.float32)
    w = np.zeros((3, len(CATS)), dtype=bool)
    w[:, FI] = True
    q = water_to_room(w, e, ["the shower water was cold", "the water is fine", "no water here"], b)
    assert not q[0, FI] and q[0, RI]
    assert q[1, FI] and not q[1, RI]
    assert q[2, FI]
    empty = water_to_room(np.zeros((0, len(CATS)), dtype=bool), np.zeros((0, 4), dtype=np.float32),
                          [], np.zeros((1, 4), dtype=np.float32))
    assert empty.shape == (0, len(CATS))

    f = ["Staff Attitude", "Staff Attitude", "Staff Attitude", "Management & Policies", "Staff Attitude"]
    o = [("Staff Attitude", "Management & Policies"), ("Staff Attitude", "Management & Policies"),
         ("Staff Attitude", "General Experience"), ("Reception & Check-in/out", "Management & Policies"),
         ("Staff Attitude", "Management & Policies")]
    c = ["Staff & Service"] * 5
    tx = ["Directions from manager with pictures were easy to follow",
          "The staff are amazing",
          "Best manager Nick sorted it out",
          "The manager told us we were safe",
          "Directions from manager with pictures were easy to follow"]
    c[4] = "Room & Property"
    g = manager_rule(f, o, c, tx)
    assert g[0] == "Management & Policies"
    assert g[1] == "Staff Attitude"
    assert g[2] == "Staff Attitude"
    assert g[3] == "Management & Policies"
    assert g[4] == "Staff Attitude"
    print("cat_rules self-check ok")


if __name__ == "__main__":
    _selftest()
