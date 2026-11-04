"""Generate the synthetic sample CSV.

The text in this file is generated from templates. It is not derived from any real
review, hotel or guest. The same seed always produces the same bytes.

    python scripts/make_sample_data.py
    python scripts/make_sample_data.py --rows 400 --out data/sample/reviews.csv
"""
import argparse
import csv
import os
import random
from datetime import date, timedelta


EN = {
    ("Room & Property", "positive"): [
        "The room was spotless and the bed was genuinely comfortable.",
        "Our balcony looked over the garden and the air conditioning was quiet.",
        "Housekeeping kept the bathroom fresh every day and the shower had strong pressure.",
        "The room was spacious, with a wardrobe that actually fit our luggage.",
    ],
    ("Room & Property", "negative"): [
        "The room smelled of damp and the bathroom tap dripped all night.",
        "The air conditioning was noisy and never really cooled the room.",
        "We found stains on the sheets and the carpet looked tired.",
        "The bathroom was cramped and the shower leaked onto the floor.",
    ],
    ("Room & Property", "neutral"): [
        "The room was a standard size with a small balcony.",
        "The furniture was simple and the safe needed a key from reception.",
    ],
    ("Food & Drink", "positive"): [
        "Breakfast had fresh bread, fruit and a proper omelette every morning.",
        "The buffet changed nightly and the grilled fish was excellent.",
        "The bar made a very good iced coffee and the staff remembered our order.",
        "Dinner was tasty, with plenty of vegetarian options and a good dessert table.",
    ],
    ("Food & Drink", "negative"): [
        "The buffet was repetitive and most dishes were lukewarm.",
        "Breakfast ran out of eggs and the coffee was thin and cold.",
        "The restaurant menu was limited and the food was bland.",
        "We waited forty minutes for a main course that arrived overcooked.",
    ],
    ("Food & Drink", "neutral"): [
        "The menu offered the usual hotel standards and a few local dishes.",
        "Lunch was served between noon and three, with a small snack bar in between.",
    ],
    ("Staff & Service", "positive"): [
        "The reception team was friendly and sorted our late check-out without fuss.",
        "The waiter noticed our empty glasses and brought refills before we asked.",
        "Housekeeping was kind and left extra towels when we requested them.",
        "The manager listened to a small issue and fixed it the same afternoon.",
    ],
    ("Staff & Service", "negative"): [
        "Reception was slow and no one answered the phone when we called.",
        "The waiter was rude when we asked about the vegetarian option.",
        "We reported a broken lamp twice and nobody came to fix it.",
        "Check-in took almost an hour and the desk staff seemed overwhelmed.",
    ],
    ("Staff & Service", "neutral"): [
        "Check-in was straightforward and the desk explained the meal times.",
        "The staff spoke good English.",
    ],
    ("Pool & Beach", "positive"): [
        "The pool was clean and warm, with enough sunbeds in the morning.",
        "The beach was a short walk away and the water was clear.",
        "The pool bar served cold drinks and the towels were changed daily.",
        "We spent every afternoon by the pool and never struggled to find shade.",
    ],
    ("Pool & Beach", "negative"): [
        "The pool was crowded and the water looked cloudy.",
        "The beach had seaweed and the sunbeds were all reserved by breakfast.",
        "The pool area was noisy and the showers were out of order.",
        "There were not enough umbrellas and the deck was uncomfortably hot.",
    ],
    ("Pool & Beach", "neutral"): [
        "The pool is open from morning until early evening.",
        "The beach is across the road, with a small path to the sand.",
    ],
    ("Spa & Wellness", "positive"): [
        "The massage was excellent and the spa felt calm and clean.",
        "The steam room was a good way to end the day and the staff were attentive.",
        "The spa package was good value and the therapist explained every step.",
    ],
    ("Spa & Wellness", "negative"): [
        "The spa was overpriced and the treatment room felt cramped.",
        "The sauna was out of order and nobody told us before we booked.",
        "The massage was rushed and the therapist kept checking the clock.",
    ],
    ("Spa & Wellness", "neutral"): [
        "The spa offers massages and a steam room by appointment.",
        "Treatments are charged separately from the room rate.",
    ],
    ("Entertainment & Animation", "positive"): [
        "The evening show was surprisingly good and the animation team worked hard.",
        "There were aqua gym classes and a kids club that our children loved.",
        "The live music on the terrace made the evening feel special.",
    ],
    ("Entertainment & Animation", "negative"): [
        "The evening entertainment was loud and the same songs played every night.",
        "The kids club was closed for two days without any notice.",
        "The animation team kept pushing us to join games we did not want.",
    ],
    ("Entertainment & Animation", "neutral"): [
        "There is a small animation programme during the day and a show at night.",
        "The kids club runs in the morning and again after lunch.",
    ],
    ("Location", "positive"): [
        "The hotel is close to the beach and a short taxi ride from the old town.",
        "We could walk to the market and the harbour in under twenty minutes.",
    ],
    ("Location", "negative"): [
        "The hotel is far from the centre and the road outside is busy.",
        "There is nothing within walking distance except a small shop.",
    ],
    ("Location", "neutral"): [
        "The hotel sits between the beach road and the main street.",
        "A taxi to the centre takes about fifteen minutes.",
    ],
}

OPENERS = {
    "en": {
        "positive": ["We had a lovely stay overall.", "This was a good choice for our holiday.", "We would happily come back."],
        "neutral": ["The stay was fine, with a few ups and downs.", "It did the job for a short break."],
        "negative": ["Unfortunately the stay did not meet expectations.", "We left disappointed.", "There were too many small problems."],
    },
}

CATS = list(dict.fromkeys(k[0] for k in EN))


def _rating(rng):
    return rng.choices([1, 2, 3, 4, 5], weights=[12, 14, 20, 30, 24], k=1)[0]


def _polarity(rating, rng):
    if rating <= 2:
        return "negative" if rng.random() < 0.85 else "neutral"
    if rating == 3:
        return rng.choice(["negative", "neutral", "positive"])
    return "positive" if rng.random() < 0.85 else "neutral"


def _sentence(cat, pol, rng):
    return rng.choice(EN[(cat, pol)])


def _review(rng, rating):
    pol = _polarity(rating, rng)
    n = rng.randint(2, 6)
    out = [rng.choice(OPENERS["en"][pol])]
    cats = CATS[:]
    rng.shuffle(cats)
    for i in range(n - 1):
        cat = cats[i % len(cats)]
        p = pol
        if rating == 3 and i > 0:
            p = rng.choice(["positive", "negative", "neutral"])
        elif rating >= 4 and i == n - 2 and rng.random() < 0.25:
            p = "neutral"
        elif rating <= 2 and i == n - 2 and rng.random() < 0.2:
            p = "neutral"
        out.append(_sentence(cat, p, rng))
    return " ".join(out)


def build_rows(rows=400, seed=20261104):
    rng = random.Random(seed)
    out = []
    for i in range(rows):
        rating = _rating(rng)
        day = rng.randint(0, 700)
        d = date(2024, 11, 1) + timedelta(days=day)
        out.append({
            "review_id": f"r-{i:04d}",
            "date": d.isoformat(),
            "rating": rating,
            "text": _review(rng, rating),
        })
    return out


def write_csv(path, rows=400, seed=20261104):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = build_rows(rows, seed)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["review_id", "date", "rating", "text"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(data)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("data", "sample", "reviews.csv"))
    ap.add_argument("--rows", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261104)
    a = ap.parse_args()
    data = write_csv(a.out, a.rows, a.seed)
    print(f"wrote {a.out}: {len(data)} rows")


if __name__ == "__main__":
    main()

