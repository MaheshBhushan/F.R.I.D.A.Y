"""The two wake phrases and, more importantly, what each must NOT fire on.

Adversarial negatives are the whole reason a wake word survives contact with
real speech. Without them the model learns "roughly friday-shaped noise" and
fires on "birthday". With them it has to find the actual boundary.

The lists are hand-built rather than generated because one mistake here is
poisonous in a specific way: putting a phrase that CONTAINS the wake word into
the negatives ("see you friday") teaches the model to reject the very thing it
exists to detect. Every entry below is deliberately near-miss-but-different.
"""

TARGETS = {
    "friday": {
        "phrase": "friday",
        # Near-rhymes and words sharing the "fr-" onset or "-day" coda. These
        # are the sounds that actually cost false accepts in a room where
        # someone is talking about calendars.
        "adversarial": [
            "fry day", "free day", "freddy", "ready", "already", "birthday",
            "my day", "the day", "someday", "good day", "pay day", "friend",
            "fridge", "afraid", "frighten", "friendly", "fried egg", "priday",
            "thursday", "saturday", "sunday", "monday", "tuesday", "wednesday",
            "holiday", "everyday", "yesterday", "midday", "fritter", "frida",
        ],
    },
    "hey_friday": {
        "phrase": "hey friday",
        # "friday" alone belongs here: with both models loaded the bare word is
        # the OTHER model's job, and teaching this one to ignore it is what
        # keeps the two from firing together on every utterance.
        "adversarial": [
            "friday", "hey fred", "hey freddy", "hey ready", "hey brady",
            "a friday", "they friday", "hey friend", "hey there", "okay friday",
            "hey frank", "hey fridge", "hey day", "hey birthday", "on friday",
            "hey good day", "hey", "hello friday", "say friday", "eight friday",
            "hey free day", "hey fry day", "made friday", "hey monday",
            "hey thursday", "hey holiday", "hey frida", "hey afraid",
        ],
    },
}

# Every phrase the generator has to synthesise, deduplicated across targets so
# a shared adversarial ("hey frida" vs "frida") is not paid for twice.
def all_phrases() -> dict:
    out = {}
    for name, spec in TARGETS.items():
        out.setdefault(spec["phrase"], set()).add(f"{name}:positive")
        for adv in spec["adversarial"]:
            out.setdefault(adv, set()).add(f"{name}:adversarial")
    return out


if __name__ == "__main__":
    phrases = all_phrases()
    print(f"{len(phrases)} distinct phrases")
    for name, spec in TARGETS.items():
        print(f"  {name}: 1 positive + {len(spec['adversarial'])} adversarial")
