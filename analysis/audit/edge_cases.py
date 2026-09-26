import math

from lalib import la

C = la.COLUMNS
print("n columns:", len(C))
# toy statistics: (log ratio vs base, log1p df1, df23)
toy = {
    "services": (math.log(3.0), math.log1p(500), 5000),
    "groupe": (math.log(2.0), math.log1p(300), 2000),
    "amis": (math.log(0.85), math.log1p(400), 1500),
    "collectif": (math.log(0.85), math.log1p(300), 1200),
    "de": (math.log(0.9), math.log1p(40000), 150000),
    "du": (math.log(0.9), math.log1p(20000), 80000),
    "la": (math.log(0.9), math.log1p(30000), 110000),
    "tata": (math.log(0.9), math.log1p(300), 1100),
    "consultancy": (math.log(0.9), math.log1p(3000), 11000),
    "sky": (math.log(0.9), math.log1p(100), 400),
    "marie": (math.log(0.9), math.log1p(500), 1800),
    "coiffure": (math.log(0.9), math.log1p(900), 3500),
    "boulangerie": (math.log(0.9), math.log1p(900), 3500),
    "bar": (math.log(0.9), math.log1p(900), 3500),
}
la._init({"X": toy})
cases = [
    ("", "abc def"),
    ("sas", "mauges sas"),                        # S1 name only a legal form
    ("a b c", "abc"),                             # single letters only
    ("mauges amis sas", "mauges collectif sas"),  # the planted look-alike
    ("mauges amis sas", "mauges amis groupe sas"),  # filler added
    ("pediatric clinic inc", "pediatric inc services"),
    ("tata consultancy services ltd", "tcs ltd"),  # acronym (S1 long, record short)
    ("tcs ltd", "tata consultancy services ltd"),  # acronym the other way round
    ("blue sky travels", "blueskytravels"),       # glued / domain-style
    ("blueskytravels", "blue sky travels"),
    ("salon coiffure marie", "salon de coiffure"),  # French article added, first name dropped
    ("boulangerie du centre", "boulangerie centre"),
    ("bar du port", "boulangerie du port"),       # genuine swap, but "bar" is a subsequence of "boulangerie"
    ("latelier", "atelier"),                      # l' elision glued by normalization
    ("123 456", "123 789"),                       # digit tokens
]
for a, b in cases:
    r = la.pair_row(a, b, "X")
    assert len(r) == len(C), (a, b, len(r))
    d = {k: (round(v, 3) if v == v else "nan") for k, v in zip(C, r)}
    short = {k.replace("la_", ""): v for k, v in d.items() if v not in (0.0, "nan")}
    print(f"{a!r:36} -> {b!r:34} {short}")
print("close('bar','boulangerie') =", la._close("bar", "boulangerie"))
print("close('art','atelier') =", la._close("art", "atelier"))
print("close('amis','ami') =", la._close("amis", "ami"))
print("name_tokens('la boulangerie de la gare sarl') =", la.name_tokens("la boulangerie de la gare sarl"))
print("unknown country ->", la.pair_row("mauges amis", "mauges collectif", "Nowhere"))
