"""Last comma-separated address component of France records (raw), S1 vs S2/S3: region/department coverage."""
from collections import Counter

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from lalib import ROOT, la  # noqa: F401  (sets up sys.path without writing bytecode)
from src.normalize import STATES

opts = dict(parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
            convert_options=pacsv.ConvertOptions(include_columns=["business_address", "country"],
                                                 column_types={"business_address": pa.string(),
                                                               "country": pa.string()}))
for n in (1, 2, 3):
    t = pacsv.read_csv(fr"{ROOT}\student_resource\dataset\test\test_source{n}.tsv", **opts)
    t = t.filter(pc.equal(t.column("country"), "France"))
    vc = Counter(a.rsplit(",", 1)[-1].strip().lower().replace("-", " ")
                 for a in t.column("business_address").to_pylist())
    mapped = sum(v for k, v in vc.items() if k in STATES)
    print(f"source{n}: France rows {t.num_rows:,}; distinct last components {len(vc):,}; "
          f"share whose last component is in STATES {mapped / t.num_rows:.3f}")
    print("   top:", vc.most_common(12))
    del t
