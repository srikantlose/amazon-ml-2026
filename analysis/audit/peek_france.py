import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lalib import CACHE

a = np.load("sub10_pattern_France.npy")
rng = np.random.default_rng(1)
pick = rng.choice(a.shape[1], 14, replace=False)
s1i, ri = a[0][pick], a[1][pick]
t1 = pq.read_table(fr"{CACHE}\test\records_s1.parquet", columns=["name_n", "addr_n"])
t2 = pq.read_table(fr"{CACHE}\test\records_s23.parquet", columns=["name_n", "addr_n", "entity_id"])
n1 = t1.column("name_n").take(pa.array(s1i)).to_pylist(); a1 = t1.column("addr_n").take(pa.array(s1i)).to_pylist()
n2 = t2.column("name_n").take(pa.array(ri)).to_pylist(); a2 = t2.column("addr_n").take(pa.array(ri)).to_pylist()
same_addr = np.mean([x == y for x, y in zip(t1.column("addr_n").take(pa.array(a[0])).to_pylist(),
                                              t2.column("addr_n").take(pa.array(a[1])).to_pylist())])
print(f"France accepted pattern pairs with identical normalized address: {same_addr:.3f}")
for x, y, u, v in zip(n1, n2, a1, a2):
    print(f"S1 {x!r:42} | {u[:55]!r}\n   rec {y!r:38} | {v[:55]!r}")
