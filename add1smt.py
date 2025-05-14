# add1smt.py  -----------------------------------------------------------
from z3 import *

# ---------- 1. Declare variables ----------
fp32 = Float32()
a, b = FP('a', fp32), FP('b', fp32)
rm = RNE()                        # Rounding mode: round-to-nearest-even

# ---------- 2. Define constants ----------
FLT_MAX = FPVal(3.402823466e38, fp32)   # Maximum finite value for float32
ZERO = FPVal(0.0, fp32)

# ---------- 3. Translate if / else logic ----------
# cond1: a > 0 && b > 0 && a > FLT_MAX - b
cond1 = And(fpGT(a, ZERO),
            fpGT(b, ZERO),
            fpGT(a, fpSub(rm, FLT_MAX, b)))

# cond2: a < 0 && b < 0 && a < -FLT_MAX - b
cond2 = And(fpLT(a, ZERO),
            fpLT(b, ZERO),
            fpLT(a, fpSub(rm, fpNeg(FLT_MAX), b)))

# c = (cond1 ? 0 : (cond2 ? 0 : a + b))
c = If(cond1,
       ZERO,
       If(cond2,
          ZERO,
          fpAdd(rm, a, b)))

# ---------- 4. Input constraints (optional: exclude NaN / Inf inputs) ----------
safe_in = [Not(fpIsInf(a)), Not(fpIsNaN(a)),
           Not(fpIsInf(b)), Not(fpIsNaN(b))]

# ---------- 5. Build solver ----------
goal = Or(fpIsInf(c), fpIsNaN(c))
s = Solver()
s.add(safe_in)   # Optional: constrain a, b to be normal numbers
s.add(goal)

print("Checking addition overflow ...")
if s.check() == sat:
    m = s.model()
    print("> Inf/NaN may occur!")
    print("  a =", m[a])
    print("  b =", m[b])
    print("  c =", m.eval(c, model_completion=True))
else:
    print("> Proven: no Inf/NaN can occur (UNSAT)")
