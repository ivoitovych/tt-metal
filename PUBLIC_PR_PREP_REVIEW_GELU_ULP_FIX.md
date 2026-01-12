# Public PR Prep Review — GELU ULP Fix Branch

**Branch:** `ivoitovych/issue-35290-gelu-ulp-fix-draft-pr-prep`  
**Scope reviewed:** All files changed in this branch **except** `PR_REVIEW_GELU_ULP_FIX.md` (explicitly not opened/read).

This document is a PR-readiness review focused on: correctness risk, build/CI impact, maintainability, test strategy, and what should/shouldn’t be merged into the public repository.

---

### Executive summary

- **Core kernel changes look plausible and well-structured**, and it’s good that Wormhole + Blackhole are kept in sync.
- The branch contains **too much long-form research/progress documentation** for a typical public product PR; much of it should move to the PR description and/or the GitHub issue.
- The current tests/docs have **internal inconsistencies in the “DAZ+FTZ ULP model” definition**, and reviewers will likely challenge the metric and/or the mapping near zero.
- Public PR should **avoid introducing heavy dependencies** (MPFR/GMP/mpmath) unless the repo already expects them in CI; right now the checked-in tests *do not actually use MPFR/mpmath* (they use fp64 `erf()` with “safe” vectors).

---

### Changed files (reviewed)

- **Kernel implementation (must keep)**
  - `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h`
  - `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h`
- **C++ tests / build wiring (keep with cleanup)**
  - `tests/ttnn/unit_tests/gtests/test_gelu_ulp_bug.cpp`
  - `tests/ttnn/unit_tests/gtests/CMakeLists.txt`
- **Python tests (keep with cleanup)**
  - `tests/ttnn/unit_tests/operations/eltwise/test_gelu_floor_value_bug.py`
- **Docs / reports (recommend moving out of main tree or trimming heavily)**
  - `GELU_BF16_Zero_Saturation_Threshold_Research.md`
  - `GELU_ULP_FIX_IMPLEMENTATION.md`
  - `tests/ttnn/unit_tests/operations/eltwise/GELU_FLOOR_VALUE_BUG_REPORT.md`

---

### What looks good (recommended to keep)

- **Kernel fix structure is reviewer-friendly**
  - Clear piecewise behavior: near-zero Taylor, deep-negative tail handling, piecewise polynomials in the mid-range, and positive saturation.
  - Same logic appears in both Wormhole and Blackhole kernel headers, reducing drift risk.
  - The near-zero region explicitly avoids the “floor value” behavior described in the bug report.

- **Tests target the right failure modes**
  - Both C++ and Python tests are oriented around the historically problematic regions:
    - Deep negative tail
    - Near-zero behavior
    - Transition region behavior around segment boundaries
  - Device tests assert **bounded ULP** rather than “exact-match floats,” which is generally the right approach for approximations.

- **CMake wiring is simple**
  - Adding `test_gelu_ulp_bug.cpp` into the existing `unit_tests_ttnn_basic` target is straightforward (assuming it does not add new link-time dependencies).

---

### What is not good for a public PR (recommended to exclude or move)

- **Root-level investigation/progress docs**
  - `GELU_BF16_Zero_Saturation_Threshold_Research.md`
  - `GELU_ULP_FIX_IMPLEMENTATION.md`

  These are valuable artifacts for your work, but for a public PR they present multiple risks:
  - Read like **personal research notes + progress log** (dates, branch history, “Phase” steps, cherry-pick commands).
  - Include installation/compilation instructions for external tools (e.g., MPFR) that a repo may not want to own in-tree.
  - Duplicate information better suited for the **PR description** and the **issue thread**.

  **Recommendation:**
  - Move the detailed narrative to the **PR description / GitHub issue**.
  - If in-repo docs are required, create a **short, stable** doc under an approved location (often `docs/` or `tech_reports/`) that states only:
    - the intended hardware model assumptions (DAZ/FTZ),
    - the chosen piecewise regions,
    - the acceptance criteria (ULP bounds + known worst-case),
    - and a link to the issue for full history.

- **Bug report markdown living under unit tests**
  - `tests/ttnn/unit_tests/operations/eltwise/GELU_FLOOR_VALUE_BUG_REPORT.md`

  This is likely to be seen as “issue discussion embedded in repo”. It also contains older (now corrected) statistics that can confuse reviewers.

  **Recommendation:**
  - Prefer linking to the GitHub issue from the tests rather than keeping a large report in-tree.
  - If retained, trim it to a short “context + link” note.

---

### Questionable / likely review feedback (fix before public PR)

#### 1) DAZ+FTZ ULP definition is inconsistent and will be challenged

In `tests/ttnn/unit_tests/gtests/test_gelu_ulp_bug.cpp` and the Python mirror, the text claims “denormals map to zero (DAZ)” and “for ULP purposes, all denormals map to the same value as zero,” but then the index/ULP math around zero effectively counts intermediate “phantom steps” that look like denormal slots.

This is a red flag because reviewers will ask:
- Are you measuring ULP over the **set of values the hardware can actually distinguish** (post-DAZ canonicalization), or
- Are you measuring ULP over a **theoretical BF16 ordering that includes denormals**?

**Recommendation: pick one and enforce it consistently:**
- **Preferred for product PRs:** define ULP distance over **post-DAZ canonical BF16** values (normals + zero), with no special “denormal gaps.”
- If an alternative metric is needed, document it explicitly and justify it with hardware semantics and why it’s the right acceptance criterion.

#### 2) “MPFR/mpmath reference tests” are claimed in docs, but tests currently use fp64 `erf()`

Both C++ and Python test files state that standard fp64 `erf()` may saturate for sufficiently negative inputs, and they avoid the extreme negative tail by restricting vectors. Meanwhile, the docs claim MPFR/mpmath-based reference tests.

This mismatch can cause reviewer confusion and skepticism (“what is the reference, actually?”).

**Recommendation:**
- For public CI, prefer **no MPFR/GMP/mpmath** dependencies unless already accepted in the repo.
- Align docs with reality: describe the reference as fp64 `erf()` with safe vectors, and treat deep-tail behavior via FTZ threshold expectations rather than precision references.

#### 3) Python test dependencies / hygiene

`tests/ttnn/unit_tests/operations/eltwise/test_gelu_floor_value_bug.py` imports `numpy` and `loguru`; at minimum, unused imports should be removed, and logging dependencies should match repo conventions.

**Recommendation:**
- Remove unused imports.
- Avoid introducing new runtime deps (like `loguru`) unless they are already guaranteed in the test environment.

#### 4) Diagnostic “mega tests” should not ship as default

The C++ test file includes multiple `DISABLED_...` tests that appear intended for full sweeps / diagnostics. Even if disabled, reviewers may ask to:
- remove them,
- move them to a dedicated “manual/diagnostic” suite,
- or gate them behind a build flag.

**Recommendation:**
- Keep the “small, fast, stable” tests enabled by default.
- Move sweep-style diagnostics elsewhere or behind an explicit opt-in.

---

### Proposed ULP metric spec (make this explicit in the PR)

The PR needs one crisp, internally consistent definition so reviewers can evaluate “Max ULP” claims.

**Recommended product-facing definition (post-DAZ canonical BF16 ULP):**

- **Domain**: BF16 inputs as seen by hardware, i.e. after applying DAZ on input.
  - All BF16 denormal inputs are treated as +0.
  - `-0` is canonicalized to `+0` for comparison purposes.
- **Codomain**: BF16 outputs as produced by hardware, i.e. after applying FTZ on output.
  - Any BF16 denormal output is flushed to +0.
- **ULP distance**: distance between two BF16 values in the **sorted order of distinguishable post-DAZ values**:
  - Allowed distinct values are: all **normal** BF16 values (negative + positive) and **zero**.
  - NaN/Inf handling should be defined (recommended: exclude from the metric; tests should avoid them).
- **Interpretation**:
  - If both expected and actual canonicalize to `0`, ULP distance is `0`.
  - The metric should not count “phantom denormal steps” around zero if the hardware cannot distinguish them.

If the project prefers a different metric (e.g., counting theoretical BF16 denormals), the PR must:
- name it explicitly (do not call it “DAZ model” if denormals are counted as distinct), and
- justify why it is the right acceptance criterion for Tenstorrent hardware behavior.

---

### Suggested PR structure (to reduce review friction)

- **PR 1 (core):** kernel changes + minimal tests + build wiring
  - Keep:
    - `tt_metal/hw/ckernels/*/ckernel_sfpu_gelu.h`
    - `tests/ttnn/unit_tests/gtests/test_gelu_ulp_bug.cpp` (trimmed + consistent ULP model)
    - `tests/ttnn/unit_tests/operations/eltwise/test_gelu_floor_value_bug.py` (trimmed deps)
    - `tests/ttnn/unit_tests/gtests/CMakeLists.txt`
- **PR 2 (optional):** documentation (only if repo policy wants it in-tree)
  - Add a short doc under an approved docs location, and keep the detailed history in the GitHub issue / PR description.

---

### Detailed review by file (keep / change / move decisions)

#### 1) `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h`

**Decision:** keep (core of the fix), but expect reviewers to ask for clarifications.

**Good:**
- Clear top-level regioning:
  - near-zero Taylor approximation,
  - positive saturation shortcut,
  - deep-negative asymptotic handling,
  - higher-precision polynomial segments.
- Same implementation in WH/BH (reduces cross-arch drift).

**Potential review questions / clarifications to add (comments or PR description):**
- **Threshold mismatch risk**: code uses `-13.2f` as the “FTZ boundary”, while research claims a BF16/FTZ-related threshold around `-13.1875`.
  - If `-13.2f` is intentional (float32 exp underflow/FTZ boundary, SFPU exp approximation behavior, safety margin), say so explicitly.
  - Otherwise reviewers will flag it as “magic constant inconsistent with your own research”.
- **Asymptotic formula choice**: deep-negative uses `-exp(-x²/2)/sqrt(2π)` (i.e., `-φ(x)`) and omits the `(1/x - 1/x^3 + ...)` correction terms sometimes used for GELU tails.
  - This is likely fine given BF16+FTZ constraints, but the PR should state why this is sufficient and what bounds it achieves.
- **Testability / determinism**: relies on `_sfpu_exp_21f_<false>`; reviewers may ask how stable this is across chips/firmware revisions. The PR should frame acceptance criteria as “ULP bounded under current SFPU exp implementation” and/or keep tests tolerant.
- **Maintainability**: large inline coefficient tables in headers are acceptable for kernels, but reviewers may ask for provenance (script, generation method, or at least a stable reference).

**Spot-check item reviewers may raise (worth proactively verifying):**
- In `calculate_gelu()` the loop is unrolled and uses `sfpi::dst_reg[0]` then increments `sfpi::dst_reg++` each iteration.
  - This is likely intentional SFPI style, but it is easy to misread; consider a short comment “SFPI dst_reg is an iterator; use [0] then ++”.

#### 2) `tests/ttnn/unit_tests/gtests/test_gelu_ulp_bug.cpp`

**Decision:** keep, but strongly recommend tightening it for product CI.

**Good:**
- Clear grouping: ULP calculator verification + device tests for the 3 bug regions.
- Uses bounded-ULP assertions rather than brittle exact compares.

**Needs changes before public PR:**
- **ULP definition inconsistency**: the DAZ/FTZ narrative and the index math around zero conflict. Pick one metric (see “Proposed ULP metric spec”) and make C++ consistent with it.
- **Remove/relocate heavy diagnostics**: multiple `DISABLED_...` tests are effectively “analysis tools”.
  - Keep them only if repo policy allows disabled diagnostics in-tree; otherwise move to a dedicated diagnostic location or gate behind a compile flag.
- **Reference mismatch**: file comments and some docs mention MPFR-like references; current code uses fp64 `erf()` and explicitly avoids deep negative vectors where it can saturate.
  - That is fine, but be consistent: call it out as “fp64 reference with safe vectors”.

#### 3) `tests/ttnn/unit_tests/operations/eltwise/test_gelu_floor_value_bug.py`

**Decision:** keep, but make it CI-friendly.

**Good:**
- Mirrors the same 3-region intent as the C++ device tests.
- Uses ULP-based comparisons, which is appropriate.

**Needs changes before public PR:**
- **Dependency hygiene**:
  - Remove unused imports (e.g., `numpy` if unused).
  - Avoid `loguru` unless it is already guaranteed in the repo’s test environment.
  - Prefer standard `logging` or plain asserts if possible.
- **ULP definition consistency**: must match the C++ metric and narrative.

#### 4) `tests/ttnn/unit_tests/gtests/CMakeLists.txt`

**Decision:** keep, but ensure no new link dependencies are silently required.

**What to verify:**
- Adding `test_gelu_ulp_bug.cpp` should not require MPFR/GMP to link.
  - If it does, that is a major public-PR concern unless CI already provides those libs and the repo endorses that dependency.

#### 5) `GELU_BF16_Zero_Saturation_Threshold_Research.md`

**Decision:** move out of main tree (recommended) or trim drastically.

**Why:**
- It is detailed investigation material; great for the issue/PR description.
- It includes external tool instructions and long code examples that are not stable product documentation.

**If the repo wants it in-tree:**
- Convert it to a short “final facts” doc under `docs/` / `tech_reports/`, and link out to the issue for the full research narrative.

#### 6) `GELU_ULP_FIX_IMPLEMENTATION.md`

**Decision:** move out of main tree (recommended) or rewrite into a short design note.

**Why:**
- Contains branch history, version history, progress logs, cherry-pick commands, and multiple narrative sections.
- Claims “MPFR/mpmath reference tests”, but the checked-in tests use fp64 `erf()` safe vectors.

**If kept in-tree:**
- Strip it to: “what changed”, “why it’s correct under DAZ+FTZ”, “test coverage”, “known limitations”, “worst-case”.

#### 7) `tests/ttnn/unit_tests/operations/eltwise/GELU_FLOOR_VALUE_BUG_REPORT.md`

**Decision:** remove from tree or reduce to a short pointer.

**Why:**
- Bug reports generally belong in the GitHub issue tracker, not under unit tests.
- Contains older “incorrect model” stats (even if corrected later), which can confuse future readers.

---

### PR narrative guidance (what reviewers will want to see in the PR description)

To make review smooth, the PR description should include:

- **What changed**: “Accurate-mode GELU SFPU implementation updated to piecewise approx (near-zero Taylor, deep-negative asymptotic, piecewise poly segments, positive saturation) for WH + BH.”
- **Why it’s correct under hardware semantics**: explicitly state DAZ (inputs) and FTZ (outputs) assumptions, with a reference to the relevant repo doc.
- **Acceptance criteria**:
  - Define the ULP metric (one definition, not multiple).
  - Report Max ULP / worst-case input(s) under that metric.
  - Call out any known “unsafe reference zones” if using fp64 `erf()` as oracle.
- **Test coverage**:
  - List the enabled tests that cover the 3 historical regions.
  - Mention diagnostics (if any) as opt-in.
- **Compatibility**:
  - “No API changes; functional output improvements; same behavior for x >= 3.0; deep-negative behavior consistent with FTZ.”

---

### Repository-wide MPFR/GMP availability check (implications for MPFR-based reference tests)

This section answers: “Is MPFR/GMP already present in tt-metal, so adding MPFR-based oracle tests should be fine?”

**Findings (repo-wide):**

- **MPFR/GMP are not currently used by compiled tt-metal code/tests**:
  - No existing source code includes `mpfr.h` / `gmp.h`.
  - No existing CMake integration was found (`find_package(MPFR)`, `find_package(GMP)`, `FindMPFR.cmake`, etc.).
  - No existing link flags were found for `-lmpfr` / `-lgmp` in the build system.

- **MPFR/GMP *do* appear in tooling / container environments** (incidental availability):
  - `dockerfile/Dockerfile`: installs **`libmpfr-dev`** in the **`dev`** stage (not clearly in the `ci-build` stage that most CI builds use).
  - `dockerfile/Dockerfile.manylinux`: installs **`gmp-devel`, `mpfr-devel`, `libmpc-devel`** for manylinux wheel builds.
  - `tt_metal/sfpi-info.sh` and `tt_metal/third_party/tt_llk/tests/sfpi-info.sh`: list distro package names including `libgmp-dev`, `libmpfr-dev`, `libmpc-dev` (toolchain context).
  - `tt-train/scripts/install_gdb_14_2.sh`: installs `libgmp-dev libmpfr-dev libmpc-dev` (debugger/toolchain).

**Conclusion / PR implications:**

- MPFR is an excellent reference oracle, but in this repo it currently looks like a **toolchain/dev-image dependency**, not a **guaranteed unit-test dependency**.
- If you add MPFR-based tests, you should treat that as a deliberate build/CI decision:
  - either **make MPFR/GMP an explicit dependency** of the unit-test build (and ensure CI images install the dev packages), or
  - keep MPFR as an **opt-in diagnostic** path (disabled tests / separate target / local-only), while CI uses fp64-oracle “safe vectors.”

---

### Concrete pre-PR checklist (actionable)

- **ULP metric consistency**
  - Choose a single DAZ+FTZ ULP definition and update both C++ and Python accordingly.
  - Ensure the comments and the code match the chosen definition.

- **Test environment safety**
  - Remove or avoid new dependencies (especially MPFR/GMP/mpmath/loguru) unless they’re already standard in CI.
  - Keep enabled tests small/fast; move sweeps to opt-in.

- **Documentation scope**
  - Move research/progress logs out of the main tree or trim drastically.
  - Keep only stable behavioral contracts in-tree; link to the issue for narrative/history.

- **Kernel clarity**
  - Add short comments to justify key constants (e.g., `-13.2f`, `-5.5f`, `0.125f`, `3.0f`) and tie them to intended semantics.
  - If `-13.2f` is chosen as a safety margin vs `-13.1875`, document that explicitly.

---

### Notes

- This review is based on the contents of the changed files listed by `git diff --name-only` against the merge-base with `main`.
- `PR_REVIEW_GELU_ULP_FIX.md` was explicitly excluded from review and not opened/read.

