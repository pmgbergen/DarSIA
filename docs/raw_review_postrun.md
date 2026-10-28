# Review a completed RAW run

Register the original acquisition preparation for each run in `series.toml`:

```toml
[raw_review.preparations.run_01]
path = "data/preparations/run_01/raw_acquisitions.json"
sha256 = "<exact SHA-256 of that file>"
size_bytes = 12345
```

Paths may be absolute or relative to the series folder. A preparation must
match the completed run's original native and phase configurations, baseline,
RAW identities and geometry. The application does not search neighbouring
folders or substitute another preparation.

In Jobs, select the completed job and choose **Files → Prepare RAW review…**.
The saved preparations are used directly. Missing declarations or a different
series are explained beside the action. While Jobs is open, an observed
transition to verified completion also starts this step once.

The output lives separately under the series' annotations folder,
`raw_reviews/<job SHA>`. It includes photographic assets and Change when the
baseline retains its own rig geometry and all frames retain their classes.
Production products and original RAW files remain unchanged. Existing complete
reviews are verified and reused; partial attempts are retained.

With the GUI closed, use the same configured inputs explicitly:

```text
python -B -m darsia.fluidflower.raw_review_postrun --job-root <job> --stores <store-map> --series-root <series>
```

An attached immutable request can also be verified and run with `--request
<request.json> --request-sha256 <SHA>`. No scheduler is installed.

Photographs alone do not establish model identity or training eligibility.
Reviewing predictions requires their saved model/collection binding; training
also requires a compatible source-bound context. Missing physical fields and
QA are reported instead of being inferred from the rendered photographs.
