# v0.6.0 reference result, v0.6.2 evidence repair

`reference_summary.json` is a compact transcription of the completed frozen
run. `protocol.json` is the reconstructed frozen protocol artifact; its
canonical SHA-256 matches the displayed v0.6.0 protocol hash.

`report.json.gz` is the full compressed Monte Carlo report regenerated with the
frozen v0.6.0 protocol and pinned v0.6.2 dependencies. It includes raw per-case
failure counts, Wilson upper bounds, and crossover decisions.

`wilson_distance_table.csv` is derived from `report.json.gz` for plotting and
independent inspection of the Wilson crossover decision.

`claim_certificate.json` records the v0.6.2 claim boundary and points to the
archived full report.

To regenerate those complete artifacts:

```bash
python src/ft_unit_change_time_rotated_surface_code_v0_6_0.py
```

That command writes `protocol.json` and `report.json` under
`ft_unit_change_time_v0_6_0_results/`. Compress the report with `gzip -n` when
recreating `results/v0.6.0/report.json.gz`.

Expected identifying hashes:

```text
protocol_sha256 = fec91e30001712f3d9ac84c0e45a6b70f2d5ae7189d3c9ac6d1096d47505cbf6
certificate_sha256_before_self_field = 2db9620419ac5a7ff64510c65e0d391c4603b6c361fdd8aadd2d9f96165cbc79
report_json_gz_sha256 = f9edf8692aaa0f116cc6584507e7f326d184831f251669aa6e2dd2dd143bb95a
```

Because Monte Carlo samples are deterministic for the pinned software path used
here but can still vary across runtime implementations, compare regenerated
protocols, gates, and reported decision fields before treating a rerun as
byte-identical to this archived artifact.
