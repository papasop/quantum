# v0.6.0 reference result, v0.6.1 evidence repair

`reference_summary.json` is a compact transcription of the completed frozen
run. `protocol.json` is the reconstructed frozen protocol artifact; its
canonical SHA-256 matches the displayed v0.6.0 protocol hash.

`claim_certificate.json` records the v0.6.1 claim boundary and makes the
evidence gap explicit. The v0.6.0 GitHub zip did not include the full
Monte Carlo `report.json`, so raw per-case failure counts and per-distance
Wilson tables must be regenerated from the frozen script before claiming a
fully independently recomputable report certificate.

To regenerate those complete artifacts:

```bash
python src/ft_unit_change_time_rotated_surface_code_v0_6_0.py
```

That command writes `protocol.json` and `report.json` under
`ft_unit_change_time_v0_6_0_results/`. Compress the report as
`results/v0.6.0/report.json.gz` when preserving a full release artifact.

Expected identifying hashes:

```text
protocol_sha256 = fec91e30001712f3d9ac84c0e45a6b70f2d5ae7189d3c9ac6d1096d47505cbf6
certificate_sha256_before_self_field = e4658480b6a4cb71caadfb6532453c4a31ba11f732dfaf36ffd66cc10921138c
```

Because Monte Carlo samples are generated during a rerun, reproduce the
protocol and gates rather than assuming byte-identical result artifacts across
all software platforms.
