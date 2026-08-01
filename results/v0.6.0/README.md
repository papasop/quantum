# v0.6.0 reference result

`reference_summary.json` is a compact transcription of the completed frozen
run. It is not a substitute for the full generated `protocol.json` and
`report.json`.

To regenerate those complete artifacts:

```bash
python src/ft_unit_change_time_rotated_surface_code_v0_6_0.py
```

Expected identifying hashes:

```text
protocol_sha256 = fec91e30001712f3d9ac84c0e45a6b70f2d5ae7189d3c9ac6d1096d47505cbf6
certificate_sha256_before_self_field = e4658480b6a4cb71caadfb6532453c4a31ba11f732dfaf36ffd66cc10921138c
```

Because Monte Carlo samples are generated during a rerun, reproduce the
protocol and gates rather than assuming byte-identical result artifacts across
all software platforms.
