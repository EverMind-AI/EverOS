# Frozen compatibility baseline

These JSON files were hand-authored on 2026-10-27 against EverOS commit
`0d3b50276efc971d8030ef7836eb32e3acd7cf1f` and installed `everalgo-core==0.6.0`.
They are synthetic fixtures, not exported production records or payloads
captured from a previous release. No current model serializer generated them.

- `memcell.json` freezes ordered text/tool-result union members and exact
  millisecond timestamps. The SQLite test inserts these bytes directly into
  `memcell.payload_json`; it does not call the current store writer.

Keep existing fixtures unchanged when upgrading dependencies. Add a separately
attributed versioned fixture for new formats; repair readers or declare and test
an explicit migration when a frozen fixture stops loading. These cases do not
establish universal historical compatibility, downgrade compatibility, or
compatibility with untested tool/message variants.
