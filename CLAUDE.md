# Project: Medical Imaging AI (AIML-02)

For education only, not for clinical use.

## Rules
- Ask before destructive operations or changes that may discard data.
- Never drop, delete or overwrite a database, file, folder or registry entry without asking
  first, even copies or test data.
- Every comparison check (backup vs live, before vs after, A vs B) must assert both sides are
  non-empty and print row counts.
- Patient-level splits only: all images of one patient stay in one split, and a check must
  confirm that no patient ID appears in two splits.
- Never claim clinical accuracy. Report metrics only as measured on a named dataset and split.
- Never create containers, Compose projects, volumes, folders or databases outside the agreed
  plan without asking first. This includes probes and "dry runs": `docker compose create`
  creates containers and volumes, and Docker Desktop creates a host folder for a missing bind
  source.
- If a stray action happens anyway, stop and report it: what was created or changed, and where.
  Leave it in place until the user decides; do not clean it up on your own.
