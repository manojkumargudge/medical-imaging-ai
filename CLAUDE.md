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
