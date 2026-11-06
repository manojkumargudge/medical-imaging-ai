**For education only. Not for clinical use.**

# Viewer API

The viewer's own API (`src/api/`, FastAPI). The browser reaches it through nginx at
`http://127.0.0.1:8080/api/...`; it also listens on `127.0.0.1:8000` directly. It is the only
part of the stack that holds the Orthanc credentials. The machine-readable description is
[openapi.json](openapi.json) (exported from the code with `python -m scripts.export_openapi`; a
test checks it is up to date), and the running API serves it at `/docs`.

## Errors

Every error body is JSON with a `detail` sentence (422 also lists the fields in `errors`).

| code | when |
|---|---|
| 404 | Unknown study ID (not listed by Orthanc), unknown label (not one of the 14), CT slice out of range, no mask, or a stored file path that does not resolve inside the results folder. Path-traversal attempts get 404. |
| 409 | A review whose submit token was saved before ("already saved"), or whose result is not the study's newest of that model ("a newer result exists, reload"). |
| 413 | An upload larger than 20 MB. |
| 422 | Missing or invalid input: a review field, an upload that is not an accepted PNG or DICOM, a page number out of range, an image that cannot be read. |
| 503 | Orthanc or the results database is down, or Orthanc refused the API's credentials. The message names the service and the reason. |

## Endpoints

IDs: `{study_id}` is Orthanc's study ID (as the worklist returns it); `{k}` is a 0-based CT slice
index in patient order.

### `GET /api/health`
Database and Orthanc reachable, Orthanc's name and stack (`demo` or `test`), and the notice.
Errors: 503 (with a body saying which service is down).

### `GET /api/studies`
The worklist, newest first. Per study: `id`, `study_uid`, `patient_id`, `description` (DICOM
StudyDescription, else SeriesDescription), `modality`, `status` (`receiving`, `processing`,
`done`, `failed`, `no_model`), `review_status` (`none`, `approved`, `rejected`; the newest
review), `source` (`demo`, `upload`, `unknown`), `summary` (DX: "N of 13 above threshold"; CT:
"Spleen X mL"), `updated_at` (UTC). Errors: 503.

### `GET /api/studies/{study_id}`
The study (as in the worklist), its series, the newest result per model with its results JSON,
the latest review, and the note that scores are not calibrated probabilities. Errors: 404, 503.

### `GET /api/studies/{study_id}/image`
The first DX/CR image as an 8-bit PNG, from the DICOM in Orthanc. Errors: 404 (no DX/CR image),
422 (pixels cannot be shown), 503.

### `GET /api/studies/{study_id}/cam/{label}`
The raw Grad-CAM map (8-bit grey, 224 x 224 PNG) of the newest classifier result for one of the
14 labels; the browser colours it. Errors: 404 (unknown label, no result, no raw map), 503.

### `GET /api/config/thresholds`
The 14 labels from `configs/viewer_thresholds.yaml` (threshold, set, low-reliability flag) and
the note that scores are not calibrated probabilities.

### `GET /api/studies/{study_id}/ct/meta`
`slices`, `columns`, `rows`, `window` (40/400), `mask` (true or false), `first_mask_slice`,
`last_mask_slice`, `spleen_ml` (stored, from the 1.5 mm model grid), `spleen_ml_dicom_grid`,
`orientation` (edge letters: top A, bottom P, left R, right L after the display flip),
`display_flips`, `result_id`. Errors: 404 (no CT series), 503.

### `GET /api/studies/{study_id}/ct/slice/{k}`
Slice k as an 8-bit PNG, window 40/400, shown anterior up with the patient's right on the left.
Errors: 404 (k out of range), 422 (k not an integer), 503.

### `GET /api/studies/{study_id}/ct/mask/{k}`
The spleen mask on slice k, on the DICOM grid, as a transparent RGBA PNG with the same display
flip as the slice. Errors: 404 (k out of range, no mask), 422, 503.

### `GET /api/studies/{study_id}/reviews`
The study's reviews, newest first. Errors: 404, 503.

### `POST /api/studies/{study_id}/reviews`
JSON body:

| field | type | rule |
|---|---|---|
| `decision` | `approve` or `reject` | required |
| `reviewer` | string | required, 1 to 100 characters after trimming; not verified (no login) |
| `comment` | string or null | required (non-blank) to reject; at most 2000 characters |
| `submit_token` | UUID | required; a new one each time the review panel opens |
| `inference_result_id` | integer | required; the result the reviewer saw |

201 with the saved review; the review and its `review` audit row are written in one
transaction. Errors: 404, 409, 422, 503.

### `POST /api/uploads`
Multipart form: `file` (one file) and `uploader` (1 to 100 characters, not verified).

- PNG (mode L, RGB or RGBA) or DICOM DX/CR with `PatientIdentityRemoved = YES`, checked by
  content; a CT answers 422 "CT needs a full series; not supported for upload".
- Each side 256 to 4096 pixels.
- 201 with `upload_id` and `study_id`. The same file again (its study still in Orthanc): 200
  with `duplicate: true` and the earlier study; nothing is added.
- The file name is never written into the DICOM or returned.

Errors: 413, 422, 503 (no rows are written).

### `GET /api/audit?page=1&page_size=50`
The audit log, newest first: `entries` (id, created_at, actor, event, study_uid, details),
`page`, `page_size` (at most 200), `total`. Errors: 422 (page < 1, page_size > 200), 503.
