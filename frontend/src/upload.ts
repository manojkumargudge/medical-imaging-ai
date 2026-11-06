// Client-side checks before an upload. The API checks everything again (src/api/uploads.py);
// these only save a round trip and say early what is wrong.

export const MAX_BYTES = 20 * 1024 * 1024;
export const SCOPE_WARNING =
  "Trained on NIH ChestX-ray14 frontal chest X-rays. Scores for other images have no meaning.";

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const DICM = [0x44, 0x49, 0x43, 0x4d];

export type FileKind = "png" | "dicom";
export type Check = { ok: true; kind: FileKind } | { ok: false; message: string };

/** The kind from the first bytes (content, not the name): PNG signature or DICOM preamble. */
export function kindFromBytes(head: Uint8Array): FileKind | null {
  if (PNG_SIGNATURE.every((byte, i) => head[i] === byte)) return "png";
  if (head.length >= 132 && DICM.every((byte, i) => head[128 + i] === byte)) return "dicom";
  return null;
}

async function readHead(file: Blob): Promise<Uint8Array> {
  const slice = file.slice(0, 132);
  const buffer =
    typeof slice.arrayBuffer === "function"
      ? await slice.arrayBuffer()
      : await new Promise<ArrayBuffer>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as ArrayBuffer);
          reader.onerror = () => reject(reader.error);
          reader.readAsArrayBuffer(slice);
        });
  return new Uint8Array(buffer);
}

export async function checkFile(file: Blob): Promise<Check> {
  if (file.size === 0) return { ok: false, message: "The file is empty." };
  if (file.size > MAX_BYTES) return { ok: false, message: "The file is larger than 20 MB." };
  const kind = kindFromBytes(await readHead(file));
  if (!kind) {
    return { ok: false, message: "Only PNG or DICOM files (checked by content, not by name)." };
  }
  return { ok: true, kind };
}
