import { useEffect, useState } from "react";
import { getJson, type StudyDetail, type Thresholds } from "./api";
import { AuditPage } from "./components/AuditPage";
import { CtView } from "./components/CtView";
import { DxView } from "./components/DxView";
import { ErrorBox, Layout } from "./components/Layout";
import { Worklist } from "./components/Worklist";
import { viewerKind } from "./status";

export interface Route {
  page: "worklist" | "study" | "audit";
  id?: string;
  label?: string;
}

/** "#/" -> worklist; "#/audit" -> audit log; "#/studies/<id>?label=<label>" -> one study. */
export function parseRoute(hash: string): Route {
  const [path = "", query = ""] = hash.replace(/^#/, "").split("?");
  if (path === "/audit") return { page: "audit" };
  const match = /^\/studies\/([^/]+)$/.exec(path);
  if (!match?.[1]) return { page: "worklist" };
  const label = new URLSearchParams(query).get("label") ?? undefined;
  return { page: "study", id: decodeURIComponent(match[1]), label };
}

function StudyPage({ id, label }: { id: string; label?: string }) {
  const [data, setData] = useState<{ detail: StudyDetail; thresholds: Thresholds } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setData(null);
    setError(null);
    Promise.all([
      getJson<StudyDetail>(`/api/studies/${encodeURIComponent(id)}`),
      getJson<Thresholds>("/api/config/thresholds"),
    ])
      .then(([detail, thresholds]) => setData({ detail, thresholds }))
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, [id]);

  if (error) return <ErrorBox title="Could not load this study" lines={[error]} />;
  if (!data) return <p>Loading…</p>;
  const kind = viewerKind(data.detail.study.modality);
  if (kind === "dx")
    return <DxView detail={data.detail} thresholds={data.thresholds} initialLabel={label} />;
  if (kind === "ct") return <CtView detail={data.detail} />;
  return <p>No viewer for modality {data.detail.study.modality || "unknown"}.</p>;
}

export function App() {
  const [route, setRoute] = useState(() => parseRoute(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return (
    <Layout>
      {route.page === "audit" ? (
        <AuditPage />
      ) : route.page === "study" && route.id ? (
        <>
          <p>
            <a href="#/">← Worklist</a>
          </p>
          <StudyPage id={route.id} label={route.label} />
        </>
      ) : (
        <Worklist />
      )}
    </Layout>
  );
}
