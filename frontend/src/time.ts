/** A UTC timestamp from the API in the browser's time zone, with the zone's name:
 * "06 Nov 2026, 10:40 IST". The zone name comes from the browser's locale (en-IN gives "IST",
 * en-US gives "GMT+5:30" for the same zone). timeZone and locale are for tests. */
export function formatLocal(iso: string | null, timeZone?: string, locale?: string): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const parts = new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
    timeZone,
  }).formatToParts(date);
  const part = (type: Intl.DateTimeFormatPartTypes) => parts.find((p) => p.type === type)?.value;
  const zone = new Intl.DateTimeFormat(locale ?? navigator.language, {
    timeZoneName: "short",
    timeZone,
  })
    .formatToParts(date)
    .find((p) => p.type === "timeZoneName")?.value;
  const text = `${part("day")} ${part("month")} ${part("year")}, ${part("hour")}:${part("minute")}`;
  return zone ? `${text} ${zone}` : text;
}
