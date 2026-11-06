# The viewer: the React build served by nginx (unprivileged, port 8080), which proxies /api to
# the API service. The browser reaches only this origin; it never calls Orthanc, and no
# credentials are in this image. For education only, not for clinical use.
FROM node:22.22.2-bookworm-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.28.0-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=5s --retries=6 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/ || exit 1
