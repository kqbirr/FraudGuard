# syntax=docker/dockerfile:1
#
# FraudGuard dashboard image: build the Vite bundle, serve it with Nginx.
#
# Build from frontend/ :
#
#   docker build -t fraudguard-frontend \
#     --build-arg VITE_API_BASE_URL=http://127.0.0.1:8000 .
#
# VITE_API_BASE_URL is public configuration that Vite inlines into the bundle at
# build time, so it must be a URL the *browser* can reach - never an internal
# Compose service name such as http://backend:8000. No secret belongs in it: the
# bundle is downloaded by every visitor.

FROM node:24-alpine AS build

WORKDIR /app

# Default matches the backend port published by docker-compose.yml.
ARG VITE_API_BASE_URL=http://127.0.0.1:8000
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

# Lockfile-only install: reproducible and uses the committed dependency graph.
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

# The rest of the app (node_modules, dist and .env* are excluded by
# frontend/.dockerignore, so this does not overwrite the installed modules).
COPY . .

# `npm run build` is `tsc --noEmit && vite build`: the image cannot be produced
# from sources that fail the type check.
RUN npm run build

FROM nginx:1.29-alpine AS runtime

# SPA routing, static asset caching and a small set of security headers.
COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["wget", "-q", "--spider", "http://127.0.0.1/"]

# Nginx runs as PID 1 in the foreground; the master process drops to the nginx
# user for the workers after binding port 80.
CMD ["nginx", "-g", "daemon off;"]
