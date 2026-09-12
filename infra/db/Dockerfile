ARG PG_SEARCH_VERSION=0.25.9

FROM curlimages/curl:8.17.0 AS pg-search-downloader

ARG PG_SEARCH_VERSION
ARG TARGETARCH

RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) checksum="4f1399158243dc3339563d75f5617ebc4b74bcb4ca2722335f767bc5edfb1544" ;; \
        arm64) checksum="83022d643ecf04775ffd0f96539ce1ae624efb4123b244d2b44914af7b9f384d" ;; \
        *) echo "Unsupported architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    package="postgresql-17-pg-search_${PG_SEARCH_VERSION}-1PARADEDB-bookworm_${TARGETARCH}.deb"; \
    curl -fL \
        "https://github.com/paradedb/paradedb/releases/download/v${PG_SEARCH_VERSION}/${package}" \
        -o /tmp/pg_search.deb; \
    printf '%s  %s\n' "${checksum}" /tmp/pg_search.deb > /tmp/pg_search.sha256; \
    sha256sum -c /tmp/pg_search.sha256

FROM pgvector/pgvector:0.8.6-pg17

COPY --from=pg-search-downloader /tmp/pg_search.deb /tmp/pg_search.deb

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends /tmp/pg_search.deb; \
    rm -f /tmp/pg_search.deb; \
    rm -rf /var/lib/apt/lists/*

CMD ["postgres", "-c", "shared_preload_libraries=pg_search"]
