"""Qdrant-backed MemPalace storage backend (RFC 001).

Provides :class:`QdrantBackend` and :class:`QdrantCollection` — a local-mode
Qdrant implementation of the :mod:`mempalace.backends.base` contract.

Key design decisions
--------------------

* **Local mode only** — uses ``QdrantClient(path=...)``; no server needed.
* **Deterministic ID mapping** — ChromaDB uses arbitrary string IDs; Qdrant
  requires UUID strings or unsigned integers.  We map via UUID-5 and store the
  original ID in the ``_chroma_id`` payload field.
* **Embedding-aware** — Qdrant does not auto-embed like ChromaDB.  When
  embeddings are not provided, a lazily-loaded ``sentence-transformers`` model
  is used (or a caller-supplied ``embedding_fn``).
* **Distance convention** — Qdrant returns cosine similarity (higher = better).
  We convert to ChromaDB-style distance (``1 - score``) so downstream code
  sees the same convention.
"""

import logging
import os
import uuid
from typing import Callable, ClassVar, Optional

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from .base import (
    BaseBackend,
    BaseCollection,
    GetResult,
    HealthStatus,
    PalaceNotFoundError,
    PalaceRef,
    QueryResult,
    _IncludeSpec,
)
from .qdrant_filters import translate_where, translate_where_document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# UUID namespace for deterministic string-id → Qdrant-uuid mapping.
_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# Payload keys reserved for internal use.
_INTERNAL_ID_KEY = "_chroma_id"
_DOCUMENT_KEY = "document"

# Default vector dimensionality (all-MiniLM-L6-v2).
_DEFAULT_VECTOR_SIZE = 384

# Upper bound on points fetched in a single ``scroll`` call.
_MAX_SCROLL_LIMIT = 10000

# Subdirectory created inside the palace path for Qdrant storage.
_QDRANT_DATA_DIR = "qdrant_data"


# ---------------------------------------------------------------------------
# ID mapping
# ---------------------------------------------------------------------------


def _to_point_id(string_id: str) -> str:
    """Map an arbitrary string ID to a deterministic Qdrant-compatible UUID."""
    return str(uuid.uuid5(_ID_NAMESPACE, string_id))


# ---------------------------------------------------------------------------
# Lazy default embedding
# ---------------------------------------------------------------------------

_embed_fn_cache: Optional[Callable[[list[str]], list[list[float]]]] = None


def _get_default_embed_fn() -> Callable[[list[str]], list[list[float]]]:
    """Return (and cache) a default embedding function using *sentence-transformers*.

    Lazily imported so that ``mempalace`` boots without ``sentence-transformers``
    installed — the error is raised only when auto-embedding is actually needed.
    """
    global _embed_fn_cache  # noqa: PLW0603
    if _embed_fn_cache is not None:
        return _embed_fn_cache

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for auto-embedding in the "
            "Qdrant backend.  Install with: pip install sentence-transformers"
        ) from exc

    _model = SentenceTransformer("all-MiniLM-L6-v2")

    def _embed(texts: list[str]) -> list[list[float]]:
        return [list(e) for e in _model.encode(texts)]

    _embed_fn_cache = _embed
    return _embed_fn_cache


# ---------------------------------------------------------------------------
# Vector extraction helper
# ---------------------------------------------------------------------------


def _extract_vector(vec: object) -> list[float]:
    """Extract a flat ``list[float]`` from a Qdrant vector response.

    Qdrant may return vectors as a plain list (single vector) or as a
    ``dict[str, list[float]]`` (named vectors).  This helper normalises
    both forms.
    """
    if isinstance(vec, list):
        return [float(v) for v in vec]
    if isinstance(vec, dict):
        for v in vec.values():
            if isinstance(v, list):
                return [float(x) for x in v]
    return []


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _extract_metadata(payload: Optional[dict]) -> dict:
    """Return payload minus internal keys (``_chroma_id``, ``document``)."""
    if payload is None:
        return {}
    return {k: v for k, v in payload.items() if k not in (_INTERNAL_ID_KEY, _DOCUMENT_KEY)}


def _extract_id(payload: Optional[dict], fallback: object) -> str:
    """Return the original ChromaDB ID from *payload*, falling back to Qdrant point ID."""
    if payload is not None and _INTERNAL_ID_KEY in payload:
        return str(payload[_INTERNAL_ID_KEY])
    return str(fallback)


# ---------------------------------------------------------------------------
# Collection adapter
# ---------------------------------------------------------------------------


class QdrantCollection(BaseCollection):
    """Thin adapter translating the :class:`BaseCollection` contract to Qdrant calls."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._embedding_fn = embedding_fn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* using the configured or default embedding function."""
        if self._embedding_fn is not None:
            return self._embedding_fn(texts)
        return _get_default_embed_fn()(texts)

    def _build_points(
        self,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]],
        embeddings: list[list[float]],
    ) -> list[qmodels.PointStruct]:
        """Build a list of :class:`qmodels.PointStruct` for upsert."""
        points: list[qmodels.PointStruct] = []
        for i, doc in enumerate(documents):
            payload: dict = {
                _INTERNAL_ID_KEY: ids[i],
                _DOCUMENT_KEY: doc,
            }
            if metadatas is not None and i < len(metadatas) and metadatas[i] is not None:
                payload.update(metadatas[i])
            points.append(
                qmodels.PointStruct(
                    id=_to_point_id(ids[i]),
                    vector=embeddings[i],
                    payload=payload,
                )
            )
        return points

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def add(
        self,
        *,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        if embeddings is None:
            embeddings = self._embed(documents)
        points = self._build_points(documents, ids, metadatas, embeddings)
        self._client.upsert(collection_name=self._collection_name, points=points)

    def upsert(
        self,
        *,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        if embeddings is None:
            embeddings = self._embed(documents)
        points = self._build_points(documents, ids, metadatas, embeddings)
        self._client.upsert(collection_name=self._collection_name, points=points)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def query(
        self,
        *,
        query_texts: Optional[list[str]] = None,
        query_embeddings: Optional[list[list[float]]] = None,
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        include: Optional[list[str]] = None,
    ) -> QueryResult:
        # Validate filter clauses (raises UnsupportedFilterError if unsupported).
        q_filter = translate_where(where)
        translate_where_document(where_document)

        if (query_texts is None) == (query_embeddings is None):
            raise ValueError("query requires exactly one of query_texts or query_embeddings")

        spec = _IncludeSpec.resolve(include, default_distances=True)
        need_payload = spec.documents or spec.metadatas

        if query_texts is not None:
            if not query_texts:
                raise ValueError("query_texts must be a non-empty list")
            query_embeddings = self._embed(query_texts)

        # Guard: after potential embedding, query_embeddings must be set.
        if query_embeddings is None:
            raise ValueError("no query vectors available")

        num_queries = len(query_embeddings)

        all_ids: list[list[str]] = []
        all_docs: list[list[str]] = []
        all_metas: list[list[dict]] = []
        all_dists: list[list[float]] = []
        all_embeds: Optional[list[list[list[float]]]] = [] if spec.embeddings else None

        for qvec in query_embeddings:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=qvec,
                query_filter=q_filter,
                limit=n_results,
                with_payload=need_payload,
                with_vectors=spec.embeddings,
            )
            points = response.points

            ids_list: list[str] = []
            docs_list: list[str] = []
            metas_list: list[dict] = []
            dists_list: list[float] = []
            embeds_list: list[list[float]] = []

            for pt in points:
                ids_list.append(_extract_id(pt.payload, pt.id))
                if spec.documents:
                    docs_list.append(
                        str(pt.payload.get(_DOCUMENT_KEY, "")) if pt.payload else ""
                    )
                if spec.metadatas:
                    metas_list.append(_extract_metadata(pt.payload))
                # Convert cosine similarity → ChromaDB-style distance.
                dists_list.append(1.0 - pt.score)
                if spec.embeddings and all_embeds is not None and pt.vector is not None:
                    embeds_list.append(_extract_vector(pt.vector))

            all_ids.append(ids_list)
            all_docs.append(docs_list)
            all_metas.append(metas_list)
            all_dists.append(dists_list)
            if all_embeds is not None:
                all_embeds.append(embeds_list)

        return QueryResult(
            ids=all_ids,
            documents=all_docs,
            metadatas=all_metas,
            distances=all_dists,
            embeddings=all_embeds,
        )

    def get(
        self,
        *,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[list[str]] = None,
    ) -> GetResult:
        q_filter = translate_where(where)
        translate_where_document(where_document)

        spec = _IncludeSpec.resolve(include, default_distances=False)
        need_payload = spec.documents or spec.metadatas

        if ids is not None:
            # Retrieve by explicit IDs — ignore filter / limit / offset.
            records = self._client.retrieve(
                collection_name=self._collection_name,
                ids=[_to_point_id(i) for i in ids],
                with_payload=need_payload,
                with_vectors=spec.embeddings,
            )
        else:
            # Scroll with optional filter.
            if limit is not None and limit <= 0:
                return GetResult.empty()

            effective_limit = (limit if limit is not None else _MAX_SCROLL_LIMIT) + (
                offset or 0
            )
            records, _next = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=q_filter,
                limit=effective_limit,
                with_payload=need_payload,
                with_vectors=spec.embeddings,
            )
            start = offset or 0
            if limit is not None:
                records = records[start : start + limit]
            else:
                records = records[start:]

        if not records:
            return GetResult.empty()

        out_ids: list[str] = []
        out_docs: list[str] = []
        out_metas: list[dict] = []
        out_embeds: Optional[list[list[float]]] = [] if spec.embeddings else None

        for rec in records:
            out_ids.append(_extract_id(rec.payload, rec.id))
            if spec.documents:
                out_docs.append(
                    str(rec.payload.get(_DOCUMENT_KEY, "")) if rec.payload else ""
                )
            if spec.metadatas:
                out_metas.append(_extract_metadata(rec.payload))
            if spec.embeddings and out_embeds is not None and rec.vector is not None:
                out_embeds.append(_extract_vector(rec.vector))

        # Pad lists to match ids length for downstream zipping safety.
        if spec.documents and len(out_docs) < len(out_ids):
            out_docs.extend([""] * (len(out_ids) - len(out_docs)))
        if spec.metadatas and len(out_metas) < len(out_ids):
            out_metas.extend([{}] * (len(out_ids) - len(out_metas)))

        return GetResult(
            ids=out_ids,
            documents=out_docs,
            metadatas=out_metas,
            embeddings=out_embeds,
        )

    def delete(
        self,
        *,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
    ) -> None:
        if ids is not None:
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=[_to_point_id(i) for i in ids],
            )
        elif where is not None:
            q_filter = translate_where(where)
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=qmodels.FilterSelector(filter=q_filter),
            )
        else:
            raise ValueError("delete requires at least one of ids or where")

    def count(self) -> int:
        result = self._client.count(collection_name=self._collection_name)
        return result.count


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class QdrantBackend(BaseBackend):
    """MemPalace storage backend using local-mode Qdrant.

    Maintains a cache of ``QdrantClient`` instances keyed by palace path.
    All I/O is deferred to first use — construction is lightweight.
    """

    name: ClassVar[str] = "qdrant"
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {
            "supports_embeddings_in",
            "supports_embeddings_passthrough",
            "supports_embeddings_out",
            "supports_metadata_filters",
            "local_mode",
        }
    )

    def __init__(
        self,
        embedding_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
    ) -> None:
        self._clients: dict[str, QdrantClient] = {}
        self._embedding_fn = embedding_fn
        self._closed = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _client_for(self, palace_path: str) -> QdrantClient:
        """Return a cached ``QdrantClient`` for *palace_path*, creating one if needed."""
        if self._closed:
            from .base import BackendClosedError  # late import avoids cycles

            raise BackendClosedError("QdrantBackend has been closed")

        cached = self._clients.get(palace_path)
        if cached is not None:
            return cached

        qdrant_path = os.path.join(palace_path, _QDRANT_DATA_DIR)
        os.makedirs(qdrant_path, exist_ok=True)
        client = QdrantClient(path=qdrant_path)
        self._clients[palace_path] = client
        return client

    # ------------------------------------------------------------------
    # BaseBackend surface
    # ------------------------------------------------------------------

    def get_collection(
        self,
        *,
        palace: PalaceRef,
        collection_name: str,
        create: bool = False,
        options: Optional[dict] = None,
    ) -> QdrantCollection:
        if self._closed:
            from .base import BackendClosedError

            raise BackendClosedError("QdrantBackend has been closed")

        palace_path = palace.local_path
        if palace_path is None:
            raise PalaceNotFoundError("QdrantBackend requires PalaceRef.local_path")

        if not create and not os.path.isdir(palace_path):
            raise PalaceNotFoundError(palace_path)

        if create:
            os.makedirs(palace_path, exist_ok=True)

        client = self._client_for(palace_path)
        qdrant_name = f"{palace.id}__{collection_name}"
        vector_size = int((options or {}).get("vector_size", _DEFAULT_VECTOR_SIZE))

        if create:
            # get_or_create: attempt to fetch first, create on failure.
            try:
                client.get_collection(qdrant_name)
            except Exception:
                client.create_collection(
                    collection_name=qdrant_name,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
        else:
            # Verify the collection exists (will raise if not found).
            client.get_collection(qdrant_name)

        return QdrantCollection(client, qdrant_name, self._embedding_fn)

    def close_palace(self, palace: PalaceRef) -> None:
        """Evict cached client handles for *palace*."""
        path = palace.local_path
        if path is None:
            return
        client = self._clients.pop(path, None)
        if client is not None:
            client.close()

    def close(self) -> None:
        """Shut down all cached Qdrant clients."""
        for client in self._clients.values():
            client.close()
        self._clients.clear()
        self._closed = True

    def health(self, palace: Optional[PalaceRef] = None) -> HealthStatus:
        if self._closed:
            return HealthStatus.unhealthy("backend closed")
        return HealthStatus.healthy()

    @classmethod
    def detect(cls, path: str) -> bool:
        """Return ``True`` if *path* contains a ``qdrant_data/`` directory."""
        return os.path.isdir(os.path.join(path, _QDRANT_DATA_DIR))
