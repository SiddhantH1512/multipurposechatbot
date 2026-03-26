from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from src.auth.jwt import get_current_user
from src.database.engine import get_async_session_dep, sync_engine
from src.database.table_models import User

documents_router = APIRouter(prefix="/documents", tags=["documents"], redirect_slashes=False)

PRIVILEGED_ROLES = {"HR", "EXECUTIVE"}


def _user_role(user: User) -> str:
    return user.role.value if hasattr(user.role, "value") else str(user.role)


# ─────────────────────────────────────────────────────────────────────
# GET /documents  — list all documents the current user can see
# ─────────────────────────────────────────────────────────────────────
@documents_router.get("")
async def list_documents(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session_dep),
):
    """
    Returns one row per distinct document_id (or filename fallback).
    HR / EXECUTIVE see everything.
    Others see global docs + their own department's docs.
    """
    role = _user_role(current_user)
    is_privileged = role in PRIVILEGED_ROLES

    # langchain_pg_embedding stores metadata in the cmetadata JSONB column.
    # We aggregate per document_id to get one row per document.
    if is_privileged:
        filter_clause = ""
        params: dict = {}
    else:
        filter_clause = """
            AND (
                cmetadata->>'visibility' = 'global'
                OR (
                    cmetadata->>'visibility' = 'dept'
                    AND cmetadata->>'department' = :dept
                )
                OR cmetadata->>'visibility' = 'confidential'
                    AND :role IN ('HR', 'EXECUTIVE')
            )
        """
        params = {"dept": current_user.department, "role": role}

    # Sync query — langchain_pg_embedding is only accessible via sync_engine
    with sync_engine.connect() as conn:
        rows = conn.execute(
            text(f"""
                SELECT
                    COALESCE(cmetadata->>'document_id', cmetadata->>'filename') AS document_id,
                    cmetadata->>'filename'           AS filename,
                    cmetadata->>'visibility'         AS visibility,
                    cmetadata->>'department'         AS department,
                    cmetadata->>'uploaded_by_email'  AS uploaded_by,
                    cmetadata->>'uploaded_at'        AS uploaded_at,
                    COUNT(*)::int                    AS chunk_count
                FROM langchain_pg_embedding
                WHERE 1=1 {filter_clause}
                GROUP BY
                    COALESCE(cmetadata->>'document_id', cmetadata->>'filename'),
                    cmetadata->>'filename',
                    cmetadata->>'visibility',
                    cmetadata->>'department',
                    cmetadata->>'uploaded_by_email',
                    cmetadata->>'uploaded_at'
                ORDER BY MAX(cmetadata->>'uploaded_at') DESC NULLS LAST
            """),
            params,
        ).fetchall()

    return {
        "documents": [
            {
                "document_id": r[0],
                "filename":    r[1],
                "visibility":  r[2],
                "department":  r[3],
                "uploaded_by": r[4],
                "uploaded_at": r[5],
                "chunk_count": r[6],
            }
            for r in rows
        ]
    }


# ─────────────────────────────────────────────────────────────────────
# Request body for PATCH
# ─────────────────────────────────────────────────────────────────────
class VisibilityUpdate(BaseModel):
    visibility: str                    # "global" | "dept" | "confidential"
    department: Optional[str] = None   # required when visibility == "dept"


# ─────────────────────────────────────────────────────────────────────
# PATCH /documents/{document_id}/visibility  — HR only
# ─────────────────────────────────────────────────────────────────────
@documents_router.patch("/{document_id}/visibility")
async def update_visibility(
    document_id: str,
    body: VisibilityUpdate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session_dep),
):
    """
    Update the visibility (and optionally department) of every chunk
    belonging to the given document_id, then sync thread_metadata.

    Rules:
      - Only HR can call this endpoint.
      - visibility must be one of: global, dept, confidential.
      - When visibility == "dept", department is required.
    """
    role = _user_role(current_user)
    if role != "HR":
        raise HTTPException(status_code=403, detail="Only HR can change document visibility")

    if body.visibility not in ("global", "dept", "confidential"):
        raise HTTPException(status_code=400, detail="visibility must be global, dept, or confidential")

    if body.visibility == "dept" and not body.department:
        raise HTTPException(status_code=400, detail="department is required when visibility is 'dept'")

    # Resolve new department value
    if body.visibility == "global":
        new_department = "General"
    elif body.visibility == "confidential":
        new_department = "HR"
    else:
        new_department = body.department

    # ── 1. Check document exists ────────────────────────────────────
    with sync_engine.connect() as conn:
        count = conn.execute(
            text("""
                SELECT COUNT(*) FROM langchain_pg_embedding
                WHERE COALESCE(cmetadata->>'document_id', cmetadata->>'filename') = :doc_id
            """),
            {"doc_id": document_id},
        ).scalar()

    if not count:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

    # ── 2. Update all chunks in langchain_pg_embedding ─────────────
    # jsonb_set patches individual keys without touching the rest of cmetadata
    with sync_engine.begin() as conn:
        updated = conn.execute(
            text("""
                UPDATE langchain_pg_embedding
                SET cmetadata = jsonb_set(
                    jsonb_set(cmetadata, '{visibility}',  to_jsonb(cast(:visibility as text))),
                    '{department}', to_jsonb(cast(:department as text))
                )
                WHERE COALESCE(cmetadata->>'document_id', cmetadata->>'filename') = :doc_id
            """),
            {
                "visibility":  body.visibility,
                "department":  new_department,
                "doc_id":      document_id,
            },
        )
        chunks_updated = updated.rowcount

    # ── 3. Sync thread_metadata ─────────────────────────────────────
    await session.execute(
        text("""
            UPDATE thread_metadata
            SET
                is_global  = :is_global,
                department = :department
            WHERE document_id = :doc_id
        """),
        {
            "is_global":  body.visibility == "global",
            "department": new_department,
            "doc_id":     document_id,
        },
    )
    await session.commit()

    return {
        "document_id":    document_id,
        "visibility":     body.visibility,
        "department":     new_department,
        "chunks_updated": chunks_updated,
        "message":        f"Visibility updated to '{body.visibility}' for {chunks_updated} chunks",
    }