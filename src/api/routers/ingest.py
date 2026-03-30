from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from src.auth.jwt import get_current_user
from src.backend.audit import log_audit
from src.backend.langgraph_backend import ingest_pdf
from src.backend.rate_limiter import check_rate_limit
from src.database.engine import get_async_session_dep, rls_context
from src.database.table_models import User
from sqlalchemy.ext.asyncio import AsyncSession

ingest_router = APIRouter(prefix="/ingest", tags=["Documents"])
@ingest_router.post("")
async def ingest_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...),
    visibility: str = Form("global"),
    target_department: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    rate_check: User = Depends(check_rate_limit("/ingest")),
    session: AsyncSession = Depends(get_async_session_dep),
):
    # HR-only check
    if current_user.role != "HR":
        raise HTTPException(status_code=403, detail="Only HR can ingest documents")
    
    # Validate visibility
    if visibility not in ["global", "dept", "confidential"]:
        raise HTTPException(status_code=400, detail="Invalid visibility")
    
    async with rls_context(session, current_user):
        contents = await file.read()
        
        summary = await ingest_pdf(
            file_bytes=contents,
            thread_id=thread_id,
            filename=file.filename,
            session=session,
            current_user=current_user,
            visibility=visibility,
            department=target_department
        )
    
        # === AUDIT LOGGING (only this block added) ===
        await log_audit(
            user_id=current_user.id,
            action="ingest_document",
            resource=file.filename or "unknown.pdf",
            details=f"visibility={visibility}, department={target_department or 'ALL'}",
            session=session
        )
        
        return JSONResponse(content={
            "status": "success",
            "summary": summary,
            "thread_id": thread_id,
            "uploaded_by": current_user.email,
            "visibility": visibility
        })
