from fastapi import HTTPException, Depends, APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from api.core.auth import verify_token
from api.utils.constants import SHARED

router = APIRouter(dependencies=[Depends(verify_token)])


@router.get("/agent-rag")
async def search_companies_endpoint(
        background_tasks: BackgroundTasks,
        query: str
):
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter cannot be empty.")

    try:
        agent = SHARED["extraction_agent"]
        supabase = SHARED["supabase_client"]
        result = agent.invoke(query)

        # Add new companies to the Supabase
        # background_tasks.add_task(add_apollo_companies_supabase, supabase, companies)

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
