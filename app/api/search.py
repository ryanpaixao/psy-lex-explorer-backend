from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.search import semantic_search

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    text: str
    score: float
    index: int

class SearchResponse(BaseModel):
    results: list[SearchResult]

@router.post("", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    try:
        results = await semantic_search(request.query, request.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )