from fastapi import FastAPI
from app.routes import buyers_router, sellers_router, deals_router, deal_room_router, stripe_webhook_router

app = FastAPI(title="VortexAI Money Machine", version="1.0.0")

app.include_router(buyers_router)
app.include_router(sellers_router)
app.include_router(deals_router)
app.include_router(deal_room_router)
app.include_router(stripe_webhook_router)

@app.get("/")
def root():
    return {"ok": True, "service": "vortexai-money"}
