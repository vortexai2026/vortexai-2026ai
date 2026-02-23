from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Numeric, select
import os

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

app = FastAPI(title="VortexAI Clean System")

# -----------------------
# Database Dependency
# -----------------------

async def get_db():
    async with SessionLocal() as session:
        yield session

# -----------------------
# Deal Model
# -----------------------

class Deal(Base):
    __tablename__ = "deals"

    id = Column(Integer, primary_key=True, index=True)
    property_address = Column(String)
    city = Column(String)
    state = Column(String)
    arv = Column(Numeric)
    repairs = Column(Numeric)
    offer_price = Column(Numeric)
    status = Column(String, default="NEW")

# -----------------------
# Startup - Create Tables
# -----------------------

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# -----------------------
# Root
# -----------------------

@app.get("/")
def root():
    return {"ok": True}

# -----------------------
# Create Deal
# -----------------------

@app.post("/deals/create")
async def create_deal(data: dict, db: AsyncSession = Depends(get_db)):
    deal = Deal(
        property_address=data.get("property_address"),
        city=data.get("city"),
        state=data.get("state"),
        status="NEW"
    )
    db.add(deal)
    await db.commit()
    await db.refresh(deal)
    return {"deal_id": deal.id}

# -----------------------
# Underwrite
# -----------------------

@app.post("/deals/{deal_id}/underwrite")
async def underwrite(deal_id: int, data: dict, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Deal).where(Deal.id == deal_id))
    deal = result.scalar_one_or_none()

    if not deal:
        return {"error": "Deal not found"}

    arv = float(data.get("arv"))
    repairs = float(data.get("repairs", 0))

    mao = (arv * 0.70) - repairs

    deal.arv = arv
    deal.repairs = repairs
    deal.offer_price = mao
    deal.status = "UNDERWRITTEN"

    await db.commit()

    return {
        "deal_id": deal.id,
        "offer_price": mao,
        "status": deal.status
    }

# -----------------------
# List Deals
# -----------------------

@app.get("/deals")
async def list_deals(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Deal))
    deals = result.scalars().all()

    return [
        {
            "id": d.id,
            "address": d.property_address,
            "city": d.city,
            "state": d.state,
            "status": d.status,
            "offer_price": float(d.offer_price) if d.offer_price else None
        }
        for d in deals
    ]
