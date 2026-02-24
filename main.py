import os
import csv
import io
import base64
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, EmailStr
from sqlalchemy import (
    Column, Integer, String, Text, Numeric, DateTime, ForeignKey, select, desc
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

import httpx

# PDF generator
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from io import BytesIO

# Stripe (optional)
import stripe


# ---------------------------
# ENV / SETTINGS
# ---------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing in environment")

BREVO_API_KEY = os.getenv("BREVO_API_KEY", "").strip()
BREVO_SENDER_EMAIL = os.getenv("BREVO_SENDER_EMAIL", "").strip()
BREVO_SENDER_NAME = os.getenv("BREVO_SENDER_NAME", "Vortex AI").strip()

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://localhost:8080").strip()

DEAL_ROOM_TOKEN_HOURS = int(os.getenv("DEAL_ROOM_TOKEN_HOURS", "48"))
CITIES = [c.strip() for c in os.getenv("CITIES", "Dallas,Atlanta,Phoenix").split(",") if c.strip()]


# ---------------------------
# DB
# ---------------------------

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

async def get_db():
    async with SessionLocal() as session:
        yield session


# ---------------------------
# MODELS
# ---------------------------

class Buyer(Base):
    __tablename__ = "buyers"

    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)
    email = Column(Text, nullable=True, index=True)
    phone = Column(Text, nullable=True)

    city = Column(Text, nullable=True, index=True)
    state = Column(Text, nullable=True)

    tier = Column(Text, nullable=False, default="Weak Buyer")
    score = Column(Integer, nullable=False, default=0)
    tags = Column(Text, nullable=True)
    source = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class SellerLead(Base):
    __tablename__ = "seller_leads"

    id = Column(Integer, primary_key=True)
    full_name = Column(Text, nullable=False)
    email = Column(Text, nullable=True)
    phone = Column(Text, nullable=True)

    property_address = Column(Text, nullable=False)
    city = Column(Text, nullable=True, index=True)
    state = Column(Text, nullable=True)
    zip = Column(Text, nullable=True)

    asking_price = Column(Numeric, nullable=True)
    reason = Column(Text, nullable=True)
    condition_notes = Column(Text, nullable=True)
    source = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class Deal(Base):
    __tablename__ = "deals"

    id = Column(Integer, primary_key=True)

    seller_lead_id = Column(Integer, ForeignKey("seller_leads.id"), nullable=True)
    seller_lead = relationship("SellerLead", lazy="selectin")

    property_address = Column(Text, nullable=False)
    city = Column(Text, nullable=True, index=True)
    state = Column(Text, nullable=True)
    zip = Column(Text, nullable=True)

    arv = Column(Numeric, nullable=True)
    repairs = Column(Numeric, nullable=True)
    mao = Column(Numeric, nullable=True)
    offer_price = Column(Numeric, nullable=True)

    assignment_fee_target = Column(Numeric, nullable=True)
    estimated_spread = Column(Numeric, nullable=True)
    confidence = Column(Integer, nullable=False, default=0)

    status = Column(Text, nullable=False, default="NEW")
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class OutreachLog(Base):
    __tablename__ = "outreach_logs"

    id = Column(Integer, primary_key=True)
    deal_id = Column(Integer, ForeignKey("deals.id"), nullable=False)
    buyer_id = Column(Integer, ForeignKey("buyers.id"), nullable=True)

    channel = Column(Text, nullable=False, default="email")
    to_address = Column(Text, nullable=False)
    subject = Column(Text, nullable=True)
    status = Column(Text, nullable=False, default="SENT")
    provider_message_id = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class ContractDoc(Base):
    __tablename__ = "contract_docs"

    id = Column(Integer, primary_key=True)
    deal_id = Column(Integer, ForeignKey("deals.id"), nullable=False)

    doc_type = Column(Text, nullable=False, default="PURCHASE_AGREEMENT")
    filename = Column(Text, nullable=False)
    content_base64 = Column(Text, nullable=False)

    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class DealRoomToken(Base):
    __tablename__ = "deal_room_tokens"

    id = Column(Integer, primary_key=True)
    deal_id = Column(Integer, ForeignKey("deals.id"), nullable=False)

    token = Column(Text, nullable=False, unique=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)

    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class Commitment(Base):
    __tablename__ = "commitments"

    id = Column(Integer, primary_key=True)
    deal_id = Column(Integer, ForeignKey("deals.id"), nullable=False)
    buyer_id = Column(Integer, ForeignKey("buyers.id"), nullable=False)

    status = Column(Text, nullable=False, default="PENDING")
    proof_of_funds_base64 = Column(Text, nullable=True)

    assignment_fee_amount = Column(Numeric, nullable=True)
    stripe_checkout_url = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


# ---------------------------
# SCHEMAS
# ---------------------------

class BuyerCreateIn(BaseModel):
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    source: Optional[str] = "manual"


class SellerIntakeIn(BaseModel):
    full_name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None

    property_address: str
    city: str
    state: str
    zip: Optional[str] = None

    asking_price: Optional[float] = None
    reason: Optional[str] = None
    condition_notes: Optional[str] = None
    source: Optional[str] = "manual"


class DealCreateIn(BaseModel):
    property_address: str
    city: str
    state: str
    zip: Optional[str] = None
    seller_lead_id: Optional[int] = None
    notes: Optional[str] = None


class DealUnderwriteIn(BaseModel):
    arv: float
    repairs: float = 0.0
    investor_discount: float = 0.70
    assignment_fee_target: float = 10000.0


# ---------------------------
# HELPERS
# ---------------------------

INVESTOR_WORDS = ["LLC", "HOLDINGS", "INVEST", "INVESTMENTS", "PROPERTIES", "CAPITAL", "GROUP", "VENTURES"]

def is_investor_name(name: str) -> bool:
    up = (name or "").upper()
    return any(w in up for w in INVESTOR_WORDS)

def score_buyer(owner_name: str, flip_count: int = 0, last_flip_days: int | None = None) -> tuple[int, str, str]:
    score = 10
    tags = []

    if is_investor_name(owner_name):
        score += 25
        tags.append("investor_name")

    if flip_count >= 3:
        score += 35
        tags.append("multi_flip")
    elif flip_count == 2:
        score += 25
        tags.append("two_flip")
    elif flip_count == 1:
        score += 15
        tags.append("one_flip")

    if last_flip_days is not None:
        if last_flip_days <= 90:
            score += 25
            tags.append("recent_90d")
        elif last_flip_days <= 180:
            score += 15
            tags.append("recent_180d")

    score = max(0, min(100, score))
    tier = "VIP Buyer" if score >= 80 else ("Active Buyer" if score >= 50 else "Weak Buyer")
    return score, tier, ",".join(tags) if tags else ""

def calc_mao_offer(arv: float, repairs: float, investor_discount: float = 0.70) -> tuple[float, float]:
    mao = (arv * investor_discount) - repairs
    offer = mao
    return round(mao, 2), round(offer, 2)

def confidence_score(arv: float, repairs: float) -> int:
    score = 40
    if arv >= 200000:
        score += 10
    if repairs <= 25000:
        score += 15
    elif repairs <= 50000:
        score += 8
    return max(0, min(100, score))

def new_token() -> str:
    return secrets.token_urlsafe(24)

def token_expires_at() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=DEAL_ROOM_TOKEN_HOURS)

def generate_purchase_agreement_pdf(
    seller_name: str,
    property_address: str,
    city: str,
    state: str,
    zip_code: str | None,
    offer_price: float,
) -> tuple[str, str]:
    """
    TEMPLATE contract PDF. Have an attorney review for your state before using in real deals.
    Returns (filename, base64_pdf).
    """
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER

    y = height - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "PURCHASE AGREEMENT (TEMPLATE)")
    y -= 28

    c.setFont("Helvetica", 11)
    lines = [
        f"Seller: {seller_name}",
        "Buyer: Vortex AI Acquisition (Template)",
        "",
        f"Property: {property_address}, {city}, {state} {zip_code or ''}",
        f"Offer Price: ${offer_price:,.2f}",
        "",
        "Terms (Template):",
        "- Inspection/Due Diligence: ___ days",
        "- Closing Date: ___",
        "- Earnest Money: ___",
        "- Assignment permitted where lawful (consult attorney)",
        "",
        "Signatures:",
        "Seller: ____________________________   Date: __________",
        "Buyer:  ____________________________   Date: __________",
    ]
    for line in lines:
        c.drawString(50, y, line)
        y -= 16
        if y < 80:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 11)

    c.showPage()
    c.save()

    pdf_bytes = buf.getvalue()
    filename = "purchase_agreement_template.pdf"
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return filename, b64

async def send_brevo_email(to_email: str, subject: str, html: str) -> dict:
    # If not configured, we skip but keep system working
    if not BREVO_API_KEY or not BREVO_SENDER_EMAIL:
        return {"ok": False, "skipped": True, "reason": "Brevo not configured"}

    url = "https://api.brevo.com/v3/smtp/email"
    headers = {"api-key": BREVO_API_KEY, "Content-Type": "application/json", "accept": "application/json"}
    payload = {
        "sender": {"name": BREVO_SENDER_NAME, "email": BREVO_SENDER_EMAIL},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": html,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            return {"ok": False, "error": r.text, "status": r.status_code}
        return {"ok": True, "data": r.json()}

def create_assignment_checkout(deal_id: int, amount_usd: float) -> str | None:
    if not STRIPE_SECRET_KEY:
        return None
    stripe.api_key = STRIPE_SECRET_KEY

    session = stripe.checkout.Session.create(
        mode="payment",
        line_items=[{
            "price_data": {
                "currency": "usd",
                "product_data": {"name": f"Assignment Fee - Deal #{deal_id}"},
                "unit_amount": int(amount_usd * 100),
            },
            "quantity": 1
        }],
        success_url=f"{FRONTEND_BASE_URL}/success?deal_id={deal_id}",
        cancel_url=f"{FRONTEND_BASE_URL}/cancel?deal_id={deal_id}",
        metadata={"deal_id": str(deal_id), "type": "assignment_fee"},
    )
    return session.url


# ---------------------------
# FASTAPI APP
# ---------------------------

app = FastAPI(title="VortexAI Money Machine v1", version="1.0.0")

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.get("/")
def root():
    return {"ok": True, "service": "vortexai-money-v1", "cities": CITIES}


# ---------------------------
# BUYERS (Manual)
# ---------------------------

@app.post("/buyers")
async def create_buyer(payload: BuyerCreateIn, db: AsyncSession = Depends(get_db)):
    b = Buyer(
        name=payload.name,
        email=str(payload.email) if payload.email else None,
        phone=payload.phone,
        city=payload.city,
        state=payload.state,
        source=payload.source or "manual",
    )
    db.add(b)
    await db.commit()
    await db.refresh(b)
    return {"ok": True, "buyer_id": b.id}

@app.get("/buyers")
async def list_buyers(city: str | None = None, limit: int = 50, db: AsyncSession = Depends(get_db)):
    limit = min(max(limit, 1), 500)
    q = select(Buyer).order_by(Buyer.score.desc())
    if city:
        q = q.where(Buyer.city == city)
    q = q.limit(limit)
    res = await db.execute(q)
    buyers = res.scalars().all()
    return [{
        "id": b.id,
        "name": b.name,
        "email": b.email,
        "phone": b.phone,
        "city": b.city,
        "state": b.state,
        "tier": b.tier,
        "score": b.score,
        "tags": b.tags,
        "source": b.source,
    } for b in buyers]

@app.post("/buyers/import-csv")
async def import_buyers_csv(
    city: str = Query(..., description="Dallas / Atlanta / Phoenix"),
    state: str = Query(..., description="TX / GA / AZ"),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if city not in CITIES:
        raise HTTPException(400, f"City must be one of: {CITIES}")

    data = (await file.read()).decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(data))

    created = updated = skipped = 0

    for row in reader:
        name = (row.get("name") or row.get("owner_name") or "").strip()
        email = (row.get("email") or "").strip() or None
        phone = (row.get("phone") or "").strip() or None

        if not name:
            skipped += 1
            continue

        flip_count = int(float(row.get("flip_count") or 0))
        last_flip_days_raw = row.get("last_flip_days")
        last_flip_days = int(float(last_flip_days_raw)) if last_flip_days_raw not in (None, "", "null") else None

        score, tier, tags = score_buyer(name, flip_count=flip_count, last_flip_days=last_flip_days)

        # dedupe by email or by name+city
        existing = None
        if email:
            res = await db.execute(select(Buyer).where(Buyer.email == email).limit(1))
            existing = res.scalar_one_or_none()

        if not existing:
            res = await db.execute(select(Buyer).where(Buyer.name == name).where(Buyer.city == city).limit(1))
            existing = res.scalar_one_or_none()

        if existing:
            existing.phone = existing.phone or phone
            if score > existing.score:
                existing.score = score
                existing.tier = tier
            # merge tags
            merged = set(filter(None, (existing.tags or "").split(",") + (tags or "").split(",")))
            existing.tags = ",".join(sorted(merged)) if merged else existing.tags
            existing.source = existing.source or "csv"
            updated += 1
        else:
            db.add(Buyer(
                name=name,
                email=email,
                phone=phone,
                city=city,
                state=state,
                score=score,
                tier=tier,
                tags=tags,
                source="csv",
            ))
            created += 1

    await db.commit()
    return {"ok": True, "created": created, "updated": updated, "skipped": skipped}


# ---------------------------
# SELLERS (Manual)
# ---------------------------

@app.post("/sellers/intake")
async def seller_intake(payload: SellerIntakeIn, db: AsyncSession = Depends(get_db)):
    if payload.city not in CITIES:
        raise HTTPException(400, f"City must be one of: {CITIES}")

    lead = SellerLead(
        full_name=payload.full_name,
        email=str(payload.email) if payload.email else None,
        phone=payload.phone,
        property_address=payload.property_address,
        city=payload.city,
        state=payload.state,
        zip=payload.zip,
        asking_price=payload.asking_price,
        reason=payload.reason,
        condition_notes=payload.condition_notes,
        source=payload.source or "manual"
    )
    db.add(lead)
    await db.commit()
    await db.refresh(lead)

    deal = Deal(
        seller_lead_id=lead.id,
        property_address=lead.property_address,
        city=lead.city,
        state=lead.state,
        zip=lead.zip,
        status="NEW",
        notes="created_from_seller_intake",
        updated_at=datetime.now(timezone.utc)
    )
    db.add(deal)
    await db.commit()
    await db.refresh(deal)

    return {"ok": True, "seller_lead_id": lead.id, "deal_id": deal.id}


# ---------------------------
# DEALS (Manual)
# ---------------------------

@app.get("/deals")
async def list_deals(limit: int = 50, db: AsyncSession = Depends(get_db)):
    limit = min(max(limit, 1), 200)
    res = await db.execute(select(Deal).order_by(desc(Deal.id)).limit(limit))
    deals = res.scalars().all()
    return [{
        "id": d.id,
        "address": d.property_address,
        "city": d.city,
        "state": d.state,
        "status": d.status,
        "arv": float(d.arv) if d.arv is not None else None,
        "repairs": float(d.repairs) if d.repairs is not None else None,
        "offer_price": float(d.offer_price) if d.offer_price is not None else None,
        "confidence": d.confidence,
    } for d in deals]

@app.post("/deals/create")
async def create_deal(payload: DealCreateIn, db: AsyncSession = Depends(get_db)):
    if payload.city not in CITIES:
        raise HTTPException(400, f"City must be one of: {CITIES}")

    d = Deal(
        seller_lead_id=payload.seller_lead_id,
        property_address=payload.property_address,
        city=payload.city,
        state=payload.state,
        zip=payload.zip,
        status="NEW",
        notes=payload.notes,
        updated_at=datetime.now(timezone.utc)
    )
    db.add(d)
    await db.commit()
    await db.refresh(d)
    return {"ok": True, "deal_id": d.id}

@app.post("/deals/{deal_id}/underwrite")
async def underwrite(deal_id: int, payload: DealUnderwriteIn, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(Deal).where(Deal.id == deal_id).limit(1))
    d = res.scalar_one_or_none()
    if not d:
        raise HTTPException(404, "Deal not found")

    mao, offer = calc_mao_offer(payload.arv, payload.repairs, payload.investor_discount)
    conf = confidence_score(payload.arv, payload.repairs)

    d.arv = payload.arv
    d.repairs = payload.repairs
    d.mao = mao
    d.offer_price = offer
    d.assignment_fee_target = payload.assignment_fee_target
    d.estimated_spread = float(payload.assignment_fee_target)
    d.confidence = conf
    d.status = "UNDERWRITTEN"
    d.updated_at = datetime.now(timezone.utc)

    await db.commit()
    return {"ok": True, "deal_id": d.id, "mao": mao, "offer": offer, "confidence": conf, "status": d.status}

@app.post("/deals/{deal_id}/contract")
async def generate_contract(deal_id: int, db: AsyncSession = Depends(get_db)):
    dres = await db.execute(select(Deal).where(Deal.id == deal_id).limit(1))
    d = dres.scalar_one_or_none()
    if not d:
        raise HTTPException(404, "Deal not found")
    if d.offer_price is None:
        raise HTTPException(400, "Underwrite first (offer_price missing)")

    seller_name = "Seller"
    if d.seller_lead_id:
        sres = await db.execute(select(SellerLead).where(SellerLead.id == d.seller_lead_id).limit(1))
        s = sres.scalar_one_or_none()
        if s:
            seller_name = s.full_name

    filename, b64 = generate_purchase_agreement_pdf(
        seller_name=seller_name,
        property_address=d.property_address,
        city=d.city or "",
        state=d.state or "",
        zip_code=d.zip,
        offer_price=float(d.offer_price),
    )

    doc = ContractDoc(deal_id=d.id, doc_type="PURCHASE_AGREEMENT", filename=filename, content_base64=b64)
    db.add(doc)

    d.status = "OFFER_READY"
    d.updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(doc)
    return {"ok": True, "document_id": doc.id, "filename": doc.filename, "status": d.status}

@app.get("/deals/{deal_id}/contract/latest")
async def get_latest_contract(deal_id: int, db: AsyncSession = Depends(get_db)):
    res = await db.execute(
        select(ContractDoc)
        .where(ContractDoc.deal_id == deal_id)
        .order_by(desc(ContractDoc.id))
        .limit(1)
    )
    doc = res.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "No contract found")
    return {"ok": True, "filename": doc.filename, "content_base64": doc.content_base64}

@app.post("/deals/{deal_id}/blast")
async def blast_buyers(deal_id: int, db: AsyncSession = Depends(get_db)):
    dres = await db.execute(select(Deal).where(Deal.id == deal_id).limit(1))
    deal = dres.scalar_one_or_none()
    if not deal:
        raise HTTPException(404, "Deal not found")

    # create deal room token
    token = DealRoomToken(deal_id=deal.id, token=new_token(), expires_at=token_expires_at())
    db.add(token)
    await db.commit()
    await db.refresh(token)

    # fetch buyers in same city (email only)
    bres = await db.execute(
        select(Buyer)
        .where(Buyer.city == deal.city)
        .where(Buyer.email.is_not(None))
        .order_by(Buyer.score.desc())
        .limit(200)
    )
    buyers = list(bres.scalars().all())

    sent = 0
    skipped = 0

    for b in buyers:
        subject = f"Off-Market Deal in {deal.city}, {deal.state} - {deal.property_address}"
        deal_room_url = f"/deal-room/{token.token}"

        html = f"""
        <p>Hi {b.name},</p>
        <p>Off-market opportunity:</p>
        <ul>
          <li><b>Address:</b> {deal.property_address}, {deal.city}, {deal.state} {deal.zip or ''}</li>
          <li><b>ARV:</b> {float(deal.arv) if deal.arv else "TBD"}</li>
          <li><b>Repairs:</b> {float(deal.repairs) if deal.repairs else "TBD"}</li>
          <li><b>Contract Price:</b> ${float(deal.offer_price) if deal.offer_price else 0:,.2f}</li>
        </ul>
        <p><b>Deal Room:</b> <a href="{deal_room_url}">{deal_room_url}</a></p>
        <p>Reply “INTERESTED” and send Proof of Funds.</p>
        <p>- Vortex AI</p>
        """

        resp = await send_brevo_email(b.email, subject, html)
        status = "SENT" if resp.get("ok") else "SKIPPED"

        if status == "SENT":
            sent += 1
        else:
            skipped += 1

        db.add(OutreachLog(
            deal_id=deal.id,
            buyer_id=b.id,
            channel="email",
            to_address=b.email,
            subject=subject,
            status=status,
            provider_message_id=str((resp.get("data") or {}).get("messageId")) if resp.get("ok") else None
        ))

    deal.status = "BLASTED"
    deal.updated_at = datetime.now(timezone.utc)
    await db.commit()

    return {
        "ok": True,
        "deal_id": deal.id,
        "deal_room_token": token.token,
        "buyers_targeted": len(buyers),
        "sent": sent,
        "skipped": skipped,
        "note": "If Brevo not configured, emails are skipped but logs still created."
    }


# ---------------------------
# DEAL ROOM (Manual Link)
# ---------------------------

@app.get("/deal-room/{token}")
async def deal_room(token: str, db: AsyncSession = Depends(get_db)):
    tres = await db.execute(select(DealRoomToken).where(DealRoomToken.token == token).limit(1))
    t = tres.scalar_one_or_none()
    if not t:
        raise HTTPException(404, "Invalid token")

    if datetime.now(timezone.utc) > t.expires_at:
        raise HTTPException(410, "Token expired")

    dres = await db.execute(select(Deal).where(Deal.id == t.deal_id).limit(1))
    d = dres.scalar_one_or_none()
    if not d:
        raise HTTPException(404, "Deal not found")

    return {
        "deal_id": d.id,
        "address": d.property_address,
        "city": d.city,
        "state": d.state,
        "status": d.status,
        "arv": float(d.arv) if d.arv else None,
        "repairs": float(d.repairs) if d.repairs else None,
        "offer_price": float(d.offer_price) if d.offer_price else None,
    }


# ---------------------------
# STRIPE CHECKOUT (Optional)
# ---------------------------

@app.post("/deals/{deal_id}/assignment/checkout")
async def assignment_checkout(deal_id: int, buyer_id: int, amount_usd: float = 10000.0, db: AsyncSession = Depends(get_db)):
    url = create_assignment_checkout(deal_id=deal_id, amount_usd=float(amount_usd))
    if not url:
        raise HTTPException(400, "Stripe not configured (STRIPE_SECRET_KEY missing)")

    # upsert commitment
    cres = await db.execute(
        select(Commitment).where(Commitment.deal_id == deal_id).where(Commitment.buyer_id == buyer_id).limit(1)
    )
    c = cres.scalar_one_or_none()
    if not c:
        c = Commitment(
            deal_id=deal_id,
            buyer_id=buyer_id,
            status="PENDING",
            assignment_fee_amount=amount_usd,
            stripe_checkout_url=url
        )
        db.add(c)
    else:
        c.assignment_fee_amount = amount_usd
        c.stripe_checkout_url = url

    await db.commit()
    return {"ok": True, "stripe_checkout_url": url}
