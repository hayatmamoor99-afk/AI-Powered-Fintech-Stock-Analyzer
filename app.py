import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from decimal import Decimal
import random

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import bcrypt
import jwt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests
import openai
from redis import Redis

# -------------------- Configuration --------------------
load_dotenv()

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fintwin.db")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Database Setup --------------------
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# -------------------- Redis Client (caching) --------------------
redis_client = Redis(host='localhost', port=6379, decode_responses=True) if not os.getenv("DISABLE_REDIS") else None

# -------------------- Models --------------------
class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, index=True)  # Firebase UID or generated
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=True)  # if using local auth
    full_name = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    accounts = relationship("Account", back_populates="user")
    transactions = relationship("Transaction", back_populates="user")
    goals = relationship("Goal", back_populates="user")
    simulations = relationship("Simulation", back_populates="user")
    family_members = relationship("FamilyMember", back_populates="user")

class Account(Base):
    __tablename__ = "accounts"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    account_type = Column(String(50))  # checking, savings, investment, loan, credit_card
    name = Column(String(255))
    balance = Column(Float, default=0.0)
    currency = Column(String(3), default="USD")
    institution = Column(String(255))
    last_updated = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="accounts")

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    account_id = Column(String(36), ForeignKey("accounts.id"), nullable=True)
    amount = Column(Float)
    category = Column(String(100))
    description = Column(Text)
    transaction_date = Column(DateTime)
    is_income = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="transactions")

class Goal(Base):
    __tablename__ = "goals"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    name = Column(String(255))
    target_amount = Column(Float)
    current_amount = Column(Float, default=0.0)
    target_date = Column(DateTime)
    priority = Column(Integer, default=1)
    category = Column(String(100))  # retirement, house, education, etc.
    completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="goals")

class Simulation(Base):
    __tablename__ = "simulations"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    name = Column(String(255))
    parameters = Column(JSON)  # stores scenario inputs
    results = Column(JSON)     # stores simulation outputs
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="simulations")

class FamilyMember(Base):
    __tablename__ = "family_members"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    member_email = Column(String(255))
    relation = Column(String(50))  # spouse, child, parent
    permissions = Column(JSON)     # e.g., {"view": true, "edit": false}
    added_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="family_members")

# -------------------- Pydantic Schemas --------------------
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class AccountCreate(BaseModel):
    account_type: str
    name: str
    balance: float = 0.0
    currency: str = "USD"
    institution: Optional[str] = None

class TransactionCreate(BaseModel):
    account_id: Optional[str] = None
    amount: float
    category: str
    description: str
    transaction_date: datetime
    is_income: bool = False

class GoalCreate(BaseModel):
    name: str
    target_amount: float
    target_date: datetime
    priority: int = 1
    category: str

class SimulationRequest(BaseModel):
    name: str
    parameters: Dict[str, Any]  # e.g., {"income_growth": 0.03, "inflation": 0.02, ...}

class InvestmentAnalysisRequest(BaseModel):
    ticker: str
    period: str = "1y"

class ScamDetectionRequest(BaseModel):
    text: str
    context: Optional[str] = None  # email, sms, etc.

class ChatRequest(BaseModel):
    message: str

class HealthScoreResponse(BaseModel):
    score: int
    details: Dict[str, Any]

# -------------------- Security --------------------
security = HTTPBearer()

def get_password_hash(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

# -------------------- Dependency --------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------- App Initialization --------------------
app = FastAPI(title="FinTwin AI API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Helper Functions (Placeholders) --------------------
def generate_health_score(user_id: str, db: Session) -> Dict[str, Any]:
    """Mock health score calculation."""
    # In reality, we would fetch user's accounts, transactions, goals, etc.
    return {
        "score": random.randint(60, 95),
        "details": {
            "income_stability": random.randint(50, 100),
            "savings_habits": random.randint(50, 100),
            "debt_management": random.randint(50, 100),
            "investment_diversification": random.randint(50, 100),
            "emergency_preparedness": random.randint(50, 100),
            "spending_efficiency": random.randint(50, 100),
            "credit_health": random.randint(50, 100),
            "overall_resilience": random.randint(50, 100),
        }
    }

def run_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    """Mock future simulation."""
    # Real implementation would use monte carlo, etc.
    years = params.get("years", 10)
    inflation = params.get("inflation", 0.02)
    growth = params.get("growth", 0.05)
    initial_wealth = params.get("initial_wealth", 10000)
    monthly_savings = params.get("monthly_savings", 500)

    final_wealth = initial_wealth * (1 + growth - inflation) ** years + monthly_savings * 12 * ((1 + growth - inflation) ** years - 1) / (growth - inflation) if growth != inflation else initial_wealth + monthly_savings * 12 * years
    return {
        "final_wealth": round(final_wealth, 2),
        "projections": [round(initial_wealth * (1 + growth - inflation) ** y + monthly_savings * 12 * ((1 + growth - inflation) ** y - 1) / (growth - inflation) if growth != inflation else initial_wealth + monthly_savings * 12 * y, 2) for y in range(1, years+1)],
        "confidence": random.uniform(0.7, 0.95),
        "risk_metrics": {"volatility": random.uniform(0.1, 0.3)}
    }

def analyze_investment(ticker: str, period: str) -> Dict[str, Any]:
    """Fetch and analyze a ticker using Alpha Vantage or mock."""
    # Using mock data for demo
    return {
        "ticker": ticker,
        "period": period,
        "current_price": random.uniform(10, 500),
        "historical_return": random.uniform(-0.2, 0.4),
        "volatility": random.uniform(0.1, 0.4),
        "sharpe_ratio": random.uniform(0.5, 2.5),
        "sortino_ratio": random.uniform(0.3, 2.0),
        "max_drawdown": random.uniform(-0.3, -0.05),
        "var_95": random.uniform(-0.15, -0.02),
        "sector_allocation": {"Technology": 0.4, "Finance": 0.3, "Healthcare": 0.2, "Other": 0.1},
        "dividend_yield": random.uniform(0, 0.05),
        "recommendation": random.choice(["Buy", "Hold", "Sell"]),
        "explanation": "Based on historical performance and volatility, this asset aligns with a moderate risk profile."
    }

def detect_scam(text: str) -> Dict[str, Any]:
    """Mock scam detection using simple keyword checks."""
    suspicious_keywords = ["urgent", "click here", "verify account", "win", "prize", "inheritance", "bank account", "password", "phishing", "lottery"]
    found = [kw for kw in suspicious_keywords if kw in text.lower()]
    probability = min(0.95, len(found) * 0.1 + random.uniform(0, 0.1))
    return {
        "is_scam": probability > 0.6,
        "probability": probability,
        "detected_indicators": found,
        "explanation": f"Detected suspicious terms: {', '.join(found)}. This communication has a {probability*100:.1f}% chance of being fraudulent."
    }

def get_ai_response(message: str, user_id: str, db: Session) -> str:
    """Mock AI assistant response. Could integrate OpenAI."""
    if OPENAI_API_KEY:
        try:
            openai.api_key = OPENAI_API_KEY
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are FinTwin AI, a personal financial assistant. Provide concise, actionable financial advice."},
                    {"role": "user", "content": message}
                ],
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return "I'm sorry, I'm having trouble processing your request. Please try again later."
    else:
        # Fallback mock
        responses = [
            "Based on your current spending, you could save an additional $200 per month by reducing dining out.",
            "Your savings rate is excellent. Consider investing more in a diversified ETF.",
            "Paying off your high-interest credit card should be your top priority.",
            "You're on track to reach your retirement goal by age 62.",
        ]
        return random.choice(responses)

# -------------------- Routes --------------------
@app.post("/auth/register", response_model=Token)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    # Create user (using UUID for id, but simplified with email as id for demo)
    import uuid
    user_id = str(uuid.uuid4())
    hashed = get_password_hash(user_data.password)
    user = User(id=user_id, email=user_data.email, hashed_password=hashed, full_name=user_data.full_name)
    db.add(user)
    db.commit()
    db.refresh(user)
    # Create access token
    access_token = create_access_token(data={"sub": user.id})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/login", response_model=Token)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user.id})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=Dict[str, Any])
async def get_me(current_user: User = Depends(get_current_user)):
    return {"id": current_user.id, "email": current_user.email, "full_name": current_user.full_name}

@app.post("/accounts", response_model=Dict[str, Any])
async def create_account(account: AccountCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    import uuid
    acc = Account(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        account_type=account.account_type,
        name=account.name,
        balance=account.balance,
        currency=account.currency,
        institution=account.institution
    )
    db.add(acc)
    db.commit()
    db.refresh(acc)
    return {"id": acc.id, "message": "Account created"}

@app.post("/transactions", response_model=Dict[str, Any])
async def create_transaction(tx: TransactionCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    import uuid
    trans = Transaction(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        account_id=tx.account_id,
        amount=tx.amount,
        category=tx.category,
        description=tx.description,
        transaction_date=tx.transaction_date,
        is_income=tx.is_income
    )
    db.add(trans)
    db.commit()
    return {"message": "Transaction recorded"}

@app.get("/health-score", response_model=HealthScoreResponse)
async def get_health_score(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    score_data = generate_health_score(current_user.id, db)
    return HealthScoreResponse(score=score_data["score"], details=score_data["details"])

@app.post("/simulate", response_model=Dict[str, Any])
async def simulate(sim: SimulationRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    result = run_simulation(sim.parameters)
    import uuid
    sim_db = Simulation(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        name=sim.name,
        parameters=sim.parameters,
        results=result
    )
    db.add(sim_db)
    db.commit()
    return {"simulation_id": sim_db.id, "results": result}

@app.post("/invest/analyze", response_model=Dict[str, Any])
async def analyze_ticker(req: InvestmentAnalysisRequest, current_user: User = Depends(get_current_user)):
    # Optionally use Alpha Vantage
    if ALPHA_VANTAGE_API_KEY:
        # Example: fetch real data here
        # url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={req.ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        # ...
        pass
    result = analyze_investment(req.ticker, req.period)
    return result

@app.post("/scam-detect", response_model=Dict[str, Any])
async def scam_detect(req: ScamDetectionRequest, current_user: User = Depends(get_current_user)):
    return detect_scam(req.text)

@app.post("/assistant/chat", response_model=Dict[str, Any])
async def assistant_chat(req: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    response = get_ai_response(req.message, current_user.id, db)
    return {"response": response}

@app.post("/goals", response_model=Dict[str, Any])
async def create_goal(goal: GoalCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    import uuid
    g = Goal(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        name=goal.name,
        target_amount=goal.target_amount,
        target_date=goal.target_date,
        priority=goal.priority,
        category=goal.category
    )
    db.add(g)
    db.commit()
    db.refresh(g)
    return {"id": g.id, "message": "Goal created"}

@app.get("/goals", response_model=List[Dict[str, Any]])
async def list_goals(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    goals = db.query(Goal).filter(Goal.user_id == current_user.id).all()
    return [{"id": g.id, "name": g.name, "target_amount": g.target_amount, "current_amount": g.current_amount, "target_date": g.target_date} for g in goals]

@app.post("/family/add", response_model=Dict[str, Any])
async def add_family_member(member_email: str, relation: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Check if member exists
    member = db.query(User).filter(User.email == member_email).first()
    if not member:
        raise HTTPException(status_code=404, detail="User not found")
    # Add relationship
    import uuid
    fm = FamilyMember(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        member_email=member_email,
        relation=relation,
        permissions={"view": True, "edit": False}
    )
    db.add(fm)
    db.commit()
    return {"message": f"Added {member_email} as {relation}"}

# -------------------- Main --------------------
if __name__ == "__main__":
    # Create tables
    Base.metadata.create_all(bind=engine)
    uvicorn.run(app, host="0.0.0.0", port=8000)
