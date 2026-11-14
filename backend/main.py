"""SmartMaint AI: 压缩机异常检测服务"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR") or Path(__file__).resolve().parent.parent / "serving" / "model")

LIMITATIONS = [
    "只有 4 次已报告故障,且故障 4 的传感器特征与前三次不同,指标不能外推。",
    "Isolation Forest 对故障 4 只检出约 41% 的行,规则基线在同一测试集上表现更好。",
    "规则的参数是看过故障特征后设定的,其表现偏乐观。",
    "这是故障发生后的快速发现,不是提前预警。",
    "测试期的正常数据已剔除卡死和未报告的持续加载时段,误报率偏乐观。",
]


class Reading(BaseModel):
    """一个时间点的 15 个传感器读数(列名与 MetroPT-3 原始数据一致,含 DV_eletric 的拼写)"""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    TP2: float
    TP3: float
    H1: float
    DV_pressure: float
    Reservoirs: float
    Oil_temperature: float
    Motor_current: float
    COMP: float
    DV_eletric: float
    Towers: float
    MPG: float
    LPS: float
    Pressure_switch: float
    Oil_level: float
    Caudal_impulses: float


class PredictRequest(BaseModel):
    equipment_id: str = Field(min_length=1, max_length=64)
    readings: list[Reading] = Field(
        min_length=1, max_length=5000,
        description="按时间从旧到新排列,每 10 秒一行;完整窗口需要 180 行(30 分钟)",
    )


class PredictResponse(BaseModel):
    equipment_id: str
    champion_version: int
    rows_received: int
    window_rows_used: int
    window_complete: bool
    latest_score: float
    score_threshold: float
    latest_flag: bool
    flagged_fraction: float
    if_alarm: Optional[bool]
    load_fraction: float
    rule_alarm: Optional[bool]
    note: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.pipe = joblib.load(MODEL_DIR / "model.joblib")
        app.state.info = json.loads((MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise RuntimeError(f"找不到模型文件,请先运行 mlops/export_champion.py: {e}")
    yield


app = FastAPI(title="SmartMaint AI", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "champion_version": request.app.state.info["version"]}


@app.get("/model-info")
def model_info(request: Request):
    info = request.app.state.info
    return {
        **info,
        "if_alarm_definition": "最近 window_rows 行中,超过 50% 的行分数高于 threshold",
        "rule_alarm_definition": "最近 window_rows 行中,DV_eletric 的平均值高于 rule_load_threshold",
        "known_limitations": LIMITATIONS,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    pipe = request.app.state.pipe
    info = request.app.state.info
    names = info["features"]
    win = info["window_rows"]

    X = np.array([[getattr(r, n) for n in names] for r in req.readings], dtype="float64")
    scores = -pipe.score_samples(X)
    flags = scores > info["threshold"]

    complete = len(req.readings) >= win
    flagged_fraction = float(flags[-win:].mean())
    load_fraction = float(X[-win:, names.index("DV_eletric")].mean())

    return PredictResponse(
        equipment_id=req.equipment_id,
        champion_version=int(info["version"]),
        rows_received=len(req.readings),
        window_rows_used=int(min(len(req.readings), win)),
        window_complete=complete,
        latest_score=float(scores[-1]),
        score_threshold=float(info["threshold"]),
        latest_flag=bool(flags[-1]),
        flagged_fraction=flagged_fraction,
        if_alarm=(flagged_fraction > 0.5) if complete else None,
        load_fraction=load_fraction,
        rule_alarm=(load_fraction > info["rule_load_threshold"]) if complete else None,
        note="ok" if complete else f"窗口不足 {win} 行,报警字段为 null",
    )