import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "google-genai", "pandas", "yfinance", "requests"]
)

import pandas as pd
import requests
import yfinance as yf
from google import genai
from google.genai import types

BUY_SENTIMENT_MIN = 7
SELL_SENTIMENT_MAX = -6
VOLUME_SPIKE_RATIO = 1.8
RSI_OVERBOUGHT = 75.0
TARGET_PROFIT_PCT = 10.0
STOP_LOSS_PCT = -4.0
ALERT_COOLDOWN_SEC = 4 * 3600

PORTFOLIO = {
    # "005930.KS": {"name": "삼성전자", "avg_buy_price": 62000},
    # "NVDA": {"name": "엔비디아", "avg_buy_price": 135.5},
}

KR_QUERIES = ["특징주 수주", "실적 서프라이즈", "공급계약 체결", "신고가 호재"]
STATE_FILE = "state.json"


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"seen_news": [], "last_alert": {}}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def clean_html(text: str) -> str:
    for tag in ("<b>", "</b>", "&quot;", "&apos;", "&amp;", "&lt;", "&gt;"):
        text = text.replace(tag, "")
    return text.strip()


def fetch_kr_news():
    items = []
    ncp_id = os.getenv("NAVER_NCP_KEY_ID")
    ncp_secret = os.getenv("NAVER_NCP_KEY_SECRET")
    legacy_id = os.getenv("NAVER_CLIENT_ID")
    legacy_secret = os.getenv("NAVER_CLIENT_SECRET")

    if ncp_id and ncp_secret:
        url = "https://naverapihub.apigw.ntruss.com/search/v1/news"
        headers = {"X-NCP-APIGW-API-KEY-ID": ncp_id, "X-NCP-APIGW-API-KEY": ncp_secret}
    elif legacy_id and legacy_secret:
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {"X-Naver-Client-Id": legacy_id, "X-Naver-Client-Secret": legacy_secret}
    else:
        return items

    for q in KR_QUERIES:
        try:
            r = requests.get(url, headers=headers, params={"query": q, "display": 5, "sort": "date"}, timeout=10)
            r.raise_for_status()
            for row in r.json().get("items", []):
                link = row.get("originallink") or row.get("link", "")
                title = clean_html(row.get("title", ""))
                desc = clean_html(row.get("description", ""))
                if title and link:
                    items.append({"id": make_id(link), "src": "KR", "title": title, "summary": desc, "url": link})
        except Exception as e:
            print(f"[KR 뉴스 오류] {q}: {e}")
    return items


def fetch_us_news():
    items = []
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return items
    try:
        r = requests.get("https://finnhub.io/api/v1/news", params={"category": "general", "token": api_key}, timeout=10)
        r.raise_for_status()
        for row in r.json()[:15]:
            link = row.get("url", "")
            title = row.get("headline", "").strip()
            summary = row.get("summary", "").strip()[:500]
            if title and link:
                items.append({"id": make_id(link), "src": "US", "title": title, "summary": summary, "url": link})
    except Exception as e:
        print(f"[US 뉴스 오류] {e}")
    return items


def analyze_news(client, news):
    prompt = f"""
    한국 및 미국 주식 시장 퀀트 애널리스트로서 아래 뉴스를 분석해 JSON으로 반환하세요.
    - sentiment_score: -10(강한 악재) ~ +10(강한 호재) 정수. 단순 시황/루머는 0~3점.
    - stocks: 관련 상장 기업 최대 2개.
      * 미국 종목: 티커 그대로 (예: "NVDA", "AAPL")
      * 한국 코스피: 6자리 코드 + .KS (예: "005930.KS")
      * 한국 코스닥: 6자리 코드 + .KQ (예: "247540.KQ")
      * 비상장이거나 코드 불확실 시 빈 배열 []

    [시장]: {news['src']}
    [제목]: {news['title']}
    [내용]: {news['summary']}
    """
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "stocks": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "ticker": {"type": "STRING"},
                                "name": {"type": "STRING"},
                            },
                            "required": ["ticker", "name"],
                        },
                    },
                    "sentiment_score": {"type": "INTEGER"},
                    "reason": {"type": "STRING"},
                },
                "required": ["stocks", "sentiment_score", "reason"],
            },
            temperature=0.1,
        ),
    )
    return json.loads(resp.text)


def get_snapshot(ticker: str):
    try:
        hist = yf.Ticker(ticker).history(period="3mo", interval="1d")
        if hist is None or len(hist) < 21:
            return None
        closes = hist["Close"].astype(float)
        volumes = hist["Volume"].astype(float)
        price = float(closes.iloc[-1])
        ma20 = float(closes.rolling(20).mean().iloc[-1])
        avg_vol_5d = float(volumes.iloc[-6:-1].mean())
        cur_vol = float(volumes.iloc[-1])

        delta = closes.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi14 = float((100 - (100 / (1 + rs))).iloc[-1])

        return {
            "price": price,
            "ma20": ma20,
            "vol_ratio": cur_vol / max(avg_vol_5d, 1.0),
            "rsi14": rsi14,
        }
    except Exception as e:
        print(f"[시세 오류] {ticker}: {e}")
        return None


def send_telegram(text: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(text)
        return
    requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
        timeout=15,
    ).raise_for_status()


def main():
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

    client = genai.Client()
    state = load_state()
    seen = set(state.get("seen_news", []))
    last_alert = state.get("last_alert", {})
    now_ts = time.time()

    all_news = fetch_kr_news() + fetch_us_news()
    new_news = [n for n in all_news if n["id"] not in seen]
    print(f"전체 {len(all_news)}건 / 신규 {len(new_news)}건")

    for news in new_news[:12]:
        seen.add(news["id"])
        try:
            res = analyze_news(client, news)
        except Exception as e:
            print(f"[AI 오류] {e}")
            continue

        score = int(res.get("sentiment_score", 0))
        stocks = res.get("stocks", [])
        if not stocks or (SELL_SENTIMENT_MAX < score < BUY_SENTIMENT_MIN):
            continue

        for st in stocks:
            ticker = st["ticker"].strip().upper()
            name = st["name"].strip()
            if now_ts - float(last_alert.get(ticker, 0)) < ALERT_COOLDOWN_SEC:
                continue

            snap = get_snapshot(ticker)
            if not snap:
                continue

            is_kr = ticker.endswith(".KS") or ticker.endswith(".KQ")
            fmt = ",.0f" if is_kr else ",.2f"
            unit = "원" if is_kr else "달러"

            if ticker in PORTFOLIO and PORTFOLIO[ticker].get("avg_buy_price", 0) > 0:
                avg_buy = float(PORTFOLIO[ticker]["avg_buy_price"])
                pnl = ((snap["price"] - avg_buy) / avg_buy) * 100
                reasons = []
                if pnl >= TARGET_PROFIT_PCT:
                    reasons.append(f"목표 수익률 도달 ({pnl:+.1f}%)")
                if pnl <= STOP_LOSS_PCT:
                    reasons.append(f"손절선 이탈 ({pnl:+.1f}%)")
                if score <= SELL_SENTIMENT_MAX:
                    reasons.append(f"악재 뉴스 발생 ({score}점)")
                if snap["rsi14"] >= RSI_OVERBOUGHT:
                    reasons.append(f"RSI 과매수 ({snap['rsi14']:.1f})")
                if snap["price"] < snap["ma20"] and pnl > 0:
                    reasons.append("20일선 이탈 (차익실현 고려)")

                if reasons:
                    msg = (
                        f"🔴 [매도 타이밍] {name} ({ticker})\n"
                        f"• 현재가: {snap['price']:{fmt}}{unit} (평단 {avg_buy:{fmt}}{unit})\n"
                        f"• AI 요약: {res['reason']}\n"
                        f"• 근거: {', '.join(reasons)}\n"
                        f"• 원문: {news['url']}"
                    )
                    send_telegram(msg)
                    last_alert[ticker] = now_ts
                    continue

            if (
                score >= BUY_SENTIMENT_MIN
                and snap["price"] >= snap["ma20"]
                and snap["vol_ratio"] >= VOLUME_SPIKE_RATIO
                and snap["rsi14"] < RSI_OVERBOUGHT
            ):
                sl = snap["price"] * (1 + STOP_LOSS_PCT / 100)
                tp = snap["price"] * (1 + TARGET_PROFIT_PCT / 100)
                msg = (
                    f"🟢 [매수 타이밍] {name} ({ticker})\n"
                    f"• 현재가: {snap['price']:{fmt}}{unit}\n"
                    f"• 손절가({STOP_LOSS_PCT:+.0f}%): {sl:{fmt}}{unit} / 목표가({TARGET_PROFIT_PCT:+.0f}%): {tp:{fmt}}{unit}\n"
                    f"• AI 요약: {res['reason']}\n"
                    f"• 지표: 호재 {score}/10점 | 거래량 {snap['vol_ratio']:.1f}배 | RSI {snap['rsi14']:.1f}\n"
                    f"• 원문: {news['url']}"
                )
                send_telegram(msg)
                last_alert[ticker] = now_ts

    state["seen_news"] = list(seen)[-300:]
    state["last_alert"] = last_alert
    save_state(state)
    print(f"[{datetime.now(timezone.utc).isoformat()}] 완료")


if __name__ == "__main__":
    main()
