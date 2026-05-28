"""
Debug script: logs in via browser, captures one API page, dumps full JSON to debug_output.json.
Run: python debug_api.py
"""
import sys, json, getpass, os
sys.stdout.reconfigure(encoding='utf-8')
from playwright.sync_api import sync_playwright

def _load_env():
    env = {}
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip()
    return env

_env     = _load_env()
BASE_URL = _env.get('BASE_URL', 'https://m.fxpro1.net')
GAME_URL = _env.get('GAME_URL', f'{BASE_URL}/openHistory?gameName=DS3M')
CHROME   = _env.get('CHROME', '')
OUT_FILE = r'c:\Projects\docs\pocs\fxproapp\debug_output.json'

username = input("Username/Phone: ").strip()
password = getpass.getpass("Password: ").strip()

captured = {}

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=False, executable_path=CHROME)
    ctx = browser.new_context(
        user_agent='Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
        viewport={'width': 390, 'height': 844},
    )
    page = ctx.new_page()

    def on_response(response):
        if 'issue-result/page' in response.url and not captured:
            try:
                data = response.json()
                captured['data'] = data
                records = data.get('data', {}).get('records', [])
                print(f"\n[CAPTURED] {len(records)} records from: {response.url}")
            except Exception as e:
                print(f"[ERR] {e}")

    page.on('response', on_response)

    print("[1] Opening site...")
    page.goto(BASE_URL, wait_until='networkidle', timeout=30000)
    page.wait_for_timeout(2000)

    for sel in ['text=Login', 'a:has-text("Login")', '[href*="login"]']:
        try:
            el = page.query_selector(sel)
            if el and el.is_visible():
                el.click(); page.wait_for_timeout(1500); break
        except: pass

    for sel in ['input[type="text"]','input[name*="phone"]','input[name*="user"]','input[name*="account"]']:
        try:
            el = page.query_selector(sel)
            if el and el.is_visible():
                el.fill(username); break
        except: pass

    try:
        el = page.query_selector('input[type="password"]')
        if el and el.is_visible(): el.fill(password)
    except: pass

    for sel in ['button[type="submit"]','text=Login','button:has-text("Log")']:
        try:
            el = page.query_selector(sel)
            if el and el.is_visible():
                el.click(); break
        except: pass

    page.wait_for_timeout(5000)
    print("[2] Navigating to history...")
    page.goto(GAME_URL, wait_until='networkidle', timeout=30000)
    page.wait_for_timeout(6000)

    if not captured:
        input("No data captured yet. Navigate to history manually, then press Enter: ")
        page.wait_for_timeout(3000)

    browser.close()

if not captured:
    print("Nothing captured.")
else:
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(captured['data'], f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] Full response written to: {OUT_FILE}")

    # Print all field names and first 3 records in full
    records = captured['data'].get('data', {}).get('records', [])
    if records:
        print(f"\n=== ALL FIELDS in first record ===")
        for k, v in records[0].items():
            print(f"  {k!r:25s} : {v!r}")
        print(f"\n=== First 5 records (all fields) ===")
        for rec in records[:5]:
            print(json.dumps(rec, ensure_ascii=False))
