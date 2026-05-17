"""Pre-check: translate hard sentences on two llama-server endpoints, compare."""
import json, urllib.request

SENTS = [
    ("zh", '增加了资料性附录 "通用规范汉字表 ; 议字的代码位置" 见附录刁。'),
    ("zh", '信息扩术 "中文编码字符集'),
    ("zh", "表 2 双字闻部分的码位安排"),
    ("zh", "人工智能正在深刻改变全球经济格局。"),
    ("en", "The committee will review the proposal next week."),
]
PROMPT = {"zh": "Translate the following text from Chinese to English.\nChinese: {s}\nEnglish:",
          "en": "Translate the following text from English to Chinese.\nEnglish: {s}\nChinese:"}
SERVERS = {"8080 CPU": "http://127.0.0.1:8080/completion",
           "8081 CUDA-fa-off": "http://127.0.0.1:8081/completion"}


def tr(ep, text, sl):
    p = f"<|im_start|>user\n{PROMPT[sl].format(s=text)}<|im_end|>\n<|im_start|>assistant\n"
    body = json.dumps({"prompt": p, "n_predict": 200, "temperature": 0,
                       "stop": ["<|im_end|>"]}).encode()
    r = json.load(urllib.request.urlopen(
        urllib.request.Request(ep, body, {"Content-Type": "application/json"}), timeout=180))
    return r["content"].strip(), r.get("tokens_predicted"), r.get("stop_type")


for sl, s in SENTS:
    print(f"\nSRC: {s}")
    for name, ep in SERVERS.items():
        try:
            txt, ntok, stop = tr(ep, s, sl)
            flag = " <<< 跑飞?" if ntok and ntok >= 195 else ""
            print(f"  [{name}] ({ntok}tok,{stop}){flag} {txt[:160]}")
        except Exception as e:
            print(f"  [{name}] ERROR {e}")
