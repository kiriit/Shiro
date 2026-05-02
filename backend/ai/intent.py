import json
import os

import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

_SYSTEM_PROMPT = """\
你是一个智能日程助手的意图解析器，同时支持中文和英文输入。
分析用户的自然语言，提取意图，**只返回如下 JSON，不要任何额外文字**：

{
  "action": "create_event | add_memory | query | other",
  "title": "事件标题或内容摘要，没有则为 null",
  "start_time": "ISO8601，如 2024-01-15T14:00:00，没有则为 null",
  "end_time": "ISO8601，没有则为 null",
  "category": "exam | meeting | shopping | errand | other",
  "location": "地点，没有则为 null",
  "trigger": "条件触发词（如'下次去Costco'），仅 add_memory 使用，没有则为 null",
  "memo": "备注，没有则为 null"
}

action 规则：
- create_event：创建日历事件（明天开会、周五考试…）
- add_memory：带触发条件的备忘（下次去超市买牛奶…）
- query：查询日程或信息
- other：其他

category 规则：exam / meeting / shopping / errand / other\
"""


def parse_intent(user_input: str) -> dict:
    """Parse natural language input into a structured intent dict."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_input}],
    )

    raw = next(
        (block.text for block in response.content if block.type == "text"), ""
    ).strip()

    # Strip markdown fences if the model wraps the JSON
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) >= 2 else raw

    return json.loads(raw)
