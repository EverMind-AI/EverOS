"""Build ``mixed_assistant_session.json`` — the mixed chat + tool-call corpus.

Why this fixture exists
-----------------------
Neither existing fixture models how EverOS is actually used. LoCoMo is two
humans chatting (every message is ``role="user"``, so the agent track never
fires); the SWE-bench trajectories carry a single machine-generated harness
prompt as their only user turn (so the user track yields a malformed
profile). Real traffic is one stream carrying both: a person talking to an
assistant that reaches for tools mid-conversation.

Structural constraints encoded below (each one is a hard gate upstream, not
a stylistic choice):

* Every session ends on a plain-text assistant message. ``_should_skip``
  rejects a memcell whose last item is anything else — including a
  ``ToolCallResult`` or a trailing user turn — with ``TRAJECTORY_NOT_CLOSED``
  (everalgo/agent_memory/case.py:206-231).
* Every session issues >= 3 tool-call rounds (``min_tool_call_rounds=3``,
  everalgo/agent_memory/case.py:75).
* User turns carry extractable personal facts, otherwise the profile comes
  out empty.
* Sessions sit days apart so the boundary detector has a natural seam to cut
  on without an intermediate ``/flush``, and so the 7-day clustering window
  (``clustering.time_window_days``) is exercised in both directions.
* Sessions 1 and 3 are the same kind of task (weekly report), so the second
  one drives ``AgentSkillExtractor`` down its ``update`` branch instead of
  minting a second skill.
* One session carries no tool calls at all. A cell like that must yield user
  memory and no agent memory — the dual-pipeline split under
  ``mode="agent"`` is otherwise untested.

Language mix
------------
Measured against real Claude Code session logs on this machine (3 sessions,
1608 tool calls, aggregate stats only):

===========================  ======  =============  ==========
category                          n  CJK-dominant   ASCII-only
===========================  ======  =============  ==========
user natural language           302            52%         19%
assistant natural language      656            41%          1%
tool_use arguments             1608             3%         56%
tool_result output             1695             1%         47%
===========================  ======  =============  ==========

So: the conversation layer is CJK-led but genuinely mixed (~29% of user turns
are neither CJK-dominant nor pure ASCII — they interleave English technical
terms), while the tool layer is essentially English. This corpus mirrors that
— English tool names, paths, flags and machine output; CJK prose with English
terms woven in; one pasted-log turn that is pure ASCII. Document *content*
returned by tools stays CJK where the document itself would be, which is what
a CJK-speaking user's files actually look like.

Regenerate with::

    uv run python tests/fixtures/_build_mixed_assistant_session.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OUT = Path(__file__).with_name("mixed_assistant_session.json")

USER = "u_liang"
AGENT = "assistant_daily"
TOOL = "tool_runner"

_STEP_MS = 45_000  # 45s between turns inside a session


def _cjk_ratio(text: str) -> float:
    """Share of letters that are CJK — used by the self-check output only."""
    letters = [ch for ch in text if ch.isalpha() or "一" <= ch <= "鿿"]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if "一" <= ch <= "鿿") / len(letters)


def _ts(day: str, hour: int, minute: int) -> int:
    y, m, d = (int(x) for x in day.split("-"))
    return int(datetime(y, m, d, hour, minute, tzinfo=UTC).timestamp() * 1000)


def user(text: str) -> dict[str, Any]:
    return {"sender_id": USER, "role": "user", "content": text}


def say(text: str) -> dict[str, Any]:
    return {"sender_id": AGENT, "role": "assistant", "content": text}


def call(call_id: str, name: str, **arguments: Any) -> dict[str, Any]:
    return {
        "sender_id": AGENT,
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def result(call_id: str, output: str) -> dict[str, Any]:
    return {
        "sender_id": TOOL,
        "role": "tool",
        "content": output,
        "tool_call_id": call_id,
    }


def parallel_calls(
    *specs: tuple[str, str, dict[str, Any], str],
) -> list[dict[str, Any]]:
    """One assistant turn issuing N tool calls at once, then N result turns.

    Each spec is ``(call_id, tool_name, arguments, output)``. Splat the return
    into a session list: ``*parallel_calls(spec_a, spec_b)``.

    Why this shape: ``MessageItemDTO.tool_calls`` is a list, so the OpenAI wire
    format puts N parallel calls on a single assistant message and answers them
    with N ``role="tool"`` messages carrying matching ``tool_call_id``s. That
    fan-out counts as **one** round to everalgo — ``_count_tool_call_rounds``
    counts ToolCallRequest messages, and its docstring notes "parallel calls
    within one request count once". So parallelising calls lowers the round
    count against ``min_tool_call_rounds=3``, which is exactly why the sessions
    below keep a margin above that floor.
    """
    assistant = {
        "sender_id": AGENT,
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
            for call_id, name, args, _ in specs
        ],
    }
    return [assistant, *(result(call_id, out) for call_id, _, _, out in specs)]


# ---------------------------------------------------------------------------
# Session 1 — work: turn meeting notes into a weekly report
# ---------------------------------------------------------------------------

SESSION_1 = [
    user(
        "早。我是做支付方向的 PM，上个月刚接手一个跨境结算的新项目，"
        "团队七个人，前端两个、后端三个、QA 两个。"
    ),
    say(
        "早上好。跨境结算涉及多币种和合规，"
        "需要我帮你把项目信息整理成一份 baseline 文档吗？"
    ),
    user("先不用。帮我把这周的会议记录整理成周报吧，记录都在 notes 目录下。"),
    call("call_w1_01", "list_files", path="notes/", pattern="2026-07-*.md"),
    result(
        "call_w1_01",
        "notes/2026-07-06-kickoff.md\nnotes/2026-07-08-riskcontrol.md\nnotes/2026-07-09-vendor.md",
    ),
    # Parallel fan-out: three independent files, read in one turn. This is what
    # a real assistant does — and it collapses to ONE tool-call round.
    *parallel_calls(
        (
            "call_w1_02",
            "read_file",
            {"path": "notes/2026-07-06-kickoff.md"},
            "# 跨境结算 Kickoff\n- 一期范围锁定：香港、新加坡两个通道\n"
            "- 里程碑：8/15 联调，9/1 灰度\n- 风险：新加坡持牌方合同还没签",
        ),
        (
            "call_w1_03",
            "read_file",
            {"path": "notes/2026-07-08-riskcontrol.md"},
            "# 风控评审\n- 反洗钱规则需接入第三方名单库\n"
            "- 单笔限额初定 5 万美元\n- 待办：法务确认新加坡本地留存要求",
        ),
        (
            "call_w1_04",
            "read_file",
            {"path": "notes/2026-07-09-vendor.md"},
            "# 供应商对接\n- 通道方 A 报价 0.35%，B 报价 0.42% 但结算 T+1\n"
            "- 倾向 A，需再压一轮价",
        ),
    ),
    # Sequential again: the assistant only knows to check last week's format
    # after seeing this week's material, so this genuinely cannot be batched.
    call("call_w1_05", "read_file", path="reports/2026-W27.md"),
    result(
        "call_w1_05",
        "# 第 27 周周报\n## 进展\n- 通道选型启动\n## 风险\n- 无\n"
        "## 下周计划\n- 完成 kickoff",
    ),
    call(
        "call_w1_06",
        "write_file",
        path="reports/2026-W28.md",
        content="# 第 28 周周报\n## 进展\n...\n## 风险\n...\n## 下周计划\n...",
    ),
    result("call_w1_06", "written: reports/2026-W28.md (1.4 KB)"),
    say(
        "周报已生成，写到 reports/2026-W28.md。三条主线：\n"
        "1. 一期范围锁定香港、新加坡双通道，里程碑 8/15 联调、9/1 灰度；\n"
        "2. 风控确定单笔限额 5 万美元，反洗钱名单库待接入；\n"
        "3. 通道方倾向 A（0.35%），建议再压一轮价。\n"
        "两个阻塞项我单独标了：新加坡持牌方合同未签、法务未确认本地数据留存要求。"
    ),
    user(
        "可以。我习惯周五下午写周报，以后到点提醒我一下。另外我这人不爱看长文档，"
        "能 bullet point 就别写段落。"
    ),
    say("记下了：每周五下午提醒你写周报，输出尽量用列表而不是大段文字。"),
]

# ---------------------------------------------------------------------------
# Session 2 — pure chat, zero tool calls.
#
# Must produce user memory (episode / atomic_fact / profile) and NO agent
# memory: ``_collect_agent_ids`` finds an assistant sender, but everalgo's
# no-tool gates reject the cell (NO_TOOL_SINGLE_USER / NO_TOOL_TOO_FEW_MESSAGES
# / NO_TOOL_ASSISTANT_TOO_SHORT). This is the dual-pipeline split that nothing
# else in the suite covers. The pasted-log turn also supplies the pure-ASCII
# user message the real-log distribution shows (19%).
# ---------------------------------------------------------------------------

SESSION_CHAT = [
    user(
        "问个事，跨境结算的对账一般是 T+1 还是 T+0？"
        "我们通道方给的是 T+1，财务有点意见。"
    ),
    say(
        "两种都有，取决于通道方的清算安排：T+0 需要预置备付金垫资，成本更高；"
        "T+1 是行业更普遍的做法。财务在意的通常是资金占用天数，"
        "可以让通道方给一份 settlement cut-off time 说明，"
        "很多时候 T+1 的实际到账比想象的早。"
    ),
    # Pure-ASCII turn: a pasted traceback with no prose around it. Real users
    # paste and expect the assistant to figure out the question.
    user(
        "ValueError: max() arg is an empty sequence\n"
        '  File "/app/settle/reconcile.py", line 88, in latest_batch\n'
        "    return max(b.closed_at for b in batches)\n"
        "  reconcile_daily(batches=[], cutoff=None)"
    ),
    say(
        "`batches` 是空序列时 `max()` 就会这样抛。给它一个 default 就行："
        "`max((b.closed_at for b in batches), default=None)`，"
        "然后在调用方处理 None。空批次在对账里通常意味着当天没有清算，"
        "属于正常状态而不是异常，所以别用 try/except 吞掉。"
    ),
    # English-led turn with a little CJK — the ~29% "mixed" shape.
    user(
        "makes sense. btw is `settlement cut-off time` the standard term? "
        "our vendor doc writes it as `cutoff`, 我怕对外文档不一致。"
    ),
    say(
        "行业文档里两种拼法都有，ISO 20022 相关材料多用 `cut-off time`，"
        "供应商合同里写 `cutoff` 也不算错。对外统一成一种就行，建议跟合同保持一致。"
    ),
    user("懂了，我让后端改。周末我要陪孩子去打疫苗，周一再看。"),
    say("好，周一见。疫苗那天要是想起来什么随时说。"),
]

# ---------------------------------------------------------------------------
# Session 2 — life: plan a family trip (different task, different cluster)
# ---------------------------------------------------------------------------

SESSION_2 = [
    user(
        "国庆后想带我爸妈去京都看红叶，我爸膝盖不好走不了太多路，帮我看看行程怎么安排。"
    ),
    # Two independent lookups, fired together. Note the CJK search queries:
    # tool *arguments* are English-shaped (keys, paths, flags) but a CJK user
    # searching CJK content is exactly what the 3% CJK-dominant argument
    # measurement in the real logs looks like.
    *parallel_calls(
        (
            "call_t2_01",
            "web_search",
            {"query": "京都 红叶 2026 最佳观赏期"},
            "多个来源：京都红叶见顷通常在 11 月中下旬；"
            "东福寺、永观堂 11/20 前后进入盛期；"
            "10 月上旬仅高海拔的大原、贵船略有转色。",
        ),
        (
            "call_t2_02",
            "web_search",
            {"query": "京都 轮椅友好 寺院 无障碍"},
            "天龙寺、平安神宫神苑地面平整，坡道较少；清水寺、伏见稻荷台阶多不推荐；"
            "岚山嵯峨野小火车有无障碍车厢需提前预约。",
        ),
    ),
    call("call_t2_03", "calendar_list", start="2026-10-01", end="2026-10-31"),
    result(
        "call_t2_03",
        "10/12-10/14 跨境结算联调复盘（全天）\n10/23 季度述职\n其余工作日无全天日程",
    ),
    call(
        "call_t2_04",
        "write_file",
        path="trips/2026-10-kyoto.md",
        content="# 京都 5 日（适老版）\n- 10/16 出发...\n",
    ),
    result("call_t2_04", "written: trips/2026-10-kyoto.md (0.9 KB)"),
    say(
        "行程草案存到 trips/2026-10-kyoto.md 了，按你爸的情况做了三处调整：\n"
        "- 时间选 10/16-10/20，避开你 10/12 复盘和 10/23 述职；\n"
        "- 只安排天龙寺、平安神宫这类地面平整的点，清水寺和伏见稻荷台阶太多，去掉了；\n"
        "- 嵯峨野小火车有无障碍车厢，但要提前预约，我标成待办了。\n"
        "有个取舍要提醒你：10 月中旬红叶还没到盛期，"
        "真要看红叶得 11 月下旬，那时你有述职。"
    ),
    user("红叶看不到就算了，主要是陪老人出去走走。我妈信佛，寺院多安排两个。"),
    say("明白，那按「陪伴优先」来排，寺院给你加到四个，都挑地面平整的。"),
]

# ---------------------------------------------------------------------------
# Session 3 — work again: same weekly-report task -> skill UPDATE branch
#
# The task must repeat (so the extractor sees a skill it already has) *and*
# contain a detour or a user correction. everalgo's LLM filter rejects a
# clean repeat outright — measured, with reason FILTER_REJECTED and
# llm_reason "The task followed a straightforward procedure with no user
# corrections or detours." Only trajectories above
# ``complex_task_round_threshold`` (20 rounds) bypass that filter, and this
# corpus runs 4-5 rounds. So the repeat is given a wrong-directory false
# start that the user corrects.
# ---------------------------------------------------------------------------

SESSION_3 = [
    user("又到周五了，这周的周报麻烦你，还是老规矩。"),
    call("call_w3_00", "list_files", path="notes/", pattern="2026-07-2*.md"),
    result("call_w3_00", "notes/2026-07-20-sync.md"),
    say(
        "notes/ 下这周只翻到 07-20 的同步纪要一份，法务和议价那两场会的记录没找到。"
        "是还没归档，还是放到别的地方了？"
    ),
    user("哦对，这周的我按周归到 notes/2026-w30/ 下面了，忘了跟你说。"),
    call("call_w3_01", "list_files", path="notes/2026-w30/", pattern="*.md"),
    result(
        "call_w3_01",
        "notes/2026-w30/2026-07-22-legal.md\nnotes/2026-w30/2026-07-23-pricing.md",
    ),
    # Parallel again — both files are known up front, so one turn reads both.
    *parallel_calls(
        (
            "call_w3_02",
            "read_file",
            {"path": "notes/2026-w30/2026-07-22-legal.md"},
            "# 法务同步\n- 新加坡本地留存要求确认：交易数据需境内留存 5 年\n"
            "- 持牌方合同已签，比计划晚 9 天",
        ),
        (
            "call_w3_03",
            "read_file",
            {"path": "notes/2026-w30/2026-07-23-pricing.md"},
            "# 二轮议价\n- 通道方 A 让步到 0.31%，但要求年度保底量\n"
            "- 财务认为保底量风险可接受",
        ),
    ),
    call(
        "call_w3_04",
        "write_file",
        path="reports/2026-W30.md",
        content="# 第 30 周周报\n## 进展\n...\n",
    ),
    result("call_w3_04", "written: reports/2026-W30.md (1.2 KB)"),
    say(
        "第 30 周周报好了，在 reports/2026-W30.md：\n"
        "- 上周两个阻塞项都解了：持牌方合同已签（晚 9 天），"
        "法务确认数据境内留存 5 年；\n"
        "- 通道方 A 议价到 0.31%，代价是年度保底量，财务判断风险可接受；\n"
        "- 合同晚 9 天，8/15 联调里程碑有压力，建议下周同步一次排期。\n"
        "另外记一下：你这周改成按周目录归档了，下次我直接去 notes/<周次>/ 找。"
    ),
    user("排期这事我来推。你把「合同晚了 9 天」这个风险记着，下次述职我要用。"),
    say("记下了，述职材料里我会把这条列进风险回顾。"),
]


def _build() -> dict[str, Any]:
    # Gaps are deliberately uneven — 3d / 9d / 2d. The 9-day gap crosses
    # ``clustering.time_window_days=7.0`` so the window's exclusion path is
    # exercised; the 3d and 2d gaps stay inside it so merging is too.
    # 07-10 and 07-24 are both Fridays, matching the user's stated habit of
    # writing the weekly report on Friday afternoons.
    sessions = [
        ("2026-07-10", 9, 12, "工作：会议记录整理成周报", SESSION_1),
        ("2026-07-13", 21, 40, "闲聊：无工具调用（只应产出 user 记忆）", SESSION_CHAT),
        ("2026-07-22", 20, 30, "生活：规划家庭旅行", SESSION_2),
        ("2026-07-24", 16, 5, "工作：再次整理周报（同类任务）", SESSION_3),
    ]
    batches = []
    for idx, (day, hour, minute, label, turns) in enumerate(sessions):
        base = _ts(day, hour, minute)
        messages = [
            {**turn, "timestamp": base + i * _STEP_MS} for i, turn in enumerate(turns)
        ]
        batches.append(
            {
                "batch_index": idx,
                "label": label,
                "date": day,
                "message_count": len(messages),
                "messages": messages,
            }
        )

    return {
        "source": "hand-authored for EverOS regression; models real mixed traffic",
        "license": "Apache-2.0 (same as the project)",
        "everos_session_id": "mixed_assistant_daily",
        "everos_user_sender_id": USER,
        "everos_agent_sender_id": AGENT,
        "everos_tool_sender_id": TOOL,
        "total_messages": sum(b["message_count"] for b in batches),
        "total_batches": len(batches),
        "design_notes": {
            "closure": "every session ends on a plain-text assistant turn "
            "(TRAJECTORY_NOT_CLOSED guard)",
            "tool_rounds": "each session issues >= 3 tool-call rounds "
            "(min_tool_call_rounds=3)",
            "skill_update": "batch 0 and batch 2 are the same weekly-report task, "
            "so batch 2 should drive the skill update branch",
            "clustering": "batch 0->1 is 8 days apart (outside time_window_days=7), "
            "batch 1->2 is 6 days (inside)",
        },
        "conversion_rules": {
            "role": "user / assistant / tool",
            "sender_id": f"user -> {USER}, assistant -> {AGENT}, tool -> {TOOL}",
            "timestamp": f"session base + idx * {_STEP_MS}ms",
            "content": "empty string on assistant tool-call-only turns",
            "tool_calls": "OpenAI Chat Completions shape, on assistant rows",
            "tool_call_id": "on tool rows",
        },
        "batches": batches,
    }


def _verify(payload: dict[str, Any]) -> None:
    """Fail loudly on the structural mistakes that are invisible by eye.

    Duplicate ``tool_call_id`` is the dangerous one: it silently mispairs a
    result with the wrong call and nothing downstream complains.
    """
    seen_ids: set[str] = set()
    for b in payload["batches"]:
        msgs = b["messages"]
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                cid = tc["id"]
                if cid in seen_ids:
                    raise AssertionError(f"duplicate tool_call_id: {cid}")
                seen_ids.add(cid)

        call_ids = [tc["id"] for m in msgs for tc in (m.get("tool_calls") or [])]
        result_ids = [m["tool_call_id"] for m in msgs if m["role"] == "tool"]
        if sorted(call_ids) != sorted(result_ids):
            raise AssertionError(
                f"batch {b['batch_index']}: calls {sorted(call_ids)} "
                f"!= results {sorted(result_ids)}"
            )

        last = msgs[-1]
        if last["role"] != "assistant" or last.get("tool_calls"):
            raise AssertionError(
                f"batch {b['batch_index']} not closed — last item is "
                f"{last['role']}"
                f"{'+tool_calls' if last.get('tool_calls') else ''} "
                "(TRAJECTORY_NOT_CLOSED would reject the cell)"
            )

        rounds = sum(1 for m in msgs if m.get("tool_calls"))
        if rounds and rounds < 3:
            raise AssertionError(
                f"batch {b['batch_index']}: {rounds} tool-call round(s), "
                "below min_tool_call_rounds=3"
            )

        # tool results must directly follow their issuing assistant turn
        for i, m in enumerate(msgs):
            n = len(m.get("tool_calls") or [])
            if not n:
                continue
            following = msgs[i + 1 : i + 1 + n]
            if len(following) != n or any(f["role"] != "tool" for f in following):
                raise AssertionError(
                    f"batch {b['batch_index']}: {n} call(s) at index {i} are "
                    "not immediately followed by that many tool results"
                )


def main() -> None:
    payload = _build()
    _verify(payload)
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {OUT} — {payload['total_batches']} batches / "
        f"{payload['total_messages']} messages"
    )
    for b in payload["batches"]:
        msgs = b["messages"]
        rounds = sum(1 for m in msgs if m.get("tool_calls"))
        n_calls = sum(len(m.get("tool_calls") or []) for m in msgs)
        fanout = [len(m["tool_calls"]) for m in msgs if m.get("tool_calls")]
        prose = [m["content"] for m in msgs if m["role"] != "tool" and m["content"]]
        cjk_led = sum(1 for t in prose if _cjk_ratio(t) > 0.3)
        pure_ascii = sum(1 for t in prose if t.isascii())
        print(
            f"  batch {b['batch_index']} ({b['date']}): {b['message_count']:>2} msgs  "
            f"— {b['label']}"
        )
        print(
            f"{'':>12}tool: {rounds} round(s) / {n_calls} call(s)  "
            f"fanout={fanout}  parallel={'yes' if any(f > 1 for f in fanout) else 'no'}"
        )
        print(
            f"{'':>12}prose: {len(prose)} turns  cjk_led={cjk_led}  "
            f"pure_ascii={pure_ascii}  mixed={len(prose) - cjk_led - pure_ascii}"
        )


if __name__ == "__main__":
    main()
