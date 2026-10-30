from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core import (
    ApiConfig,
    OpenAICompatibleClient,
    annotate_record,
    build_prompt_optimizer_messages,
    dataframe_to_excel,
    make_json_compatible,
    parse_field_specs,
    parse_json_object,
)
from local_config import clear_settings, config_path, load_settings, save_settings


APP_ICON = Path(__file__).with_name("app_icon.png")

st.set_page_config(
    page_title="Excel AI 半自动标注工具",
    page_icon=str(APP_ICON) if APP_ICON.exists() else None,
    layout="wide",
    initial_sidebar_state="auto",
)


DEFAULT_SYSTEM_PROMPT = """你是严谨的数据标注员。请根据标注规范判断输入数据。

要求：
1. 只依据输入内容，不使用无法验证的外部假设。
2. 严格遵守标签定义和边界条件。
3. 信息不足、存在多种合理解释或不符合规范时，主动标记为需要人工复核。
4. 相同情况必须使用一致的判断标准。"""

DEFAULT_USER_TEMPLATE = """请标注下面这条数据：

{input_json}"""

DEFAULT_FIELDS = """[
  {
    "name": "label",
    "description": "主标签。请在 allowed_values 中选择",
    "type": "string",
    "allowed_values": ["类别A", "类别B", "其他"],
    "required": true
  },
  {
    "name": "reason",
    "description": "简短说明判断依据",
    "type": "string",
    "required": true
  }
]"""

PROVIDER_PRESETS = {
    "OpenAI": ("https://api.openai.com/v1", "gpt-4.1-mini"),
    "DeepSeek": ("https://api.deepseek.com/v1", "deepseek-chat"),
    "通义千问兼容接口": (
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen-plus",
    ),
    "自定义 OpenAI 兼容接口": ("", ""),
}

NAV_ITEMS = ["API 接入", "Prompt 调试", "数据导入", "批量标注", "复核与导出"]


def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "provider": "DeepSeek",
        "api_key": "",
        "base_url": PROVIDER_PRESETS["DeepSeek"][0],
        "model": PROVIDER_PRESETS["DeepSeek"][1],
        "api_timeout": 60,
        "json_mode": True,
        "api_verified": False,
        "api_status": "未测试",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "user_template": DEFAULT_USER_TEMPLATE,
        "fields_json": DEFAULT_FIELDS,
        "sample_input": '{"text": "这里填写一条用于测试的客户数据"}',
        "expected_output": "",
        "prompt_feedback": "",
        "last_test_output": "",
        "last_test_result": None,
        "prompt_versions": [],
        "pending_prompt": None,
        "data_bytes": None,
        "data_file_name": "",
        "data_hash": "",
        "sheet_names": [],
        "data_df": None,
        "input_columns": [],
        "id_column": "（使用行号）",
        "data_ready": False,
        "results": {},
        "review_revision": 0,
        "remember_api_key": False,
        "local_config_status": "尚未保存本机配置",
    }
    if "_local_config_loaded" not in st.session_state:
        saved = load_settings()
        for key in defaults:
            if key in saved:
                defaults[key] = saved[key]
        st.session_state._local_config_loaded = True
        if saved:
            defaults["local_config_status"] = "已加载本机配置"
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    pending = st.session_state.get("pending_prompt")
    if pending:
        st.session_state.system_prompt = pending["system_prompt"]
        st.session_state.user_template = pending["user_template"]
        st.session_state.pending_prompt = None


def local_settings_payload() -> dict[str, Any]:
    return {
        "provider": st.session_state.provider,
        "base_url": st.session_state.base_url,
        "model": st.session_state.model,
        "api_timeout": int(st.session_state.api_timeout),
        "json_mode": bool(st.session_state.json_mode),
        "system_prompt": st.session_state.system_prompt,
        "user_template": st.session_state.user_template,
        "fields_json": st.session_state.fields_json,
    }


def save_local_defaults() -> None:
    try:
        parse_field_specs(st.session_state.fields_json)
        path = save_settings(
            local_settings_payload(),
            st.session_state.api_key,
            bool(st.session_state.remember_api_key),
        )
        st.session_state.local_config_status = f"已保存到 {path}"
        st.toast("本机配置已保存")
    except Exception as exc:
        st.session_state.local_config_status = f"保存失败：{exc}"
        st.error(st.session_state.local_config_status)


def clear_local_defaults() -> None:
    if clear_settings():
        st.session_state.local_config_status = "本机保存已清除；当前会话保持不变"
        st.toast("本机配置和已保存的密钥已清除")
    else:
        st.session_state.local_config_status = "本机配置清除失败"


def api_config() -> ApiConfig:
    return ApiConfig(
        api_key=st.session_state.api_key,
        base_url=st.session_state.base_url,
        model=st.session_state.model,
        timeout=int(st.session_state.api_timeout),
        temperature=0.0,
        json_mode=bool(st.session_state.json_mode),
        max_retries=2,
    )


def require_api() -> OpenAICompatibleClient | None:
    if not st.session_state.api_key.strip():
        st.error("请先在“API 接入”中填写 API Key。")
        return None
    if not st.session_state.base_url.strip() or not st.session_state.model.strip():
        st.error("API Base URL 和模型名称不能为空。")
        return None
    try:
        return OpenAICompatibleClient(api_config())
    except ValueError as exc:
        st.error(str(exc))
        return None


def parsed_fields_or_error():
    try:
        return parse_field_specs(st.session_state.fields_json)
    except ValueError as exc:
        st.error(str(exc))
        return None


def parse_sample(value: str) -> dict[str, Any]:
    value = value.strip()
    if not value:
        return {"text": ""}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {"text": value}


def result_dataframe() -> pd.DataFrame:
    source = st.session_state.data_df
    if source is None:
        return pd.DataFrame()
    result = source.copy().reset_index(drop=True)
    result.insert(0, "_source_index", range(len(result)))
    rows: list[dict[str, Any]] = []
    for index in range(len(result)):
        item = st.session_state.results.get(index)
        row = dict(item) if item else {}
        row["AI_处理状态"] = (
            "失败" if item and item.get("AI_错误") else "已标注" if item else "未处理"
        )
        rows.append(row)
    ai_df = pd.DataFrame(rows)
    return pd.concat([result, ai_df], axis=1)


def unresolved_review_dataframe(full_df: pd.DataFrame) -> pd.DataFrame:
    if full_df.empty or "AI_需复核" not in full_df.columns:
        return full_df.iloc[0:0].copy()
    mask = full_df["AI_需复核"].fillna(False).astype(bool)
    return full_df.loc[mask].copy()


def show_api_page() -> None:
    st.header("API 接入")
    left, right = st.columns([2, 1])
    with left:
        selected = st.selectbox(
            "服务商预设",
            list(PROVIDER_PRESETS),
            index=list(PROVIDER_PRESETS).index(st.session_state.provider),
        )
        if st.button("应用预设", use_container_width=False):
            st.session_state.provider = selected
            st.session_state.base_url, st.session_state.model = PROVIDER_PRESETS[selected]
            st.session_state.api_verified = False
            st.rerun()
        st.text_input("API Key", type="password", key="api_key")
        st.text_input("API Base URL", key="base_url")
        st.text_input("模型名称", key="model")

    with right:
        st.number_input("请求超时（秒）", min_value=10, max_value=300, step=10, key="api_timeout")
        st.toggle("启用 JSON Mode", key="json_mode")
        st.caption("启用“安全记住 API Key”后，密钥由 Windows 当前用户加密保存；导出文件始终不包含密钥。")
        if st.session_state.api_verified:
            st.success(st.session_state.api_status)
        else:
            st.info(st.session_state.api_status)

    if st.button("测试连接", type="primary"):
        client = require_api()
        if client:
            with st.spinner("正在测试 API..."):
                try:
                    content, usage = client.chat(
                        [
                            {"role": "system", "content": "只回答 OK。"},
                            {"role": "user", "content": "连接测试"},
                        ],
                        json_mode=False,
                        max_tokens=20,
                    )
                    st.session_state.api_verified = True
                    st.session_state.api_status = f"连接成功：{content.strip()[:50]}"
                    st.success(st.session_state.api_status)
                    if usage:
                        st.json(usage)
                except Exception as exc:
                    st.session_state.api_verified = False
                    st.session_state.api_status = f"连接失败：{exc}"
                    st.error(st.session_state.api_status)


def show_prompt_page() -> None:
    st.header("Prompt 调试")
    editor, schema = st.columns([3, 2])
    with editor:
        st.text_area("System Prompt", height=260, key="system_prompt")
        st.text_area("用户 Prompt 模板", height=170, key="user_template")
        if "{input_json}" not in st.session_state.user_template:
            st.warning("模板中未找到 {input_json}，程序会自动把输入数据追加到末尾。")
    with schema:
        st.text_area("标注字段配置（JSON）", height=455, key="fields_json")
        try:
            fields = parse_field_specs(st.session_state.fields_json)
            st.success(f"字段配置有效，共 {len(fields)} 个输出字段。")
        except ValueError as exc:
            fields = None
            st.error(str(exc))

    st.divider()
    sample_col, criteria_col = st.columns(2)
    with sample_col:
        st.text_area("测试数据（JSON 或文本）", height=145, key="sample_input")
    with criteria_col:
        st.text_area("期望结果或优化标准", height=145, key="expected_output")

    test_button, optimize_button = st.columns(2)
    with test_button:
        run_test = st.button("运行当前 Prompt 测试", type="primary", use_container_width=True)
    with optimize_button:
        optimize = st.button("让 AI 优化 Prompt", use_container_width=True)

    if run_test:
        client = require_api()
        if client and fields:
            record = parse_sample(st.session_state.sample_input)
            with st.spinner("正在测试..."):
                result = annotate_record(
                    client,
                    st.session_state.system_prompt,
                    st.session_state.user_template,
                    record,
                    fields,
                    confidence_threshold=0.75,
                )
            st.session_state.last_test_result = result.to_flat_dict()
            st.session_state.last_test_output = result.raw_response or result.error

    if st.session_state.last_test_result:
        result = st.session_state.last_test_result
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("置信度", f"{float(result.get('AI_置信度', 0)):.2f}")
        metric2.metric("需要复核", "是" if result.get("AI_需复核") else "否")
        metric3.metric("请求状态", "失败" if result.get("AI_错误") else "成功")
        visible = {key: value for key, value in result.items() if key.startswith("AI_")}
        st.json(make_json_compatible(visible))

    st.text_area("对本轮结果的反馈", height=90, key="prompt_feedback")
    if optimize:
        client = require_api()
        if client and fields:
            messages = build_prompt_optimizer_messages(
                st.session_state.system_prompt,
                st.session_state.user_template,
                st.session_state.fields_json,
                st.session_state.sample_input,
                st.session_state.expected_output,
                st.session_state.last_test_output,
                st.session_state.prompt_feedback,
            )
            with st.spinner("正在生成优化版本..."):
                try:
                    raw, _usage = client.chat(messages, temperature=0.2, max_tokens=3500)
                    improved = parse_json_object(raw)
                    new_system = str(improved.get("optimized_system_prompt", "")).strip()
                    new_user = str(improved.get("optimized_user_template", "")).strip()
                    if not new_system or not new_user:
                        raise ValueError("优化响应缺少 Prompt 内容")
                    if "{input_json}" not in new_user:
                        new_user = new_user.rstrip() + "\n\n{input_json}"
                    st.session_state.prompt_versions.append(
                        {
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "system_prompt": st.session_state.system_prompt,
                            "user_template": st.session_state.user_template,
                            "summary": str(improved.get("change_summary", "优化前版本")),
                        }
                    )
                    st.session_state.pending_prompt = {
                        "system_prompt": new_system,
                        "user_template": new_user,
                    }
                    st.session_state.last_test_result = None
                    st.session_state.last_test_output = ""
                    st.rerun()
                except Exception as exc:
                    st.error(f"Prompt 优化失败：{exc}")

    if st.session_state.prompt_versions:
        with st.expander(f"历史版本（{len(st.session_state.prompt_versions)}）"):
            options = list(range(len(st.session_state.prompt_versions) - 1, -1, -1))
            selected_index = st.selectbox(
                "选择历史版本",
                options,
                format_func=lambda index: (
                    f"{st.session_state.prompt_versions[index]['time']} · "
                    f"{st.session_state.prompt_versions[index]['summary'][:50]}"
                ),
            )
            if st.button("恢复所选版本"):
                version = st.session_state.prompt_versions[selected_index]
                st.session_state.pending_prompt = {
                    "system_prompt": version["system_prompt"],
                    "user_template": version["user_template"],
                }
                st.rerun()


def read_uploaded_file(file_bytes: bytes, file_name: str, sheet: str | None, header_row: int) -> pd.DataFrame:
    suffix = file_name.lower().rsplit(".", 1)[-1]
    if suffix == "csv":
        for encoding in ("utf-8-sig", "gb18030", "utf-8"):
            try:
                return pd.read_csv(BytesIO(file_bytes), encoding=encoding, header=header_row)
            except UnicodeDecodeError:
                continue
        raise ValueError("无法识别 CSV 编码")
    return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet, header=header_row)


def show_data_page() -> None:
    st.header("数据导入")
    uploaded = st.file_uploader("上传 Excel 或 CSV", type=["xlsx", "xlsm", "xls", "csv"])
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        digest = hashlib.sha256(file_bytes).hexdigest()
        if digest != st.session_state.data_hash:
            st.session_state.data_bytes = file_bytes
            st.session_state.data_file_name = uploaded.name
            st.session_state.data_hash = digest
            st.session_state.data_df = None
            st.session_state.data_ready = False
            st.session_state.results = {}
            if uploaded.name.lower().endswith((".xlsx", ".xlsm", ".xls")):
                try:
                    st.session_state.sheet_names = pd.ExcelFile(BytesIO(file_bytes)).sheet_names
                except Exception as exc:
                    st.error(f"无法读取工作表：{exc}")
                    st.session_state.sheet_names = []
            else:
                st.session_state.sheet_names = []

    if not st.session_state.data_bytes:
        st.info("等待上传数据文件。")
        return

    settings1, settings2 = st.columns(2)
    with settings1:
        sheet = None
        if st.session_state.sheet_names:
            sheet = st.selectbox("工作表", st.session_state.sheet_names)
    with settings2:
        header_display = st.number_input("表头所在行", min_value=1, max_value=100, value=1)

    if st.button("读取数据", type="primary"):
        try:
            loaded = read_uploaded_file(
                st.session_state.data_bytes,
                st.session_state.data_file_name,
                sheet,
                int(header_display) - 1,
            )
            if loaded.empty:
                raise ValueError("文件中没有数据行")
            loaded.columns = [str(column).strip() for column in loaded.columns]
            loaded = loaded.reset_index(drop=True)
            st.session_state.data_df = loaded
            st.session_state.input_columns = list(loaded.columns[: min(3, len(loaded.columns))])
            st.session_state.data_ready = False
            st.session_state.results = {}
        except Exception as exc:
            st.error(f"读取失败：{exc}")

    if st.session_state.data_df is not None:
        data = st.session_state.data_df
        stat1, stat2, stat3 = st.columns(3)
        stat1.metric("数据行", f"{len(data):,}")
        stat2.metric("字段列", len(data.columns))
        stat3.metric("文件", st.session_state.data_file_name)
        selected_columns = st.multiselect(
            "发送给 AI 的输入列",
            options=list(data.columns),
            default=[column for column in st.session_state.input_columns if column in data.columns],
        )
        id_options = ["（使用行号）"] + list(data.columns)
        id_column = st.selectbox("记录 ID 列", id_options)
        st.dataframe(data.head(50), use_container_width=True, hide_index=True)
        if st.button("确认导入配置", type="primary"):
            if not selected_columns:
                st.error("至少选择一个输入列。")
            else:
                st.session_state.input_columns = selected_columns
                st.session_state.id_column = id_column
                st.session_state.data_ready = True
                st.success("数据配置已确认，可以进入批量标注。")


def run_indices(indices: list[int], threshold: float, concurrency: int) -> None:
    client = require_api()
    fields = parsed_fields_or_error()
    if not client or not fields:
        return
    source: pd.DataFrame = st.session_state.data_df
    selected_columns = st.session_state.input_columns
    progress = st.progress(0, text="准备请求...")
    status = st.empty()
    completed = 0

    def task(index: int):
        record = {
            column: make_json_compatible(source.iloc[index][column])
            for column in selected_columns
        }
        local_client = OpenAICompatibleClient(api_config())
        return index, annotate_record(
            local_client,
            st.session_state.system_prompt,
            st.session_state.user_template,
            record,
            fields,
            threshold,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(task, index) for index in indices]
        for future in as_completed(futures):
            index, result = future.result()
            st.session_state.results[index] = result.to_flat_dict()
            completed += 1
            progress.progress(completed / len(indices), text=f"已完成 {completed}/{len(indices)}")
            status.caption(
                f"当前记录：第 {index + 1} 行 · "
                f"{'需复核' if result.needs_review else '已通过'}"
            )
    progress.empty()
    status.empty()
    st.success(f"本批次完成：{len(indices)} 条。")


def show_label_page() -> None:
    st.header("批量标注")
    if not st.session_state.data_ready or st.session_state.data_df is None:
        st.error("请先在“数据导入”中上传文件并确认输入列。")
        return
    if not st.session_state.api_key.strip():
        st.error("请先配置 API Key。")
        return
    fields = parsed_fields_or_error()
    if not fields:
        return

    source: pd.DataFrame = st.session_state.data_df
    processed = set(st.session_state.results)
    pending = [index for index in range(len(source)) if index not in processed]
    failed = [
        index
        for index, item in st.session_state.results.items()
        if item.get("AI_错误")
    ]
    review_count = sum(
        1 for item in st.session_state.results.values() if item.get("AI_需复核")
    )
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("总行数", f"{len(source):,}")
    metric2.metric("已处理", f"{len(processed):,}")
    metric3.metric("待复核", f"{review_count:,}")
    metric4.metric("失败", f"{len(failed):,}")

    settings1, settings2, settings3 = st.columns(3)
    with settings1:
        threshold = st.slider("复核置信度阈值", 0.0, 1.0, 0.75, 0.05)
    with settings2:
        max_batch = max(1, min(500, max(len(pending), len(failed), 1)))
        batch_size = st.number_input("本批处理条数", 1, max_batch, min(20, max_batch))
    with settings3:
        concurrency = st.slider("并发请求数", 1, 8, 3)

    estimated = min(int(batch_size), len(pending))
    st.caption(f"下一批将调用 API {estimated} 次。高并发可能触发服务商限流。")
    action1, action2 = st.columns(2)
    with action1:
        start = st.button(
            "开始下一批",
            type="primary",
            use_container_width=True,
            disabled=not pending,
        )
    with action2:
        retry = st.button(
            "重试失败记录",
            use_container_width=True,
            disabled=not failed,
        )

    if start:
        run_indices(pending[: int(batch_size)], threshold, concurrency)
    if retry:
        run_indices(failed[: int(batch_size)], threshold, concurrency)

    current = result_dataframe()
    if not current.empty:
        display_columns = ["_source_index", "AI_处理状态"]
        display_columns += [f"AI_{field.name}" for field in fields]
        display_columns += ["AI_置信度", "AI_需复核", "AI_复核原因", "AI_错误"]
        available = [column for column in display_columns if column in current.columns]
        st.dataframe(current[available], use_container_width=True, hide_index=True)

    with st.expander("重置本次标注结果"):
        confirm = st.checkbox("确认清空已生成的标注")
        if st.button("清空标注", disabled=not confirm):
            st.session_state.results = {}
            st.session_state.review_revision += 1
            st.rerun()


def show_review_page() -> None:
    st.header("复核与导出")
    full = result_dataframe()
    if full.empty or not st.session_state.results:
        st.error("当前没有可复核或导出的标注结果。")
        return
    fields = parsed_fields_or_error()
    if not fields:
        return
    pending_review = unresolved_review_dataframe(full)
    processed = full[full["AI_处理状态"] != "未处理"].copy()

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("已生成结果", f"{len(processed):,}")
    metric2.metric("仍待复核", f"{len(pending_review):,}")
    metric3.metric("可直接通过", f"{len(processed) - len(pending_review):,}")

    if not pending_review.empty:
        st.subheader("待复核记录")
        review_columns = ["_source_index"]
        review_columns += [column for column in st.session_state.input_columns if column in pending_review.columns]
        review_columns += [f"AI_{field.name}" for field in fields]
        review_columns += ["AI_置信度", "AI_复核原因", "AI_错误"]
        editor_df = pending_review[[column for column in review_columns if column in pending_review.columns]].copy()
        for field in fields:
            column = f"AI_{field.name}"
            if column in editor_df.columns and field.type in {"array", "object"}:
                editor_df[column] = editor_df[column].map(
                    lambda value: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                )
        editor_df["人工复核状态"] = "待复核"
        editor_df["人工备注"] = ""
        disabled_columns = [
            "_source_index",
            *[column for column in st.session_state.input_columns if column in editor_df.columns],
            "AI_错误",
        ]
        edited = st.data_editor(
            editor_df,
            key=f"review_editor_{st.session_state.review_revision}",
            use_container_width=True,
            hide_index=True,
            disabled=disabled_columns,
            column_config={
                "人工复核状态": st.column_config.SelectboxColumn(
                    "人工复核状态",
                    options=["待复核", "已确认"],
                    required=True,
                ),
                "AI_置信度": st.column_config.NumberColumn(min_value=0.0, max_value=1.0),
            },
        )
        if st.button("保存人工复核", type="primary"):
            saved = 0
            save_errors: list[str] = []
            for _, row in edited.iterrows():
                if row["人工复核状态"] != "已确认":
                    continue
                index = int(row["_source_index"])
                item = dict(st.session_state.results[index])
                parsed_values: dict[str, Any] = {}
                for field in fields:
                    column = f"AI_{field.name}"
                    if column in row:
                        value = row[column]
                        if field.type in {"array", "object"} and isinstance(value, str):
                            try:
                                value = json.loads(value)
                            except json.JSONDecodeError:
                                save_errors.append(f"第 {index + 1} 行的 {field.name} 不是合法 JSON")
                                break
                            expected = list if field.type == "array" else dict
                            if not isinstance(value, expected):
                                save_errors.append(f"第 {index + 1} 行的 {field.name} 类型不正确")
                                break
                        parsed_values[column] = make_json_compatible(value)
                else:
                    item.update(parsed_values)
                    item["AI_置信度"] = float(row.get("AI_置信度", item.get("AI_置信度", 0)))
                    item["AI_需复核"] = False
                    note_value = row.get("人工备注", "")
                    note = "" if pd.isna(note_value) else str(note_value).strip()
                    item["AI_复核原因"] = f"人工已确认{f'：{note}' if note else ''}"
                    item["AI_人工复核"] = "已确认"
                    st.session_state.results[index] = item
                    saved += 1
            if save_errors:
                st.error("；".join(save_errors))
            else:
                st.session_state.review_revision += 1
                st.success(f"已保存 {saved} 条人工复核结果。")
                st.rerun()
    else:
        st.success("所有已标注记录均已通过或完成人工复核。")

    full = result_dataframe()
    processed = full[full["AI_处理状态"] != "未处理"].copy()
    pending_review = unresolved_review_dataframe(processed)
    st.subheader("整合结果")
    st.dataframe(processed, use_container_width=True, hide_index=True)

    export_config = {
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "源文件": st.session_state.data_file_name,
        "模型": st.session_state.model,
        "API Base URL": st.session_state.base_url,
        "输入列": st.session_state.input_columns,
        "System Prompt": st.session_state.system_prompt,
        "用户 Prompt 模板": st.session_state.user_template,
        "字段配置": json.loads(st.session_state.fields_json),
        "已处理条数": len(processed),
        "待复核条数": len(pending_review),
    }
    excel_bytes = dataframe_to_excel(full, pending_review, export_config)
    base_name = st.session_state.data_file_name.rsplit(".", 1)[0] or "标注结果"
    st.download_button(
        "下载整合结果 Excel",
        data=excel_bytes,
        file_name=f"{base_name}_AI标注结果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True,
    )


initialize_state()

st.markdown(
    """
    <style>
    .stApp { background: #f7f8fa; }
    [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu { display: none !important; }
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e4e7ec; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #e4e7ec; padding: 14px; border-radius: 6px; }
    .stButton > button, .stDownloadButton > button { border-radius: 6px; }
    h1, h2, h3 { letter-spacing: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Excel AI 标注")
    page = st.radio("工作流", NAV_ITEMS, label_visibility="collapsed")
    st.divider()
    st.caption("当前状态")
    st.write(f"API：{'已连接' if st.session_state.api_verified else '未验证'}")
    st.write(f"数据：{len(st.session_state.data_df) if st.session_state.data_df is not None else 0} 行")
    st.write(f"标注：{len(st.session_state.results)} 条")
    st.divider()
    st.caption("本机配置")
    st.toggle("安全记住 API Key", key="remember_api_key")
    if st.button("保存当前配置", use_container_width=True):
        save_local_defaults()
    if st.button("清除本机配置", use_container_width=True):
        clear_local_defaults()
    st.caption(st.session_state.local_config_status)

if page == "API 接入":
    show_api_page()
elif page == "Prompt 调试":
    show_prompt_page()
elif page == "数据导入":
    show_data_page()
elif page == "批量标注":
    show_label_page()
else:
    show_review_page()
