from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from io import BytesIO
from typing import Any, Iterable

import pandas as pd
import requests


SUPPORTED_TYPES = {"string", "number", "integer", "boolean", "array", "object"}
RESERVED_FIELDS = {"_confidence", "_needs_review", "_review_reason"}


@dataclass(frozen=True)
class ApiConfig:
    api_key: str
    base_url: str
    model: str
    timeout: int = 60
    temperature: float = 0.0
    json_mode: bool = True
    max_retries: int = 2


@dataclass(frozen=True)
class FieldSpec:
    name: str
    description: str
    type: str = "string"
    allowed_values: tuple[Any, ...] = ()
    required: bool = True

    def to_prompt_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "required": self.required,
        }
        if self.allowed_values:
            result["allowed_values"] = list(self.allowed_values)
        return result


@dataclass
class AnnotationResult:
    annotations: dict[str, Any]
    confidence: float
    needs_review: bool
    review_reason: str
    raw_response: str
    error: str = ""
    elapsed_seconds: float = 0.0

    def to_flat_dict(self, field_prefix: str = "AI_") -> dict[str, Any]:
        flat = {f"{field_prefix}{key}": value for key, value in self.annotations.items()}
        flat.update(
            {
                "AI_置信度": self.confidence,
                "AI_需复核": self.needs_review,
                "AI_复核原因": self.review_reason,
                "AI_错误": self.error,
                "AI_耗时秒": round(self.elapsed_seconds, 3),
                "AI_原始响应": self.raw_response,
            }
        )
        return flat


class ApiRequestError(RuntimeError):
    pass


def build_chat_completions_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    if not value:
        raise ValueError("API Base URL 不能为空")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


class OpenAICompatibleClient:
    def __init__(self, config: ApiConfig):
        self.config = config
        self.url = build_chat_completions_url(config.base_url)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        json_mode: bool | None = None,
        max_tokens: int = 2000,
    ) -> tuple[str, dict[str, Any]]:
        if not self.config.api_key.strip():
            raise ApiRequestError("API Key 不能为空")
        if not self.config.model.strip():
            raise ApiRequestError("模型名称不能为空")

        payload: dict[str, Any] = {
            "model": self.config.model.strip(),
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        use_json_mode = self.config.json_mode if json_mode is None else json_mode
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.config.api_key.strip()}",
            "Content-Type": "application/json",
        }

        last_error = ""
        for attempt in range(self.config.max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout,
                )
                if response.status_code >= 400:
                    detail = _extract_error_detail(response)
                    retryable = response.status_code in {408, 409, 429} or response.status_code >= 500
                    if retryable and attempt < self.config.max_retries:
                        time.sleep(min(2**attempt, 4))
                        continue
                    raise ApiRequestError(f"API 返回 HTTP {response.status_code}: {detail}")

                body = response.json()
                choices = body.get("choices") or []
                if not choices:
                    raise ApiRequestError("API 响应中没有 choices")
                message = choices[0].get("message") or {}
                content = message.get("content")
                if isinstance(content, list):
                    content = "".join(
                        str(item.get("text", "")) if isinstance(item, dict) else str(item)
                        for item in content
                    )
                if not content:
                    content = message.get("reasoning_content")
                if not content:
                    raise ApiRequestError("API 响应内容为空")
                return str(content), body.get("usage") or {}
            except ApiRequestError:
                raise
            except requests.RequestException as exc:
                last_error = _safe_error_text(str(exc), self.config.api_key)
                if attempt < self.config.max_retries:
                    time.sleep(min(2**attempt, 4))
                    continue
            except (ValueError, TypeError, KeyError) as exc:
                raise ApiRequestError(f"无法解析 API 响应: {exc}") from exc

        raise ApiRequestError(f"API 请求失败: {last_error or '未知网络错误'}")


def _extract_error_detail(response: requests.Response) -> str:
    try:
        body = response.json()
        error = body.get("error", body)
        if isinstance(error, dict):
            return str(error.get("message") or error.get("detail") or error)[:500]
        return str(error)[:500]
    except ValueError:
        return response.text[:500] or "无错误详情"


def _safe_error_text(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


def parse_field_specs(value: str | list[dict[str, Any]]) -> list[FieldSpec]:
    try:
        raw = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ValueError(f"字段配置不是合法 JSON: 第 {exc.lineno} 行第 {exc.colno} 列") from exc

    if not isinstance(raw, list) or not raw:
        raise ValueError("字段配置必须是非空 JSON 数组")

    fields: list[FieldSpec] = []
    names: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 个字段必须是 JSON 对象")
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        field_type = str(item.get("type", "string")).strip().lower()
        if not name:
            raise ValueError(f"第 {index} 个字段缺少 name")
        if name in names:
            raise ValueError(f"字段名重复: {name}")
        if name in RESERVED_FIELDS or name.startswith("AI_"):
            raise ValueError(f"字段名 {name} 为程序保留名称")
        if field_type not in SUPPORTED_TYPES:
            raise ValueError(f"字段 {name} 的 type 不支持: {field_type}")
        allowed = item.get("allowed_values", [])
        if allowed is None:
            allowed = []
        if not isinstance(allowed, list):
            raise ValueError(f"字段 {name} 的 allowed_values 必须是数组")
        fields.append(
            FieldSpec(
                name=name,
                description=description or name,
                type=field_type,
                allowed_values=tuple(allowed),
                required=bool(item.get("required", True)),
            )
        )
        names.add(name)
    return fields


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("响应中没有找到 JSON 对象")
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"响应不是合法 JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("响应 JSON 顶层必须是对象")
    return parsed


def make_json_compatible(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, dict):
        return {str(key): make_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_compatible(item) for item in value]
    return value


def build_annotation_messages(
    system_prompt: str,
    user_template: str,
    record: dict[str, Any],
    fields: Iterable[FieldSpec],
) -> list[dict[str, str]]:
    field_list = [field.to_prompt_dict() for field in fields]
    contract = {
        "annotations": {field["name"]: f"<{field['type']}>" for field in field_list},
        "_confidence": "<0 到 1 的数字>",
        "_needs_review": "<布尔值>",
        "_review_reason": "<存疑原因；确定时为空字符串>",
    }
    runtime_rules = (
        "\n\n【程序强制输出规则】\n"
        "1. 只输出一个 JSON 对象，不要 Markdown、解释或代码块。\n"
        "2. annotations 必须严格按字段配置填写，不确定时不要编造。\n"
        "3. _confidence 是对整条标注的置信度，范围 0 到 1。\n"
        "4. 任一字段有歧义、信息不足或需要人工判断时，将 _needs_review 设为 true，并说明原因。\n"
        f"字段配置：{json.dumps(field_list, ensure_ascii=False)}\n"
        f"输出结构：{json.dumps(contract, ensure_ascii=False)}"
    )
    input_json = json.dumps(make_json_compatible(record), ensure_ascii=False, indent=2)
    if "{input_json}" in user_template:
        user_content = user_template.replace("{input_json}", input_json)
    else:
        user_content = f"{user_template.rstrip()}\n\n待标注数据：\n{input_json}"
    return [
        {"role": "system", "content": system_prompt.strip() + runtime_rules},
        {"role": "user", "content": user_content},
    ]


def validate_annotation(
    parsed: dict[str, Any],
    fields: Iterable[FieldSpec],
    confidence_threshold: float,
) -> tuple[dict[str, Any], float, bool, str]:
    field_list = list(fields)
    annotations = parsed.get("annotations")
    if not isinstance(annotations, dict):
        annotations = {field.name: parsed.get(field.name) for field in field_list if field.name in parsed}

    normalized: dict[str, Any] = {}
    issues: list[str] = []
    for field in field_list:
        value = annotations.get(field.name)
        if value is None:
            if field.required:
                issues.append(f"缺少字段 {field.name}")
            normalized[field.name] = None
            continue
        if not _matches_type(value, field.type):
            issues.append(f"字段 {field.name} 类型应为 {field.type}")
        if field.allowed_values and value not in field.allowed_values:
            issues.append(f"字段 {field.name} 不在允许值中")
        normalized[field.name] = value

    try:
        confidence = float(parsed.get("_confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
        issues.append("置信度格式无效")
    confidence = max(0.0, min(1.0, confidence))
    if confidence < confidence_threshold:
        issues.append(f"置信度 {confidence:.2f} 低于阈值 {confidence_threshold:.2f}")

    ai_review = parsed.get("_needs_review", False)
    if isinstance(ai_review, str):
        ai_review = ai_review.strip().lower() in {"true", "1", "yes", "是", "需要"}
    reason = str(parsed.get("_review_reason", "") or "").strip()
    if bool(ai_review) and not reason:
        reason = "模型主动标记为需要复核"
    if reason:
        issues.insert(0, reason)
    return normalized, confidence, bool(ai_review or issues), "；".join(dict.fromkeys(issues))


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return False


def annotate_record(
    client: OpenAICompatibleClient,
    system_prompt: str,
    user_template: str,
    record: dict[str, Any],
    fields: Iterable[FieldSpec],
    confidence_threshold: float,
) -> AnnotationResult:
    started = time.perf_counter()
    field_list = list(fields)
    raw = ""
    try:
        messages = build_annotation_messages(system_prompt, user_template, record, field_list)
        raw, _usage = client.chat(messages, max_tokens=2000)
        parsed = parse_json_object(raw)
        annotations, confidence, needs_review, reason = validate_annotation(
            parsed, field_list, confidence_threshold
        )
        return AnnotationResult(
            annotations=annotations,
            confidence=confidence,
            needs_review=needs_review,
            review_reason=reason,
            raw_response=raw,
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return AnnotationResult(
            annotations={field.name: None for field in field_list},
            confidence=0.0,
            needs_review=True,
            review_reason="标注失败，需要人工处理",
            raw_response=raw,
            error=str(exc),
            elapsed_seconds=time.perf_counter() - started,
        )


def build_prompt_optimizer_messages(
    system_prompt: str,
    user_template: str,
    fields_json: str,
    sample_input: str,
    expected_output: str,
    last_output: str,
    feedback: str,
) -> list[dict[str, str]]:
    request = {
        "current_system_prompt": system_prompt,
        "current_user_template": user_template,
        "field_schema": fields_json,
        "test_input": sample_input,
        "expected_output_or_criteria": expected_output,
        "last_test_output": last_output,
        "user_feedback": feedback,
    }
    return [
        {
            "role": "system",
            "content": (
                "你是数据标注 Prompt 工程师。改进 Prompt，使标签边界明确、可复现、对歧义有处理规则，"
                "并保留用户原本的业务意图。user_template 必须保留 {input_json} 占位符。"
                "只返回 JSON：optimized_system_prompt、optimized_user_template、change_summary。"
            ),
        },
        {"role": "user", "content": json.dumps(request, ensure_ascii=False, indent=2)},
    ]


def dataframe_to_excel(
    result_df: pd.DataFrame,
    review_df: pd.DataFrame,
    config: dict[str, Any],
) -> bytes:
    output = BytesIO()
    safe_result = _excel_safe_dataframe(result_df)
    safe_review = _excel_safe_dataframe(review_df)
    config_df = pd.DataFrame(
        [{"配置项": str(key), "内容": _config_cell(value)} for key, value in config.items()]
    )
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        safe_result.to_excel(writer, sheet_name="标注结果", index=False)
        safe_review.to_excel(writer, sheet_name="待复核", index=False)
        config_df.to_excel(writer, sheet_name="运行配置", index=False)
        for sheet_name, worksheet in writer.sheets.items():
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                values = [str(cell.value or "") for cell in list(column_cells)[:200]]
                width = min(max(max((len(value) for value in values), default=8) + 2, 10), 45)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
    return output.getvalue()


def _excel_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    for column in safe.columns:
        safe[column] = safe[column].map(_excel_cell)
    return safe


def _excel_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(make_json_compatible(value), ensure_ascii=False)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return value


def _config_cell(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(make_json_compatible(value), ensure_ascii=False, indent=2)


def field_specs_as_json(fields: Iterable[FieldSpec]) -> str:
    return json.dumps([asdict(field) for field in fields], ensure_ascii=False, indent=2)
