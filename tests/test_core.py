import json
from io import BytesIO

import pandas as pd
import pytest

from core import (
    FieldSpec,
    annotate_record,
    build_annotation_messages,
    build_chat_completions_url,
    dataframe_to_excel,
    parse_field_specs,
    parse_json_object,
    validate_annotation,
)


def test_build_chat_completions_url_supports_common_base_formats():
    assert (
        build_chat_completions_url("https://api.example.com/v1")
        == "https://api.example.com/v1/chat/completions"
    )
    assert (
        build_chat_completions_url("https://api.example.com")
        == "https://api.example.com/v1/chat/completions"
    )
    assert (
        build_chat_completions_url("https://api.example.com/v1/chat/completions")
        == "https://api.example.com/v1/chat/completions"
    )


def test_parse_field_specs_validates_duplicates_and_types():
    fields = parse_field_specs(
        '[{"name":"label","description":"分类","type":"string",'
        '"allowed_values":["A","B"]}]'
    )
    assert fields[0].name == "label"
    assert fields[0].allowed_values == ("A", "B")

    with pytest.raises(ValueError, match="重复"):
        parse_field_specs(
            '[{"name":"label","description":"x"},{"name":"label","description":"y"}]'
        )
    with pytest.raises(ValueError, match="不支持"):
        parse_field_specs('[{"name":"label","description":"x","type":"date"}]')


def test_parse_json_object_accepts_fence_and_surrounding_text():
    assert parse_json_object('```json\n{"label":"A"}\n```') == {"label": "A"}
    assert parse_json_object('结果如下：\n{"label":"B"}\n结束') == {"label": "B"}


def test_annotation_messages_include_record_and_contract():
    fields = [FieldSpec("label", "主标签", allowed_values=("A", "B"))]
    messages = build_annotation_messages(
        "系统规则", "输入：{input_json}", {"text": "示例"}, fields
    )
    assert "示例" in messages[1]["content"]
    assert "_confidence" in messages[0]["content"]
    assert "allowed_values" in messages[0]["content"]


def test_validation_routes_uncertain_and_invalid_values_to_review():
    fields = [FieldSpec("label", "主标签", allowed_values=("A", "B"))]
    annotations, confidence, review, reason = validate_annotation(
        {
            "annotations": {"label": "C"},
            "_confidence": 0.61,
            "_needs_review": False,
            "_review_reason": "",
        },
        fields,
        confidence_threshold=0.75,
    )
    assert annotations == {"label": "C"}
    assert confidence == 0.61
    assert review is True
    assert "允许值" in reason
    assert "低于阈值" in reason


class FakeClient:
    def chat(self, messages, **kwargs):
        assert messages
        return (
            json.dumps(
                {
                    "annotations": {"label": "A"},
                    "_confidence": 0.95,
                    "_needs_review": False,
                    "_review_reason": "",
                },
                ensure_ascii=False,
            ),
            {"total_tokens": 20},
        )


def test_annotate_record_returns_flat_export_columns():
    result = annotate_record(
        FakeClient(),
        "系统规则",
        "{input_json}",
        {"text": "测试"},
        [FieldSpec("label", "主标签", allowed_values=("A", "B"))],
        0.75,
    )
    flat = result.to_flat_dict()
    assert flat["AI_label"] == "A"
    assert flat["AI_需复核"] is False
    assert flat["AI_错误"] == ""


def test_excel_export_contains_result_review_and_config_sheets():
    result = pd.DataFrame([{"原文": "测试", "AI_label": "A", "AI_需复核": False}])
    review = result.iloc[0:0].copy()
    payload = dataframe_to_excel(result, review, {"模型": "test-model"})
    workbook = pd.ExcelFile(BytesIO(payload))
    assert workbook.sheet_names == ["标注结果", "待复核", "运行配置"]
    config = pd.read_excel(BytesIO(payload), sheet_name="运行配置")
    assert "test-model" in config["内容"].astype(str).tolist()
