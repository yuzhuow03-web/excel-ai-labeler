# Excel AI 半自动标注工具

Windows 本地半自动数据标注应用：Prompt 调试、批量初标、人工复核和 Excel 导出。

## 从源码运行

## 环境要求
- Python 3.12
- Windows 推荐使用桌面窗口模式

## 安装依赖
在仓库根目录创建虚拟环境并安装依赖：

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

## 启动方式
启动桌面窗口：

```bash
.venv/Scripts/python launcher.py
```

备选方式（浏览器模式）：

```bash
.venv/Scripts/python -m streamlit run app.py
```

## API Key 保存
Windows 下，若用户选择记住 API Key，`local_config.py` 通过 DPAPI 加密保存，不写入明文配置。

## 测试

```bash
.venv/Scripts/python -m pytest -q
```

## 仓库内容说明
仓库不包含已打包的 EXE、`.venv` 或用户配置。`sample_data.xlsx` 为 5 条合成英文示例数据，可用于功能演示。主要源码为 `app.py`、`core.py`、`launcher.py`、`local_config.py`。

## 工作流

1. API 接入：选择预设或填写任意 OpenAI 兼容的 Base URL、API Key 和模型名称，然后测试连接。
2. Prompt 调试：编辑 System Prompt、用户模板和字段配置，用样例反复测试；填写反馈后可让当前模型生成优化版本。
3. 数据导入：上传 Excel/CSV，选择工作表、表头行、记录 ID 和需要发送给 AI 的列。
4. 批量标注：设置复核阈值、批次大小和并发数，分批调用 API。
5. 复核与导出：集中修改低置信度、格式异常、接口失败或模型主动标记的记录，确认后导出完整结果。

## 字段配置

字段配置是 JSON 数组，可以添加任意数量的输出字段：

```json
[
  {
    "name": "sentiment",
    "description": "文本情感倾向",
    "type": "string",
    "allowed_values": ["正向", "中性", "负向"],
    "required": true
  },
  {
    "name": "entities",
    "description": "识别到的实体名称列表",
    "type": "array",
    "required": false
  }
]
```

支持的类型：`string`、`number`、`integer`、`boolean`、`array`、`object`。

程序自动加入 `_confidence`、`_needs_review`、`_review_reason` 三个控制字段。以下情况会进入“待复核”：

- 置信度低于设定阈值；
- 缺少必填字段或类型错误；
- 输出不在 `allowed_values` 中；
- API 请求失败或响应不是合法 JSON；
- 模型主动判断存在歧义。

## 导出内容

导出的 `.xlsx` 包含三个工作表：

- `标注结果`：所有原始行及其处理状态、AI 标注、置信度、复核状态和原始响应；
- `待复核`：尚未人工确认的存疑记录；
- `运行配置`：模型、Prompt、字段配置和处理统计，不包含 API Key。

## 数据与费用

- 默认情况下 API Key 只保存在当前会话内。启用“安全记住 API Key”并保存配置后，密钥由 Windows DPAPI 加密，且只能由当前 Windows 用户解密。
- 本机配置保存在 `%APPDATA%\ExcelAILabeler\settings.json`；导出文件始终不包含 API Key。
- 被选中的输入列会发送给配置的在线模型服务；不要选择与标注无关的敏感列。
- 每条数据通常对应一次 API 调用。先用少量样例稳定 Prompt，再扩大批次。
- “JSON Mode”需要服务商兼容 `response_format={"type":"json_object"}`；不兼容时关闭该开关，程序仍会尝试解析普通文本中的 JSON。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 本次源码验证

已在 Windows 中执行现有单元测试，10 项全部通过，包括合成数据导出和本地密钥加密往返测试；同时通过 Python 源码语法检查。本次没有调用在线模型或启动桌面 UI 联调。
