# 多模态视频分析原型 (Multimodal Video Analysis Prototype)

这是一个全栈原型，展示了端到端的智能视频分析流程：
**本地视频上传 -> 点击目标对象 -> SAM2 实时分割 -> Whisper 语音转录 -> Qwen VL 百科生成 -> 前端可视化展示**

## 技术栈
- **后端**: FastAPI, Python 3.10+, PyTorch
  - 模型: `transformers.Sam2Model` (SAM2), `transformers.WhisperForConditionalGeneration` (Whisper), `OpenAI Client` (Qwen VL 适配)
  - 音频处理: `moviepy` (无需系统级 FFmpeg)
- **前端**: Vue 3, Vite, Element Plus

## 前置要求
- Python 3.10+ (需安装 `transformers>=4.45.0`, `tokenizers>=0.22.0,<=0.23.0`, `huggingface_hub>=0.23.0`)
- Node.js & npm
- NVIDIA GPU (推荐用于加速推理，非必须)
- **VLM API Key**: 推荐使用 [SiliconFlow](https://siliconflow.cn/) 或 [阿里云 Dashscope](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) 的 API Key。

## 快速启动 (Windows)

1. **环境准备**: 确保已安装 Python 和 Node.js。
2. **启动应用**: 在项目根目录下运行：
   ```bash
   python start.py
   ```
   *该脚本将自动执行以下操作：*
   - **依赖检测**: 检查并提示 `transformers` 和 `tokenizers` 版本是否匹配。
   - **下载源选择**: 提供 **HF-Mirror (国内镜像)** 和官方源的选择，解决模型下载缓慢问题。
   - **自动运行**: 同时启动后端 (8000端口) 和前端 (5173端口) 服务。

## 故障排查 (Troubleshooting)

### 1. 500 Internal Server Error (Analysis Failed)
- **查看后端控制台**: 后端已增加详细的 `[CRITICAL ERROR]` 日志和堆栈追踪。
- **维度不匹配**: SAM2 对 `transformers` 版本非常敏感，请确保运行 `pip install 'tokenizers>=0.22.0,<=0.23.0' --force-reinstall`。
- **显存不足**: 尝试切换到 `facebook/sam2-hiera-tiny` 模型。

### 2. Network Error / CORS 错误
- **前端直连**: 如果 Vite 代理失效，前端会自动尝试直连 `http://127.0.0.1:8000`。
- **CORS 配置**: 后端 `main.py` 已配置允许跨域请求。如果仍报错，请检查浏览器控制台 (F12) 的具体错误信息。

### 3. 模型下载失败
- 运行 `start.py` 时选择 `1. HF-Mirror`。
- 或者手动设置环境变量：`set HF_ENDPOINT=https://hf-mirror.com`。

## 手动安装与运行

### 1. 后端设置

```bash
cd backend
# 创建虚拟环境
python -m venv venv

# Windows 激活虚拟环境
.\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 运行服务器
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. 前端设置

```bash
cd frontend
# 安装依赖
npm install
# 启动开发服务器
npm run dev
```

### 3. 使用说明

1. 打开浏览器访问 。
2. **API 配置**: 在展开的“API & Model Configuration”中输入 Base URL (如 `https://api.siliconflow.cn/v1`) 和 API Key。
3. **上传**: 选择并上传一个本地视频文件 (MP4)。
4. **交互**: 播放视频并在感兴趣的画面暂停，点击视频中的任意目标物体。
5. **结果**:
   - **SAM2** 将实时分割出目标轮廓并叠加显示。
   - **Whisper** 将提取并转录点击点附近的语音内容。
   - **Qwen VL** 将结合视觉与语音信息生成百科卡片。

## 注意事项

- **API 兼容性**: Qwen VL 模块现在使用标准 OpenAI 客户端格式，支持所有兼容 OpenAI 接口的服务商（如 SiliconFlow, 阿里云, 自建 VLM 等）。
- **隐私安全**: API Key 仅保存在前端内存中，随请求发送，后端不进行任何持久化存储。
- **音频处理**: `moviepy` 会自动处理所需的二进制文件，无需手动安装 FFmpeg。