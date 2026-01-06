import os
import uvicorn
import requests
from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from google import genai

# ================= 配置区域 =================
# 1. Google Gemini 配置
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 2. 生图模型配置 (Nano Banana Pro)
# 请在 Render 环境变量中填入该模型的完整 API URL
IMAGE_API_URL = os.getenv("IMAGE_API_URL") 
IMAGE_API_KEY = os.getenv("IMAGE_API_KEY")

# 初始化 Google 客户端
client = None
if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)

# ================= 核心逻辑 =================

def generate_prompt_logic(title: str, content: str) -> str:
    """使用 Gemini 3 Pro 优化提示词"""
    if not client:
        return f"high quality cover for {title}"

    # 提示词工程：确保输出纯英文、高质量的 Prompt
    system_prompt = """
    You are an expert AI Art Director. 
    Your task is to convert the user's Xiaohongshu (RedNote) title and content into a specific English prompt for the "Nano Banana Pro" image model.
    
    Guidelines:
    1. Output MUST be in English.
    2. Style keywords: Photorealistic, 8k, Soft lighting, High fashion, Minimalist composition.
    3. NO markdown, NO explanations, just the raw prompt text.
    """
    
    user_input = f"Title: {title}\nContent: {content}"

    try:
        response = client.models.generate_content(
            model="gemini-3-pro", # 这里指定了 Gemini 3 Pro，如果报错请改为 gemini-1.5-pro
            contents=[system_prompt, user_input]
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Error: {e}")
        # 降级处理，防止流程中断
        return f"A creative cover image about {title}, high aesthetic, 8k resolution"

def call_image_api(prompt: str) -> str:
    """调用真实的生图 API"""
    if not IMAGE_API_URL or not IMAGE_API_KEY:
        return "Error: Missing IMAGE_API_URL or IMAGE_API_KEY in environment variables."

    print(f"Calling Image API with prompt: {prompt[:50]}...")

    # 这里的 Payload 结构是目前最通用的 (兼容 Replicate/SiliconFlow 等)
    # 如果你的服务商要求不同的参数（比如 'text' 而不是 'prompt'），请在这里修改
    payload = {
        "prompt": prompt,
        "model": "nano-banana-pro", # 部分 API 需要指定模型名
        "image_size": "1024x1024",  # 或者 width: 1024, height: 1024
        "num_inference_steps": 30,
        "guidance_scale": 7.5
    }

    headers = {
        "Authorization": f"Bearer {IMAGE_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        response = requests.post(IMAGE_API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status() # 如果状态码不是 200，抛出异常
        
        data = response.json()
        
        # === 关键：解析不同服务商的返回格式 ===
        # 情况 A: 直接返回 {"image_url": "https://..."}
        if "image_url" in data:
            return data["image_url"]
        
        # 情况 B: Replicate 风格 {"output": ["https://..."]}
        if "output" in data and isinstance(data["output"], list):
            return data["output"][0]
            
        # 情况 C: SiliconFlow/OpenAI 风格 {"data": [{"url": "..."}]}
        if "data" in data and isinstance(data["data"], list):
            return data["data"][0].get("url")

        return f"Error: Unknown API response format: {str(data)[:100]}"

    except Exception as e:
        print(f"Image API Error: {e}")
        return f"Error generating image: {str(e)}"

# ================= MCP Server 定义 =================
app = FastAPI()
mcp = Server("xhs-cover-mcp-v2")

@mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_xhs_cover",
            description="Generate generic Xiaohongshu cover image using Gemini 3 Pro and Nano Banana Pro",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "The post title"},
                    "content": {"type": "string", "description": "The post content details"},
                },
                "required": ["title", "content"],
            },
        )
    ]

@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "generate_xhs_cover":
        title = arguments.get("title", "")
        content = arguments.get("content", "")
        
        # 1. 生成提示词
        prompt = generate_prompt_logic(title, content)
        
        # 2. 生成图片
        image_url = call_image_api(prompt)
        
        return [TextContent(type="text", text=image_url)]
    
    raise ValueError(f"Unknown tool: {name}")

# SSE 路由支持
@app.get("/sse")
async def handle_sse(request: Request):
    async def event_generator():
        transport = SseServerTransport("/messages")
        async with mcp.run(transport.read_incoming, transport.write_outgoing):
            async for message in transport.outgoing_messages:
                yield message
    return EventSourceResponse(event_generator())

@app.post("/messages")
async def handle_messages(request: Request):
    return await mcp.process_request(request)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
