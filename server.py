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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
IMAGE_API_URL = os.getenv("IMAGE_API_URL") 
IMAGE_API_KEY = os.getenv("IMAGE_API_KEY")

# 初始化 Google 客户端
client = None
if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)

# ================= 核心逻辑 (已根据你的模版深度优化) =================

def generate_prompt_logic(title: str, content: str) -> str:
    """
    使用 Gemini 3 Pro，根据用户提供的【小红书高爆款模版】生成生图提示词
    """
    if not client:
        return f"xiaohongshu style cover, title '{title}', high quality"

    # --- 这里就是我们将你的模版注入给 Gemini 的地方 ---
    system_prompt = """
    You are an expert AI Visual Designer specializing in "Xiaohongshu" (RedNote) cover designs.
    Your task is to convert the user's Title and Content into a precise English text-to-image prompt for the "Nano Banana Pro" model.

    ### Design Template Requirements (Strictly Follow):
    1. **Style**: Xiaohongshu Pop Aesthetic, High Click-Through Rate (CTR).
    2. **Background**: Choose ONE representing the content best: 
       - A textured notebook page / grid paper.
       - A clean minimalist desk setup.
       - A stylized floating chat window interface.
    3. **Typography & Layout**:
       - The MAIN TITLE must be the visual focus (occupy 40% of space, bold, bubble or 3D font).
       - Visual Hierarchy: Title size > 2x Body text size.
       - Allow for white space (negative space) between text blocks.
    4. **Color**: Use eye-catching highlight colors (like bright yellow, neon pink, or electric blue) ONLY for key words.
    5. **Icons**: Add relevant 3D icons, stickers, or emojis (e.g., muscles, sparks, palettes) to add visual layers without clutter.
    6. **Aspect Ratio**: Vertical 9:16 composition.

    ### Output Format:
    - Output ONLY the final English prompt string.
    - Include the exact text to render inside single quotes, e.g., text 'YOUR TITLE'. 
    - Add style boosters: "best quality, 8k, c4d render, blender style, vector illustration, poster design".

    ### User Input:
    """
    
    user_input = f"Title: {title}\nContent: {content}"

    try:
        # 使用 gemini-2.0-flash 或 gemini-1.5-pro (等待 3 Pro)
        # 这里的指令非常复杂，建议使用能力更强的模型
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=[system_prompt, user_input]
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Error: {e}")
        return f"poster design regarding {title}, minimalist, 8k"

def call_image_api(prompt: str) -> str:
    """调用真实的生图 API"""
    if not IMAGE_API_URL:
        return "Error: Missing IMAGE_API_URL"

    print(f"Calling Image API with prompt: {prompt[:50]}...")

    # 针对 Nano Banana Pro / Flux 类型的通用 Payload
    # 注意：如果你的模型支持 width/height 参数，请确保是 9:16 比例 (如 768x1344 或 1024x1792)
    payload = {
        "prompt": prompt,
        "model": "nano-banana-pro", 
        "image_size": "1024x1792", # 设置为 9:16 竖屏比例
        "num_inference_steps": 30,
        "guidance_scale": 7.0
    }

    headers = {
        "Authorization": f"Bearer {IMAGE_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        response = requests.post(IMAGE_API_URL, json=payload, headers=headers, timeout=60)
        
        # 调试用：如果报错，打印返回内容查看原因
        if response.status_code != 200:
            print(f"API Error Response: {response.text}")
            
        response.raise_for_status()
        data = response.json()
        
        # 兼容多种常见的 API 返回格式
        if "image_url" in data: return data["image_url"]
        if "output" in data and data["output"]: return data["output"][0]
        if "data" in data and data["data"]: return data["data"][0].get("url")
        if "images" in data and data["images"]: return data["images"][0].get("url") #有时候是 base64

        return f"Error: Unknown response format: {str(data)[:50]}"

    except Exception as e:
        return f"Error generating image: {str(e)}"

# ================= MCP Server 定义 =================
app = FastAPI()
mcp = Server("xhs-cover-mcp-v3")

@mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_xhs_cover",
            description="Generate a High-CTR Xiaohongshu cover image (9:16)",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "The main headline"},
                    "content": {"type": "string", "description": "Key bullet points"},
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
        
        prompt = generate_prompt_logic(title, content)
        image_url = call_image_api(prompt)
        
        return [TextContent(type="text", text=image_url)]
    
    raise ValueError(f"Unknown tool: {name}")

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
