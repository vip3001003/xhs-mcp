import os
import uvicorn
import requests
from fastapi import FastAPI, Request, Body
from sse_starlette.sse import EventSourceResponse
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from google import genai

# ================= 配置区域 =================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
IMAGE_API_URL = os.getenv("IMAGE_API_URL") 
IMAGE_API_KEY = os.getenv("IMAGE_API_KEY")

client = None
if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)

# ================= 核心逻辑 (小红书模版) =================
def generate_prompt_logic(title: str, content: str) -> str:
    if not client: return f"cover regarding {title}"
    
    # 你的高爆款模版 Prompt
    system_prompt = """
    You are an expert AI Visual Designer specializing in "Xiaohongshu" (RedNote) cover designs.
    Your task is to convert the user's Title and Content into a precise English text-to-image prompt.

    ### Design Template Requirements:
    1. **Style**: Xiaohongshu Pop Aesthetic, High CTR.
    2. **Background**: Choose ONE: Textured notebook page OR Minimalist desk OR Floating chat window.
    3. **Typography**: MAIN TITLE must be visual focus (40% space). Hierarchy: Title > 2x Body text.
    4. **Color**: Use eye-catching highlight colors for key words.
    5. **Aspect Ratio**: Vertical 9:16.
    6. **Output**: ONLY the final English prompt. No markdown.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=[system_prompt, f"Title: {title}\nContent: {content}"]
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Error: {e}")
        return f"poster design regarding {title}, minimalist, 8k"

def call_image_api(prompt: str) -> str:
    if not IMAGE_API_URL: return "Error: Missing IMAGE_API_URL"
    
    payload = {
        "prompt": prompt,
        "model": "nano-banana-pro", 
        "image_size": "1024x1792", # 9:16
        "num_inference_steps": 30,
        "guidance_scale": 7.0
    }
    headers = {
        "Authorization": f"Bearer {IMAGE_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(IMAGE_API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        if "image_url" in data: return data["image_url"]
        if "output" in data and data["output"]: return data["output"][0]
        if "data" in data and data["data"]: return data["data"][0].get("url")
        return data # Fallback
    except Exception as e:
        return f"Error: {str(e)}"

# ================= 服务器定义 =================
app = FastAPI()
mcp = Server("xhs-cover-mcp")

# 1. 专为 n8n 准备的 HTTP 接口 (最稳)
@app.post("/n8n/run")
async def n8n_run(
    title: str = Body(..., embed=True),
    content: str = Body(..., embed=True)
):
    print(f"n8n Request: {title}")
    prompt = generate_prompt_logic(title, content)
    url = call_image_api(prompt)
    return {"result": url}

# 2. MCP 接口 (保留用于兼容)
@mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [Tool(name="generate_xhs_cover", description="Generate Cover", inputSchema={
        "type": "object", "properties": {"title": {"type": "string"}, "content": {"type": "string"}}, "required": ["title", "content"]
    })]

@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "generate_xhs_cover":
        res = await n8n_run(arguments["title"], arguments["content"])
        return [TextContent(type="text", text=res["result"])]
    raise ValueError(f"Unknown tool: {name}")

# 简单的 SSE 路由
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
