import os
import uvicorn
import requests
import google.generativeai as genai
from mcp.server.fastapi import FastAPIServer
from pydantic import BaseModel, Field

# ================= 1. 获取配置 (从云端环境变量) =================
# 如果你在本地运行报错，是因为没有设置环境变量，不用担心，部署到云端后并在Render里填写即可
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
IMAGE_API_URL = os.getenv("IMAGE_API_URL")
IMAGE_API_KEY = os.getenv("IMAGE_API_KEY")

# 初始化 Gemini (如果有 Key)
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro')

# ================= 2. 核心功能函数 =================
def generate_prompt_logic(title: str, content: str) -> str:
    """使用 Gemini 把中文文案变成英文生图提示词"""
    if not GOOGLE_API_KEY:
        return f"minimalist cover regarding {title}, high quality"
        
    system_instruction = "你是一个AI绘画提示词专家。请根据内容生成一段适配 Flux/SDXL 模型的英文提示词，强调高审美、摄影感、胶片质感。直接返回英文。"
    user_input = f"标题：{title}\n内容：{content}"
    
    try:
        response = model.generate_content([system_instruction, user_input])
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Error: {e}")
        return f"minimalist cover regarding {title}, high quality"

def call_image_api(prompt: str) -> str:
    """调用生图 API"""
    # 这里的代码逻辑用于连接真实的 Nano Banana Pro
    # 为了防止你刚部署没有 Key 导致报错，这里默认先返回一个模拟图片
    # 如果要开启真实生图，请部署后确保环境变量填写正确，并修改下方代码
    
    # --- 模拟模式 (免费，用于测试连接) ---
    # 这会根据提示词生成一个简单的 AI 图片 (来自 Pollinations AI)
    safe_prompt = prompt.replace(' ', '%20')
    return f"https://image.pollinations.ai/prompt/{safe_prompt}?nologo=true"

    # --- 真实模式 (填入 Key 后使用) ---
    # headers = {
    #     "Authorization": f"Bearer {IMAGE_API_KEY}",
    #     "Content-Type": "application/json"
    # }
    # payload = {
    #     "prompt": prompt,
    #     "image_size": "1024x1024", 
    #     "num_inference_steps": 25
    # }
    # try:
    #     resp = requests.post(IMAGE_API_URL, json=payload, headers=headers)
    #     resp.raise_for_status()
    #     # 根据你的 API 返回格式修改下面这行
    #     return resp.json().get('output')[0] 
    # except Exception as e:
    #     return f"Error: {str(e)}"

# ================= 3. 定义 MCP 服务器 =================
mcp = FastAPIServer("xhs-cover-mcp")

class XHSParams(BaseModel):
    title: str = Field(..., description="小红书标题")
    content: str = Field(..., description="小红书正文内容")

@mcp.tool(name="generate_xhs_cover", description="生成小红书封面图")
async def generate_xhs_cover(params: XHSParams) -> str:
    # 第一步：写提示词
    prompt = generate_prompt_logic(params.title, params.content)
    # 第二步：生图
    url = call_image_api(prompt)
    return url

# ================= 4. 启动设置 =================
if __name__ == "__main__":
    # 获取云平台分配的端口，默认为 8000
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on 0.0.0.0:{port}")
    uvicorn.run(mcp.app, host="0.0.0.0", port=port)