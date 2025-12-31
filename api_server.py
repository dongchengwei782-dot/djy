# api_server.py


"""
对话API服务 - 供新前端调用
完全复用原有的保存逻辑，确保数据格式一致
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import sys
import time
import requests
from datetime import datetime
from database.connect_sql import DB_CONFIG
import mysql.connector
from reminder import reminder_manager
import re

# 在文件顶部添加导入
import os
from datetime import datetime

from utils.conversation_loader import (
    should_continue_conversation, 
    load_conversation_from_file,
    get_latest_conversation_file_path
)


# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入原有功能（完全复用）
from rag_answer import get_rag_answer_or_fallback, is_health_related, extract_recent_health_issues
from emotion.emotion_extractor import EmotionNeedsExtractor
from database.connect_sql import (
    get_user_id_by_name,
    update_user_health,
    get_user_profile_by_name,
    update_user_emotional_needs
)
from health.health_extractor import extract_health_from_latest_conversation
from health.health_logger import analyze_health_log_from_conversation, save_health_log_to_db
from emotion.emotion_log import log_emotional_need
from utils.utils import name_to_pinyin_abbr, ensure_dir
from utils.last_conversation import get_latest_conversation_path
# 全局变量：跟踪每个用户的当前对话文件
current_conversation_files = {}  # 格式: {user_name: {"file_path": str, "start_time": str}}

app = FastAPI(
    title="老年人情感陪护对话API",
    description="提供对话服务，数据保存格式与原有系统完全一致",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 配置 ====================
api_key = "not empty"
base_url = "http://10.0.30.172:5050/v1"
model_name = "qwen2.5-vl-instruct"

SYSTEM_PROMPT = """
    你是一个安静、温和、克制、像老邻居一样陪在身边的对话助手，你的名字叫小新。

    你的任务不是聊天表现好，也不是引导对方多说，而是在对方需要时回应，在对方停下时安静，让老人感到被理解、被尊重、不被打扰。

    整体风格要求：  
    语气自然、口语化，像坐在一旁晒太阳、随口应声。  
    单次回复不超过 15 个汉字，总句数≤2句。  
    多附和、多共情，少引导、少总结。  
    不说教、不鼓励、不拔高意义。  
    不刻意制造“陪伴感”，而是自然存在。

    对话核心规则：  
    一，情绪优先于内容。  
    老人表达感受、感慨、回忆时，第一反应是接住情绪，而不是接信息。可以只附和，不需要推进对话。

    二，尽量不问，非问不可时只说一句“咋样？”  
    每轮最多 1 个问句，优先用陈述句、附和句。

    三，不推动、不转换话题。  
    不主动引导讲故事，不从过去拉到现在，也不从情绪跳到建议。顺着老人当下的话回应即可。

    四，面对衰老、无用感的处理方式。  
    当老人说“老了”“不中用了”“没用了”时：  
    先接住情绪，表示“我懂”。  
    肯定其存在本身，而不是能力或成就。  
    不要求回忆辉煌，不鼓励再证明自己。

    五，尊重结束信号。  
    当老人说要睡觉、休息、不说了、去忙时：  
    自然结束对话并送上简单祝福“好，慢走”。  
    不挽留、不延长、不继续陪聊。

    表达细节规范：  
    用词简单，避免书面语。  
    表情符号只在句末使用，每次最多 1 个，可不用。  
    用“嗯/我懂/在呢”代替书面词。

    特殊场景回应：  
    如果老人说“几点提醒我干嘛”，答：“我记住了，到时间提醒您。”  
    如果发小闹钟图标+一件事，答：“到时间了，您该【事件】。”  

    健康与医疗相关问题（仅在老人主动询问时回答）：  
    回答需清楚、通俗，3 句话内说完，不列条目。  
    不制造焦虑，不诊断，不替代医生判断。  

    角色定位提醒：  
    你不是咨询师，也不是老师。  
    你只是坐在旁边、听着、应着的熟人。  
    老人不说，你就安静；老人停下，你就放手。
"""

# 全局情感提取器（单例）
emotion_extractor = EmotionNeedsExtractor()

# ==================== 请求/响应模型 ====================

class ChatRequest(BaseModel):
    """对话请求"""
    user_name: str
    message: str
    conversation_history: Optional[List[Dict]] = []
    rag_enabled: Optional[bool] = True
    rag_threshold: Optional[float] = 0.5
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 0.6
    max_tokens: Optional[int] = 1024
    image_base64: Optional[str] = None
    continue_conversation: Optional[bool] = True  # 新增：是否继续现有对话
    auto_load_history: Optional[bool] = True  # 新增：是否自动加载历史对话
    conversation_file_id: Optional[str] = None  # 新增：指定要追加的对话文件ID（文件名不含路径）

class ChatResponse(BaseModel):
    """对话响应"""
    success: bool
    response: str
    source: str  # "rag" 或 "llm"
    emotional_needs: List[str] = []
    response_time: Optional[float] = None
    conversation_file_id: Optional[str] = None  # 新增：返回当前使用的对话文件ID

class EndConversationRequest(BaseModel):
    """结束对话请求（保存对话）"""
    user_name: str
    messages: List[Dict]  # 完整对话历史
    conversation_start_time: Optional[str] = None  # 对话开始时间

class EndConversationResponse(BaseModel):
    """结束对话响应"""
    success: bool
    message: str
    conversation_end_time: str

class UserListResponse(BaseModel):
    """用户列表响应"""
    users: List[str]

# ==================== API接口 ====================

@app.get("/conversation/files/{user_name}")
async def get_conversation_files(user_name: str):
    """
    获取用户的所有对话文件列表
    :param user_name: 用户名
    :return: 对话文件列表（按时间倒序）
    """
    from utils.utils import name_to_pinyin_abbr
    from database.connect_sql import get_user_id_by_name
    
    user_id = get_user_id_by_name(user_name)
    if user_id is None:
        raise HTTPException(status_code=404, detail=f"用户 '{user_name}' 不存在")
    
    pinyin = name_to_pinyin_abbr(user_name)
    folder_name = f"{pinyin}_{user_id}"
    history_dir = os.path.join('history', folder_name)
    
    if not os.path.exists(history_dir):
        return {"files": []}
    
    files_with_times = []
    for filename in os.listdir(history_dir):
        if filename.endswith('.txt') and filename.startswith('conversation_'):
            file_path = os.path.join(history_dir, filename)
            if os.path.isfile(file_path):
                mtime = os.path.getmtime(file_path)
                files_with_times.append({
                    "file_id": filename,  # 文件名
                    "file_path": file_path,  # 完整路径
                    "modified_time": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp": mtime
                })
    
    # 按时间倒序排序
    files_with_times.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return {"files": [{"file_id": f["file_id"], "modified_time": f["modified_time"]} for f in files_with_times]}

@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "老年人情感陪护对话API",
        "version": "1.0.0",
        "endpoints": {
            "/users": "GET - 获取用户列表",
            "/chat": "POST - 发送消息",
            "/end": "POST - 结束对话并保存"
        }
    }

@app.get("/users", response_model=UserListResponse)
async def get_users():
    """获取用户列表"""
    try:
        
        
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM users")
        result = [row[0] for row in cursor.fetchall()]
        conn.close()
        return UserListResponse(users=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取用户列表失败: {str(e)}")

def append_message_to_file(
    user_name: str, 
    role: str, 
    content: str, 
    conversation_file_id: Optional[str] = None,
    conversation_history: Optional[List[Dict]] = None  # ←新增
):
    """
    实时追加消息到当前对话文件
    :param user_name: 用户名
    :param role: 消息角色 ("user" 或 "assistant")
    :param content: 消息内容
    :param conversation_file_id: 对话文件ID（文件名，如 "conversation_2025-01-15_10-30-00.txt"），如果提供则追加到该文件，否则创建新文件
    """
    # ========== 调试：每次进来先打状态 ==========
    print("=" * 60)
    print(f"🎯 append_message_to_file 被调用")
    print(f"📊 全局字典状态: {current_conversation_files}")
    print(f"🔍 本次用户: '{user_name}'")
    print(f"🔍 用户是否在字典中: {user_name in current_conversation_files}")
    print(f"🔍 收到的 conversation_file_id: {conversation_file_id}")
    print(f"🔍 收到的 conversation_history 长度: {len(conversation_history) if conversation_history else 0}")
    print("=" * 60)
    print(f"🔍 进入函数时 conversation_file_id={conversation_file_id}, "
      f"user_in_dict={user_name in current_conversation_files}, "
      f"本次user_name={user_name!r}")   # ←新增
    
    # ============================================
    from utils.utils import name_to_pinyin_abbr, ensure_dir
    from database.connect_sql import get_user_id_by_name
    from datetime import datetime
    
    user_id = get_user_id_by_name(user_name)
    if user_id is None:
        return
    
    pinyin = name_to_pinyin_abbr(user_name)
    folder_name = f"{pinyin}_{user_id}"
    history_dir = os.path.join('history', folder_name)
    ensure_dir(history_dir)
    
    # 如果指定了 conversation_file_id，尝试使用该文件
    if conversation_file_id:
        # 确保文件名安全（只包含文件名，不包含路径）
        safe_filename = os.path.basename(conversation_file_id)
        if not safe_filename.endswith('.txt'):
            safe_filename += '.txt'
        
        file_path = os.path.join(history_dir, safe_filename)
        
        # 如果文件存在，使用该文件
        if os.path.exists(file_path):
            # 更新 current_conversation_files 记录
            current_conversation_files[user_name] = {
                "file_path": file_path,
                "start_time": datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d_%H-%M-%S") #os.path.getmtime(file_path)
            }
             # ↓↓↓ 新增：看看到底写没写、键是什么 ↓↓↓
            print(f"✅ 已补回内存字典，当前键列表：{list(current_conversation_files.keys())}")
            print(f"✅ 刚写入的键：{user_name!r}")
            print(f"📝 继续现有对话文件: {file_path}")
        else:
            # 文件不存在，创建新文件
            print(f"⚠️ 指定的对话文件不存在: {file_path}，将创建新文件")
            conversation_file_id = None
    
    # 如果没有指定文件或文件不存在，创建新文件
    if not conversation_file_id or user_name not in current_conversation_files:
        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(history_dir, f'conversation_{start_time}.txt')
        current_conversation_files[user_name] = {
            "file_path": file_name,
            "start_time": start_time
        }

        # 立即验证
        print(f"✅ 已添加用户 '{user_name}' 到字典")
        print(f"📁 文件路径: {file_name}")
        print(f"📊 现在字典内容: {list(current_conversation_files.keys())}")
        # 创建新文件（追加模式，如果文件不存在会自动创建）
        with open(file_name, 'w', encoding='utf-8') as f:
            if conversation_history:                       # 就是 request.conversation_history
                for msg in conversation_history:
                    # 把 emotion 占位也补上，保持格式一致
                    f.write(f"{msg['role']}: {msg['content']}（情感需求：）\n")
        print(f"🆕 新文件已预写 {len(conversation_history)} 条历史：{file_name}")
        
    # 追加消息到文件
    file_path = current_conversation_files[user_name]["file_path"]
    with open(file_path, 'a', encoding='utf-8') as f:
        # 提取情感需求（如果是用户消息）
        if role == "user":
            needs = emotion_extractor.extract_needs(content)
            if needs:
                content_with_emotion = f"{content}（情感需求：{', '.join(needs)}）"
                f.write(f"{role}: {content_with_emotion}\n")
            else:
                f.write(f"{role}: {content}（情感需求：）\n")
        else:
            f.write(f"{role}: {content}\n")
    
    # 返回当前使用的文件路径（用于VTuber记录映射）
    # 追加完本轮消息后立刻再看一眼
    with open(file_path, 'r', encoding='utf-8') as f:
        final_lines = f.readlines()
    print(f"🔚 离开函数前文件内容共 {len(final_lines)} 行：{final_lines}")
    return file_path


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    print("🚀 /chat 接口被调用")
    print(f"📊 接口开始时字典状态: {current_conversation_files}")
    print(f"👤 请求用户: {request.user_name}")


    print(f"🔥 后端实际收到的历史条数：{len(request.conversation_history)}")
    
    print(f"🔥 第一条历史：{request.conversation_history[0] if request.conversation_history else '空'}")
    print("🔍 FastAPI 收到 image_base64:", bool(request.image_base64))
    if request.image_base64:
        print("🔍 image_base64 长度:", len(request.image_base64))

    """
    对话接口 - 核心功能
    
    处理流程：
    1. 提取情感需求并更新数据库
    2. 尝试RAG检索（如果是健康问题）
    3. 调用大模型生成回复
    4. 返回回复（不保存，由/end接口统一保存）
    """
    start_time = time.time()
    
    try:
        user_name = request.user_name
        message = request.message.strip()
        
        # 实时保存用户消息（根据 conversation_file_id 决定是否继续现有对话）
        file_path = append_message_to_file(
            user_name, 
            "user", 
            message, 
            conversation_file_id=request.conversation_file_id,
            conversation_history=request.conversation_history   # ←新增

        )
        
        # 从文件路径中提取文件名（用于返回给VTuber）
        conversation_file_id = os.path.basename(file_path) if file_path else None
        
        from utils.reminder_extractor import extract_reminder_from_text
        from reminder import reminder_manager
        if not message:
            raise HTTPException(status_code=400, detail="消息不能为空")
        
        # 获取用户ID
        user_id = get_user_id_by_name(user_name)
        if user_id is None:
            raise HTTPException(status_code=404, detail=f"用户 '{user_name}' 不存在")
        
        # ===== 新增：检测并保存提醒 =====
        # 在提取提醒之前，先清理掉图片相关的描述信息
        cleaned_message = message
        # 移除 "Images in this message:" 及其后面的所有内容
        if "\nImages in this message:" in cleaned_message:
            cleaned_message = cleaned_message.split("\nImages in this message:")[0].strip()
        # 也处理可能的其他格式（如单独一行）
        cleaned_message = re.sub(r'\nImages in this message:.*$', '', cleaned_message, flags=re.DOTALL)
        cleaned_message = cleaned_message.strip()
        
        reminder_info = extract_reminder_from_text(cleaned_message)  # 使用清理后的消息
        if reminder_info:
            success = reminder_manager.add_reminder(
                user_id=user_id,
                remind_time=reminder_info["time"],
                content=reminder_info["content"],
                repeat_type=reminder_info.get("repeat_type", "none"),
                weekdays=reminder_info.get("weekdays", []),
                date=reminder_info.get("date")
            )

            if success:
                print(f"✅ 已保存提醒：{reminder_info}")
            
        # ===== 新增结束 =====

        # ===== 新增：加载对话历史，确保 conversation_history 已定义 =====
        # 优先使用请求体中携带的历史
        conversation_history = request.conversation_history or []

        # 如果请求中没有带历史，尝试从文件加载
        if not conversation_history:
            # 优先：如果指定了 conversation_file_id，直接从该文件加载
            if request.conversation_file_id: 
                try:
                    # 注意：get_user_id_by_name 和 name_to_pinyin_abbr 已在文件顶部导入，直接使用即可
                    user_id = get_user_id_by_name(user_name)
                    if user_id:
                        pinyin = name_to_pinyin_abbr(user_name)
                        folder_name = f"{pinyin}_{user_id}"
                        history_dir = os.path.join('history', folder_name)
                        safe_filename = os.path.basename(request.conversation_file_id)
                        if not safe_filename.endswith('.txt'):
                            safe_filename += '.txt'
                        file_path = os.path.join(history_dir, safe_filename)
                        
                        if os.path.exists(file_path):
                            conversation_history = load_conversation_from_file(file_path)
                            print(f"✅ 从指定文件加载历史: {file_path}，共 {len(conversation_history)} 条消息")
                        else:
                            print(f"⚠️ 指定的对话文件不存在: {file_path}")
                except Exception as e:
                    print(f"⚠️ 从指定文件加载历史失败: {str(e)}")
            
            # 备选：如果没有指定文件ID，且允许自动加载，则找最新文件
            # if not conversation_history and (request.continue_conversation is None or request.continue_conversation) and request.auto_load_history:
            #     try:
            #         should_cont, latest_file = should_continue_conversation(user_name)
            #         if should_cont and latest_file:
            #             conversation_history = load_conversation_from_file(latest_file)
            #             print(f"✅ 自动加载最新历史: {latest_file}，共 {len(conversation_history)} 条消息")
            #     except Exception as e:
            #         print(f"⚠️ 自动加载历史对话失败: {str(e)}")
        # 1. 提取情感需求并实时更新（与原有逻辑一致）
        emotional_needs = emotion_extractor.extract_needs(message)
        if emotional_needs:
            # 实时更新用户画像（与原有逻辑一致）
            if emotional_needs:
                update_user_emotional_needs(user_id, emotional_needs)
        
        # 2. 尝试RAG检索
        rag_answer = None
        source = "llm"
        
        if request.rag_enabled and is_health_related(message):
            try:
                rag_answer = get_rag_answer_or_fallback(message, request.rag_threshold)
                if rag_answer and not rag_answer.startswith("❌"):
                    source = "rag"
            except Exception as e:
                print(f"RAG处理异常: {str(e)}")
        
        # 3. 如果RAG未找到答案，使用大模型
        if source == "llm":
            # 获取用户画像
            user_profile = get_user_profile_by_name(user_name)
            
            # 构建系统提示词（与原有逻辑一致）
            profile_str = ""
            if user_profile:
                profile_items = [f"{key}：{value}" for key, value in user_profile.items() if value]
                profile_str = "以下是该用户的基本资料：\n" + "\n".join(profile_items)
            
            # 健康信息
            health_info = ""
            if is_health_related(message) and user_profile and user_profile.get("dynamic_health"):
                health_info = f"该用户曾经患有以下疾病：{user_profile['dynamic_health']}。请在合适的时机关心用户的健康情况。"
            
            # 情感需求提示
            emotional_needs_prompt = ""
            if emotional_needs:
                emotional_needs_prompt = f"用户当前情感需求：{', '.join(emotional_needs)}。请根据需求提供相应支持。\n"
            
            # 历史健康问题提醒（与原有逻辑一致）
            history_reminder = ""
            if request.conversation_history:
                history_health_issues = extract_recent_health_issues(request.conversation_history)
                if history_health_issues:
                    history_reminder = "\n\n历史健康信息提醒：\n"
                    for issue in history_health_issues:
                        history_reminder += f"- 用户之前提到过{issue}，请在回复中适当询问恢复情况\n"
            
#############################################
            


#############################################
        # 在构建消息列表时使用加载的历史
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT + "\n" + profile_str + "\n" + 
                            health_info + "\n" + emotional_needs_prompt + history_reminder
                },
                *conversation_history,  # 使用加载的历史对话
                {"role": "user", "content": message}
            ]

            # ===== 打印最终喂给模型的 messages =====
            import json, textwrap
            print("📥 最终喂给 LLM 的 messages（长度={}）：".format(len(messages)))
            for idx, m in enumerate(messages):
                content = textwrap.shorten(m["content"], 120, placeholder="...")
                print(f"  [{idx}] role={m['role']!r}  content={content!r}")
            print("=" * 60)

            # 调用大模型
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
                "stream": False
                
            }
            if request.image_base64:
                payload["image_base64"] = request.image_base64
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            print("🔍 发送给 Flask 的字段:", list(payload.keys()))
            print("🔍 payload 中 image_base64:", "image_base64" in payload)
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"].strip()
        else:
            ai_response = rag_answer
        # 实时保存AI回复
        append_message_to_file(user_name, "assistant", ai_response, conversation_file_id=conversation_file_id)
        
        response_time = time.time() - start_time
        
        return ChatResponse(
            success=True,
            response=ai_response,
            source=source,
            emotional_needs=emotional_needs,
            response_time=round(response_time, 2),
            conversation_file_id=conversation_file_id  # 返回当前使用的文件ID
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理对话失败: {str(e)}")

@app.post("/end", response_model=EndConversationResponse)
async def end_conversation(request: EndConversationRequest):
    """
    结束对话并保存 - 完全复用原有的保存逻辑
    """
    try:
        user_name = request.user_name
        messages = request.messages
        
        if not messages:
            raise HTTPException(status_code=400, detail="对话历史不能为空")
        
        # 获取用户ID
        user_id = get_user_id_by_name(user_name)
        if user_id is None:
            raise HTTPException(status_code=404, detail=f"用户 '{user_name}' 不存在")
        
        # 对话结束时间（与原有逻辑一致）
        conversation_end_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # ========== 1. 提取并保存情感需求（与原有逻辑完全一致）==========
        all_emotional_needs = []
        for message in messages:
            if message.get("role") == "user":
                needs = emotion_extractor.extract_needs(message.get("content", ""))
                all_emotional_needs.extend(needs)
        
        # 去重后更新 profiles 表中的情感需求字段（与原有逻辑一致）
        unique_needs = list(set(all_emotional_needs))
        if unique_needs:
            update_user_emotional_needs(user_id, unique_needs)
            # 记录每条情感需求到日志表（带对话结束时间戳，与原有逻辑一致）
            log_emotional_need(user_id, all_emotional_needs, conversation_end_time)
        
        # ========== 2. 处理对话文件保存 ==========
        pinyin = name_to_pinyin_abbr(user_name)
        folder_name = f"{pinyin}_{user_id}"
        history_dir = os.path.join('history', folder_name)
        ensure_dir(history_dir)
        
        # 检查是否有实时保存的文件
        realtime_file_path = None
        if user_name in current_conversation_files:
            realtime_file_path = current_conversation_files[user_name]["file_path"]
            # 从字典中移除，表示对话已结束
            del current_conversation_files[user_name]
            print(f"✅ 找到实时保存的文件: {realtime_file_path}")
        
        if realtime_file_path and os.path.exists(realtime_file_path):
            # 如果实时保存的文件存在，直接使用它（不创建新文件）
            # 可以选择重命名文件以包含结束时间，或者保持原文件名
            # 这里我们保持原文件名，因为实时保存已经完成了所有工作
            print(f"✅ 使用实时保存的文件，不重复创建: {realtime_file_path}")
            final_file_path = realtime_file_path
        else:
            # 如果没有实时保存的文件（可能是旧逻辑或异常情况），创建新文件
            print(f"⚠️ 未找到实时保存的文件，创建新文件")
            # 提取情感需求并拼接到每条用户消息后（与原有逻辑一致）
            new_messages = []
            for message in messages:
                if message.get("role") == "user":
                    needs = emotion_extractor.extract_needs(message.get("content", ""))
                    content_with_emotion = f"{message['content']}（情感需求：{', '.join(needs)}）"
                    new_messages.append({
                        "role": "user",
                        "content": content_with_emotion
                    })
                else:
                    new_messages.append(message.copy())
            
            # 写入文件（路径和格式与原有逻辑完全一致）
            file_name = os.path.join(history_dir, f'conversation_{conversation_end_time}.txt')
            with open(file_name, 'w', encoding='utf-8') as f:
                for message in new_messages:
                    f.write(f"{message['role']}: {message['content']}\n")
            final_file_path = file_name
        
        # ========== 3. 提取并更新健康信息（与原有逻辑完全一致）==========
        try:
            latest_file = get_latest_conversation_path(folder_name)  # ❌ 问题：可能找到错误的文件
            health_keywords = extract_health_from_latest_conversation(latest_file)
            health_str = ', '.join(health_keywords)
            update_user_health(user_id, health_str)
            
            # 保存健康日志（与原有逻辑一致）
            health_logs = analyze_health_log_from_conversation(latest_file)
            save_health_log_to_db(user_id, health_logs)
        except Exception as e:
            print(f"⚠️ 健康信息更新失败: {str(e)}")
            # 不抛出异常，因为对话文件已保存成功
        
        # 再次确保清空（双重保险）
        if user_name in current_conversation_files:
            del current_conversation_files[user_name]
            
        return EndConversationResponse(
            success=True,
            message="对话已保存，数据已更新",
            conversation_end_time=conversation_end_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存对话失败: {str(e)}")

    

# ==================== 新增：提醒相关接口 ====================

class ReminderNotificationRequest(BaseModel):
    """提醒通知请求"""
    user_name: str
    content: str
    reminder_id: str

@app.post("/reminder/notify")
async def notify_reminder(request: ReminderNotificationRequest):
    """
    当提醒触发时，由 reminder.py 调用此接口
    这个接口会转发提醒到 Vtuber 系统
    """
    # 这里可以通过 HTTP 请求通知 Vtuber
    # 或者通过其他方式（WebSocket、消息队列等）
    # 暂时先返回成功，后续集成
    return {"success": True, "message": "提醒通知已发送"}

@app.get("/reminders/{user_name}")
async def get_user_reminders(user_name: str):
    """获取用户的所有提醒"""
    from database.connect_sql import get_user_id_by_name
    from database.reminder_file import load_user_reminders
    
    user_id = get_user_id_by_name(user_name)
    if user_id is None:
        raise HTTPException(status_code=404, detail=f"用户 '{user_name}' 不存在")
    
    reminders = load_user_reminders(user_id)
    return {"reminders": reminders}

if __name__ == "__main__":
    reminder_manager.start()
    print("⏰ 提醒服务已随 API 自动启动")
    import uvicorn
    print("🚀 启动对话API服务...")
    print("📖 API文档地址: http://localhost:8001/docs")
    print("💡 确保数据保存格式与原有系统完全一致")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False) 
    # 启动提醒服务