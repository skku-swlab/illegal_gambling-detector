import asyncio
import random
import pandas as pd
from telethon import TelegramClient, events
from telethon.tl.functions.messages import SetTypingRequest
from telethon.tl.types import SendMessageTypingAction
from openai import OpenAI
import re
import threading

from dotenv import load_dotenv; load_dotenv()
import os
CHATBOT_OPENAPI_API_KEY = os.getenv("CHATBOT_OPENAPI_API_KEY")
if not CHATBOT_OPENAPI_API_KEY:
    raise RuntimeError("CHATBOT_OPENAPI_API_KEY is missing")

TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID")
if not TELEGRAM_API_ID:
    raise RuntimeError("TELEGRAM_API_ID is missing")

TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")
if not TELEGRAM_API_HASH:
    raise RuntimeError("TELEGRAM_API_HASH is missing")

# ==============================
# ✅ 설정값
session_name = 'user_session'
csv_file = 'targetlinks.csv'
# ==============================

# ✅ OpenAI 클라이언트 초기화
client_openai = OpenAI(api_key=CHATBOT_OPENAPI_API_KEY)

# ✅ Telethon 클라이언트 초기화
client = TelegramClient(session_name, TELEGRAM_API_ID, TELEGRAM_API_HASH)

# ✅ 대상별 대화 히스토리 및 종료 상태 관리
chat_histories = {}
stop_flags = {}
pending_tasks = {} # 사용자별 대기중인 응답 생성 Task

# ✅ GPT 응답 생성 함수
def get_gpt_response(history):
    response = client_openai.chat.completions.create(
        model="gpt-4o",
        messages=history
    )
    return response.choices[0].message.content

# ✅ GPT에서 첫 인사 메시지 생성
def get_gpt_init_message():
    system_prompt = """
너는 사람과의 DM을 통해 사이트 가입을 위해 계좌번호를 받아야 하는 60대 남자야.
60대 남자라는 특성에 맞추어 이모티콘을 사용하지 말고, 문장은 짧고 간결하게. "니다"로 끝나는 말투와 "요"로 끝나는 말투를 적절히 사용해.
전자기기에 익숙하지 않다는 컨셉이니까 너무 정중하게 말하지 말고, 띄어쓰기나 맞춤법은 가끔 틀리게 생성해줘.
"게요" 대신 "께요", "네요" 대신 "내요"를 사용해.
말투의 예시는 "계좌번호.알고 싶습니다", "회원가입 했는데 계좌번호 못.찾겠내요", "연락할께요"와 같아.
"추천코드 없어요"가 아닌 "추천코드.없어요"처럼 띄어쓰기 대신 온점으로 대체하는 부분도 한 문장에 한 번은 무조건 넣어줘.
온점을 작성한 직후에는 띄어쓰기를 하지 마.
모든 띄어쓰기 부분을 온점으로 대체하면 이상하니까 띄어쓰기 몇 개는 그대로 써.
문장의 마지막에는 절대 마침표를 붙이지 마.

답변의 앞과 뒤에 "답변을 제공하겠습니다."와 같은 문구를 붙이지 말고, 대화에 바로 보낼 수 있는 대화문만 출력해줘.
내가 계좌번호를 주는 사람 입장이라고 생각하고, 네가 계좌 번호를 알고 싶다는 식으로 말을 거는 부분부터 시작해줘.
한 번 보낼 때 너무 문장이 길면 안 돼. 최대 두 문장 정도로 구성해.
인사와 함께 계좌번호를 알고 싶다고 해.

아래의 경우들이 있을 수 있어.
- 추천인을 묻는 경우
- 추천코드를 묻는 경우
- 어느 사이트에서 확인했는지 묻는 경우
- 입금할 금액을 묻는 경우
이 중 하나의 경우거나, 각각의 경우가 하나의 대화에서 이루어질 수 있으니 이러한 부분에 대해서도 대응이 가능해야 해.
먼저 해당 경우들에 대비하여 대답할 필요는 없고, 대화 과정에 물어보는 경우 자연스럽게 대응하면 돼.
가상으로 정보를 만들면 안 되고, 추천인과 추천코드 모두 없다고 해. 사이트는 일반적인 SNS (트위터, 인스타 등) 중에 골라서 대답하면 돼.
입금할 금액을 물어보는 경우엔 10만이라고 답해.

목적 같은 거 붙이지 말고, 인사와 함께 말을 걸면 돼.
계좌를 받은 이후에는 입금 후 어떻게 하면 되는지 물어본 후, 이에 대해 먼저 답변을 받아. 이후 다음 응답에서 "입금하고.연락드릴께요 감사합니다"라고 채팅을 마치면 돼.
미사여구를 많이 붙이지 말고 간단하게 출력해줘. 과할수록 인공지능인 게 티날 수 있으니 유의해야 해. 
말한 것과 같이 답변의 앞과 뒤에 "답변을 제공하겠습니다."와 같은 문구를 붙이지 말고, 대화에 바로 보낼 수 있는 대화문만 출력해야 해. 
최대한 자연스럽고 간결하게 출력해줘. 필요한 문장 이외에는 덧붙이지 마. 사람이 말하는 것처럼 자연스러워야 해.
    """
    response = client_openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt}
        ]
    )
    return response.choices[0].message.content, [{"role": "system", "content": system_prompt}]

async def monitor_all_conversations():
    while True:
        await asyncio.sleep(5)
        if stop_flags and all(stop_flags.values()):
            print("✅ 모든 대상과의 대화가 종료되었습니다. 클라이언트를 종료합니다.")
            await client.disconnect()
            break

def console_thread():
    while True:
        cmd = input("").strip()
        if cmd.startswith("/stop "):
            username = cmd[6:].strip().lower()
            stop_flags[username] = True
            print(f"✅ 강제 종료 플래그 설정: {username}")

# ✅ 자연스러운 타이핑 유지 함수
async def simulate_typing(peer, total_duration):
    elapsed = 0
    while elapsed < total_duration:
        await client(SetTypingRequest(
            peer=peer,
            action=SendMessageTypingAction()
        ))
        interval = random.uniform(2, 4)
        await asyncio.sleep(interval)
        elapsed += interval

# 링크 파싱 함수
def parse_targetlink(raw_link):
    raw_link = raw_link.strip()

    # @username
    if raw_link.startswith("@"):
        return raw_link[1:]

    # https://t.me/...
    if raw_link.startswith("https://t.me/"):
        path = raw_link[len("https://t.me/"):]

        # ?utm=... 같은 쿼리 제거
        path = path.split("?")[0]

        # 채널 메시지 링크 차단
        if path.startswith("c/"):
            raise ValueError(f"❌ 채널 메시지 링크는 지원되지 않습니다: {raw_link}")

        # 슬래시 제거
        path = path.strip("/")

        return path

    # username 그대로
    return raw_link

# ✅ 개별 대상 처리 루프
async def handle_conversation(targetlink_raw):
    try:
        # 링크 파싱
        targetlink = parse_targetlink(targetlink_raw)
        print(f"🟢 초기 DM 준비: {targetlink}")

        # GPT 초기 인사
        gpt_message, history = get_gpt_init_message()
        history.append({"role": "assistant", "content": gpt_message})

        chat_histories[targetlink.lower()] = history
        stop_flags[targetlink.lower()] = False

        # ✅ 먼저 entity 구함
        entity = await client.get_input_entity(targetlink)

        # ✅ simulate_typing에 entity 넘김
        delay_seconds = random.uniform(5, 10)
        print(f"✍️ 'typing...' 유지 {delay_seconds:.1f}초")
        await simulate_typing(entity, delay_seconds)

        # ✅ 메시지 전송도 entity 사용
        await client.send_message(entity, gpt_message)


        print(f"✅ Sent to {targetlink}: {gpt_message}")

    except Exception as e:
        print(f"❌ 초기 전송 실패 {targetlink_raw}: {e}")

# ✅ 초기 메시지 병렬 전송
async def send_initial_messages(targetlinks):
    await client.start()
    # tasks = [handle_conversation(targetlink) for targetlink in targetlinks]
    # await asyncio.gather(*tasks)
    for targetlink in targetlinks:
        await handle_conversation(targetlink)
        
        # ✅ 랜덤 딜레이
        delay = random.uniform(30, 120)   # 예: 30초 ~ 2분
        print(f"⏳ 다음 대상까지 {delay:.1f}초 대기")
        await asyncio.sleep(delay)

# ✅ Debounced 응답 처리 함수
async def delayed_response(targetlink, chat_id):
    try:
        # 랜덤 딜레이 (마지막 메시지부터 카운트)
        delay_seconds = random.uniform(5, 10)
        print(f"⏳ @{targetlink}: {delay_seconds:.1f}초 고민 중")
        await simulate_typing(chat_id, delay_seconds)

        # 히스토리
        history = chat_histories.get(targetlink)
        if not history:
            print(f"⚠️ @{targetlink}: 히스토리 없음 → 중단")
            return

        # GPT 응답 생성
        print(f"🤖 @{targetlink}: GPT 응답 생성 중...")
        reply = get_gpt_response(history)
        history.append({"role": "assistant", "content": reply})

        # 전송
        await client.send_message(chat_id, reply)
        print(f"✅ @{targetlink}: 답장 전송 완료: {reply}")

        # 종료 선언 판단
        if "입금하고.연락드릴께요" in reply:
            print(f"✅ @{targetlink}: 대화 종료 선언")
            stop_flags[targetlink] = True

    except asyncio.CancelledError:
        print(f"⚠️ @{targetlink}: 대기 중 취소 (새 메시지 감지)")
    except Exception as e:
        print(f"❌ @{targetlink}: 응답 오류 - {e}")

# ✅ 상대방이 답장했을 때 이벤트 핸들러
@client.on(events.NewMessage(incoming=True))
async def handle_incoming(event):
    sender = await event.get_sender()
    targetlink = sender.username.lower() if sender.username else None
    chat_id = event.chat_id
    text = event.message.message.strip()

    if not targetlink:
        print(f"⚠️ targetlink 없음 → 무시")
        return

    if stop_flags.get(targetlink):
        print(f"⚠️ @{targetlink}: 대화 종료됨 → 무시")
        return

    print(f"📩 받은 메시지 from @{targetlink}: {text}")

    # 히스토리
    history = chat_histories.get(targetlink)
    if not history:
        print(f"⚠️ @{targetlink}: 히스토리 없음 → 무시")
        return

    # 히스토리에 사용자 메시지 추가
    history.append({"role": "user", "content": text})

    # 기존에 대기 중이던 응답 작업이 있으면 취소
    existing_task = pending_tasks.get(targetlink)
    if existing_task and not existing_task.done():
        existing_task.cancel()

    # 새롭게 딜레이 + 응답 생성 작업 시작
    new_task = asyncio.create_task(delayed_response(targetlink, chat_id))
    pending_tasks[targetlink] = new_task

# ✅ 메인
async def main(targetlinks):
    await client.start()
    print("✅ Telethon 세션 로그인 완료")

    # ✅ 콘솔 입력을 감시하는 스레드 시작
    threading.Thread(target=console_thread, daemon=True).start()

    # ✅ 자동 DM 전송
    await send_initial_messages(targetlinks)

    print("✅ 초기 메시지 발송 완료. 이제 상대방 응답을 기다립니다.")

    # ✅ 종료 감시 태스크 추가
    asyncio.create_task(monitor_all_conversations())

    await client.run_until_disconnected()


if __name__ == "__main__":
    print("🤖 GPT 초기 DM 발송 준비 중...")
    user_df = pd.read_csv(csv_file)
    targetlinks = user_df['targetlink'].tolist()

    with client:
        client.loop.run_until_complete(main(targetlinks))