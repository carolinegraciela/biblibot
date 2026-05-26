import os 
import chainlit as cl
from dotenv import load_dotenv

from src.controllers.ChatbotController import ChatbotController
from src.interfaces.ChatbotInterface import ChatbotInterface

load_dotenv()

MENU_BACA_ALKITAB = 0
MENU_TANYA_JAWAB = 1

DATA_PATH = os.getenv("BIBLE_DATA_PATH")

# retriever = BibleRetriever(DATA_PATH)
controller = ChatbotController()

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("menu_messages", [])
    
    session_id = cl.user_session.get("id")
    interface = ChatbotInterface(controller, session_id)
    cl.user_session.set("interface", interface)

    await main_menu(interface)

async def main_menu(interface):
    await interface.clearMenuHistory()
    res_menu = await interface.showGreeting()
    if res_menu:
        val = int(res_menu.get("payload").get("value"))
        
        if val == MENU_BACA_ALKITAB:
            await handle_menu(interface)            
        else:
            await cl.Message(content="💬 **Mode Tanya Jawab Aktif.**\nSilakan ketik pertanyaan Anda seputar isi Alkitab di kolom teks bawah ini:").send()

async def handle_menu(interface):
    res_perjanjian = await interface.showMenuPerjanjian()
    if not res_perjanjian: return
    perjanjian_id = int(res_perjanjian.get("payload").get("value"))
    
    res_kitab = await interface.showListKitab(perjanjian_id)
    if not res_kitab: return
    kitab = str(res_kitab.get("payload").get("value"))
    
    res_pasal = await interface.showListPasal(kitab)
    if not res_pasal: return
    pasal = int(res_pasal.get("payload").get("value"))
    
    res_ayat = await interface.showListAyat(pasal)
    if not res_ayat: return
    ayat = int(res_ayat.get("payload").get("value"))
    
    res_final = await interface.showTeksAyat(ayat)

    if res_final.get("payload").get("value") == "kembali":
        await main_menu(interface)
    print(res_final.get("payload").get("value"))
    # await cl.sleep(2) 
    # await main_menu(interface)

#----------------------MESSAGE HANDLER-------------------------
@cl.on_message
async def handle_message(message: cl.Message):
    interface = cl.user_session.get("interface")

    if cl.user_session.get("chat_history") is None:
        cl.user_session.set("chat_history", "")
    history_sekarang = cl.user_session.get("chat_history")
    
    # if controller.is_gibberish(message.content):
    #     pesan_error = "⚠️ Maaf, pesan Anda tidak dapat dipahami. Tolong ketikkan yang benar."
    #     await cl.Message(content = pesan_error).send()

    #     return

    await interface.showJawaban(message.content)

