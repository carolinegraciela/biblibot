import os 
import time

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
    while True:
        await interface.clearMenuHistory()
        try:
            res_menu = await interface.showGreeting()
            
            if res_menu:
                val = int(res_menu.get("payload").get("value"))
                
                if val == MENU_BACA_ALKITAB:
                    await handle_menu(interface)
                    break 
                else:
                    await cl.Message(content="💬 **Mode Tanya Jawab Aktif.**\nSilakan ketik pertanyaan Anda seputar isi Alkitab di kolom teks bawah ini:").send()
                    break 
                    
        except Exception:
            menu_messages = cl.user_session.get("menu_messages")
            last_msg = menu_messages.pop()
            await last_msg.remove()
            cl.user_session.set("menu_messages", menu_messages)

async def handle_menu(interface):
    try:
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

        if res_final and res_final.get("payload").get("value") == "kembali":
            await main_menu(interface)
            
        print(res_final.get("payload").get("value") if res_final else "No response")

    except Exception as e:
        error_message = str(e).lower()
        menu_messages = cl.user_session.get("menu_messages")

        #DELETING THE UNUSED MENU
        if menu_messages:
            last_msg = menu_messages.pop()
            try:
                # Hapus HANYA pesan terakhir dari layar UI pengguna
                await last_msg.remove()
            except:
                pass
            cl.user_session.set("menu_messages", menu_messages)

        if "timed out" in error_message or "timeout" in error_message or "no action was taken" in error_message:
            pesan_cantik = "Wah, waktu memilihmu sudah habis, nih. Yuk mulai ulang dengan mengetik sapaan di kolom chat! ⏳😊"
            await cl.Message(content=pesan_cantik).send()
        else:
            print(f"[ERROR LOGGER MENU] Terjadi kesalahan: {e}")
            pesan_error = "Duh, sistemnya lagi kewalahan nih memuat menu. Coba ulangi lagi, ya! 🙏"
            await cl.Message(content=pesan_error).send()

#----------------------MESSAGE HANDLER-------------------------
@cl.on_message
async def handle_message(message: cl.Message):
    interface = cl.user_session.get("interface")

    if cl.user_session.get("chat_history") is None:
        cl.user_session.set("chat_history", "")
    history_sekarang = cl.user_session.get("chat_history")
    
    try:
        start_time = time.perf_counter()
        await interface.showJawaban(message.content)
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        print(f"[PERFORMA] Waktu pemrosesan respons: {processing_time:.4f} detik")
    except Exception as e:
        error_message = str(e).lower()
        if "timed out" in error_message or "timeout" in error_message or "no action was taken" in error_message:
            pesan_cantik = "Wah, waktu kamu habis, nih. Yuk refresh dan ulang session baru! ⏳😊"
            await cl.Message(content=pesan_cantik).send()
        else:
            print(f"[ERROR LOGGER] Terjadi kesalahan: {e}")
            pesan_error = "Duh, sistemnya lagi sedikit kewalahan nih mencarikan ayat buat kamu. Coba ketik ulang pertanyaannya, ya! 🙏"
            await cl.Message(content=pesan_error).send()

