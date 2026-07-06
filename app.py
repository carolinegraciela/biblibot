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

controller = ChatbotController()
pesan_timeout_error = "Wah, waktu memilihmu sudah habis, nih. Yuk mulai ulang dengan memuat sesi baru! ⏳😊"

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("menu_messages", [])
    
    session_id = cl.user_session.get("id")
    interface = ChatbotInterface(controller, session_id)
    cl.user_session.set("interface", interface)

    await main_menu(interface)

async def main_menu(interface):
    # max_retries = 3
    # retries = 0
    while True:
        await interface.clearMenuHistory()
        try:
            res_menu = await interface.showGreeting()
            
            if res_menu is None:
                break
            if not res_menu or "payload" not in res_menu:
                continue

            val = int(res_menu.get("payload").get("value"))
            
            if val == MENU_BACA_ALKITAB:
                await handle_menu(interface)
                break 
            else:
                await cl.Message(content = "💬 **Mode Tanya Jawab Aktif.**\nSilakan ketik pertanyaan Anda seputar isi Alkitab di kolom teks bawah ini:").send()
                break 
            
        except TimeoutError:
            await cl.Message(content = pesan_timeout_error).send()
            break

        except Exception:
            menu_messages = cl.user_session.get("menu_messages", [])
            if menu_messages:
                last_msg = menu_messages.pop()
                await last_msg.remove()
                cl.user_session.set("menu_messages", menu_messages)
            # retries += 1

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
        
        res_ayat = await interface.showListAyat(kitab, pasal)
        if not res_ayat: return
        ayat = int(res_ayat.get("payload").get("value"))
        
        res_final = await interface.showTeksAyat(kitab, pasal, ayat)

        if res_final and res_final.get("payload").get("value") == "kembali":
            await main_menu(interface)
            
        print(res_final.get("payload").get("value") if res_final else "No response")

    except Exception as e:
        menu_messages = cl.user_session.get("menu_messages")

        #DELETING THE UNUSED MENU
        if menu_messages:
            last_msg = menu_messages.pop()
            try:
                await last_msg.remove()
            except:
                pass
            cl.user_session.set("menu_messages", menu_messages)

        if "timeout" in type(e).__name__.lower():
            await cl.Message(content = pesan_timeout_error).send()
        else:
            print(f"[ERROR LOGGER MENU] Tipe Error: {type(e).__name__}")
            print(f"[ERROR LOGGER MENU] Detail: {repr(e)}")

            pesan_error = "Duh, sistemnya lagi kewalahan nih memuat menu. Coba ulangi lagi, ya! 🙏"
            await cl.Message(content=pesan_error).send()

#----------------------MESSAGE HANDLER-------------------------
# @cl.on_message
# async def handle_message(message: cl.Message):
#     interface = cl.user_session.get("interface")

#     if cl.user_session.get("chat_history") is None:
#         cl.user_session.set("chat_history", "")
#     history_sekarang = cl.user_session.get("chat_history")
    
#     try:
#         start_time = time.perf_counter()
#         await interface.showJawaban(message.content)
#         end_time = time.perf_counter()
#         processing_time = end_time - start_time
#         print(f"[PERFORMA] Waktu pemrosesan respons: {processing_time:.4f} detik")
#     except Exception as e:
#         if "timeout" in type(e).__name__.lower():
#             await cl.Message(content = pesan_timeout_error).send()
#         else:
#             print(f"[ERROR LOGGER] Terjadi kesalahan: {e}")
#             pesan_error = "Duh, sistemnya lagi sedikit kewalahan nih mencarikan ayat buat kamu. Coba ketik ulang pertanyaannya, ya! 🙏"
#             await cl.Message(content=pesan_error).send()


#----------------------MESSAGE HANDLER-------------------------
@cl.on_message
async def handle_message(message: cl.Message):
    if cl.user_session.get("is_processing"):
        await cl.Message(content="Tunggu sebentar ya, aku masih memproses pertanyaanmu sebelumnya... ⏳").send()
        return

    cl.user_session.set("is_processing", True)
    
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
        if "timeout" in type(e).__name__.lower():
            await cl.Message(content=pesan_timeout_error).send()
        else:
            print(f"[ERROR LOGGER] Terjadi kesalahan: {e}")
            pesan_error = "Duh, sistemnya lagi sedikit kewalahan nih mencarikan ayat buat kamu. Coba ketik ulang pertanyaannya, ya! 🙏"
            await cl.Message(content=pesan_error).send()
            
    finally:
        cl.user_session.set("is_processing", False)

