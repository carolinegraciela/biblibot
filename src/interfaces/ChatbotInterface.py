import chainlit as cl

MENU_BACA_ALKITAB = 0
MENU_TANYA_JAWAB = 1

class ChatbotInterface:
    def __init__(self, controller, session_id:int):
        self.session_id = session_id
        self._controller = controller
    
    def _menu_actions(self):
        return [
            cl.Action(name="pilih_menu", payload={"value": int(MENU_BACA_ALKITAB)}, label="📖 Baca Alkitab"),
            cl.Action(name="pilih_menu", payload={"value": int(MENU_TANYA_JAWAB )}, label="🔍 Tanya Jawab")
        ]
    
    async def showGreeting(self) -> str:
        greeting = (
            "## ✝️ Selamat datang di **Biblibot**!\n\n"
            "Saya adalah asisten Alkitab berbasis AI. Saya siap membantu Anda. Yuk, pilih tombol menu di bawah untuk melanjutkan!\n\n"
            "- 📖 **Baca Alkitab Digital** — cari ayat berdasarkan Kitab, Pasal, dan Ayat\n\n"
            "- 🔍 **Tanya Jawab** — ajukan pertanyaan, saya jawab berdasarkan ayat Alkitab\n"
            "Yuk, pilih salah satu menu di bawah untuk memulai:"
        )

        action_msg = cl.AskActionMessage(
            content = greeting,
            actions = self._menu_actions(),
            raise_on_timeout = False,
            timeout = 30
        )
        
        menu_messages = cl.user_session.get("menu_messages") or []
        menu_messages.append(action_msg)
        cl.user_session.set("menu_messages", menu_messages)

        res = await action_msg.send()
        if res:
            action_msg.content = f"Dipilih: {res.get('label')}"
            await action_msg.update()
        elif res is None:
            action_msg.actions = []
            action_msg.content = "Wah, waktu memilihmu sudah habis, nih. Yuk mulai ulang di menu baru ⏳😊"

            await action_msg.update()
            return None 
        return res


    #-------------------------- MENU 2: TANYA JAWAB BIBLIBOT -------------------------- 
    async def showJawaban(self, user_query: str):
        msg = cl.Message(content=f"🔍 Menganalisis pertanyaan: '{user_query}' ...")
        await msg.send()

        jawaban = await self._controller.generate_response(user_query, self.session_id)    
        try:
            jawaban = await self._controller.generate_response(user_query, self.session_id)    
            print(f"[DEBUG UI] Tipe data jawaban: {type(jawaban)}")            
            msg.content = str(jawaban)

            await msg.update()
            
        except Exception as e:
            print(f"[ERROR UI] Gagal mengupdate interface: {str(e)}")
            msg.content = "Sistemnya kewalahan, nih. Yuk, ulang di sesi baru! 🙏"
            await msg.update()

        await msg.update()
    
    
    #-------------------------- MENU 1: BACA ALKITAB DIGITAL --------------------------
    async def showMenuPerjanjian(self):
        actions = [
            cl.Action(name="pilih_perjanjian", payload={"value": 0}, label="📜 Perjanjian Lama (PL)"),
            cl.Action(name="pilih_perjanjian", payload={"value": 1}, label="📖 Perjanjian Baru (PB)")
        ]
        action_msg = cl.AskActionMessage(
            content="Silakan pilih kategori Alkitab yang ingin Anda baca:",
            actions=actions,
            raise_on_timeout=True
        )
        menu_messages = cl.user_session.get("menu_messages") or []
        menu_messages.append(action_msg)
        cl.user_session.set("menu_messages", menu_messages)
        res = await action_msg.send()
        if res:
            action_msg.content = f"Dipilih: {res.get('label')}"
            await action_msg.update()
        return res

    async def showListKitab(self, id: int) -> list[str]:
        daftar_kitab = await self._controller.getDaftarKitab(id)
        actions = [
            cl.Action(name="pilih_kitab", payload={"value": kitab}, label=kitab) 
            for kitab in daftar_kitab
        ]
        
        judul = "Daftar Kitab **Perjanjian Lama**:" if id == 0 else "Daftar Kitab **Perjanjian Baru**:"
        
        action_msg = cl.AskActionMessage(
            content = judul,
            actions = actions,
            raise_on_timeout=True,
        )
        menu_messages = cl.user_session.get("menu_messages") or []
        menu_messages.append(action_msg)
        cl.user_session.set("menu_messages", menu_messages)
        res = await action_msg.send()
        if res:
            action_msg.content = f"Dipilih: {res.get('label')}"
            await action_msg.update()
        return res


    async def showListPasal(self, kitab: str) -> list[int]:
        daftar_pasal = await self._controller.getDaftarPasal(kitab)
        actions = [
            cl.Action(name="pilih_pasal", payload={"value": pasal}, label=str(pasal)) 
            for pasal in daftar_pasal
        ]
        
        judul = f"Daftar Pasal dalam Kitab {kitab}:"
        
        action_msg = cl.AskActionMessage(
            content = judul,
            actions = actions,
            raise_on_timeout=True
        )
        menu_messages = cl.user_session.get("menu_messages") or []
        menu_messages.append(action_msg)
        cl.user_session.set("menu_messages", menu_messages)
        res = await action_msg.send()
        if res:
            action_msg.content = f"Dipilih: {res.get('label')}"
            await action_msg.update()
        return res

    
    async def showListAyat(self, kitab: str, pasal: int) -> list[int]:
        daftar_ayat = await self._controller.getDaftarAyat(kitab, pasal)
        actions = [
            cl.Action(name="pilih_ayat", payload={"value": ayat}, label=str(ayat)) 
            for ayat in daftar_ayat
        ]
        
        judul = f"Pilih ayat:"
        
        action_msg = cl.AskActionMessage(
            content = judul,
            actions = actions,
            raise_on_timeout=True
        )
        menu_messages = cl.user_session.get("menu_messages") or []
        menu_messages.append(action_msg)
        cl.user_session.set("menu_messages", menu_messages)
        res = await action_msg.send()
        if res:
            action_msg.content = f"Dipilih: {res.get('label')}"
            await action_msg.update()
        return res
    
    async def showTeksAyat(self, kitab: str, pasal: int, ayat: int) -> str:
        kitab, pasal, ayat, teks = await self._controller.getTeksAyat(kitab, pasal, ayat)
        actions = [
            cl.Action(
                name = "kembali_menu", 
                payload = {"value": "kembali"}, 
                label = "🏠 Kembali ke Menu Utama"
            )
        ]
        
        action_msg = cl.AskActionMessage(
            content = f"📖 **{kitab} {pasal}:{ayat}**\n\n> {teks}",
            actions = actions,
            timeout = 180,
            raise_on_timeout=True
        )
        menu_messages = cl.user_session.get("menu_messages") or []
        menu_messages.append(action_msg)
        cl.user_session.set("menu_messages", menu_messages)
        res = await action_msg.send()
        if res:
            action_msg.content = f"Dipilih: {res.get('label')}"
            await action_msg.update()
        return res

    async def clearMenuHistory(self):
        menu_messages = cl.user_session.get("menu_messages") or []
        
        for msg in menu_messages:
            try:
                await msg.remove()
            except Exception:
                pass 

        cl.user_session.set("menu_messages", [])