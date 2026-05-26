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
            "Saya adalah asisten Alkitab berbasis AI. Saya siap membantu Anda:\n\n"
            "- 📖 **Baca Alkitab Digital** — cari ayat berdasarkan Kitab, Pasal, dan Ayat\n\n"
            "- 🔍 **Tanya Jawab** — ajukan pertanyaan, saya jawab berdasarkan ayat Alkitab\n"
            "Pilih mode di bawah untuk memulai:"
        )

        action_msg = cl.AskActionMessage(
            content = greeting,
            actions = self._menu_actions(),
        )
        
        menu_messages = cl.user_session.get("menu_messages") or []
        menu_messages.append(action_msg)
        cl.user_session.set("menu_messages", menu_messages)

        res = await action_msg.send()
        if res:
            action_msg.content = f"Dipilih: {res.get('label')}"
            await action_msg.update()
        return res

    #-------------------------- MENU 2: TANYA JAWAB BIBLIBOT -------------------------- 
    async def showJawaban(self, user_query: str):
        msg = cl.Message(content=f"🔍 Menganalisis pertanyaan: '{user_query}' ...")
        await msg.send()

        jawaban = self._controller.generate_response(user_query)    
        msg.content = f"Hasil Retrieval:\n**{jawaban}**)"

        await msg.update()
    
    
    #-------------------------- MENU 1: BACA ALKITAB DIGITAL --------------------------
    async def showMenuPerjanjian(self):
        actions = [
            cl.Action(name="pilih_perjanjian", payload={"value": 0}, label="📜 Perjanjian Lama (PL)"),
            cl.Action(name="pilih_perjanjian", payload={"value": 1}, label="📖 Perjanjian Baru (PB)")
        ]
        action_msg = cl.AskActionMessage(
            content="Silakan pilih kategori Alkitab yang ingin Anda baca:",
            actions=actions
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
            actions = actions
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
            actions = actions
        )
        menu_messages = cl.user_session.get("menu_messages") or []
        menu_messages.append(action_msg)
        cl.user_session.set("menu_messages", menu_messages)
        res = await action_msg.send()
        if res:
            action_msg.content = f"Dipilih: {res.get('label')}"
            await action_msg.update()
        return res

    
    async def showListAyat(self, pasal: int) -> list[int]:
        daftar_ayat = await self._controller.getDaftarAyat(pasal)
        
        actions = [
            cl.Action(name="pilih_ayat", payload={"value": ayat}, label=str(ayat)) 
            for ayat in daftar_ayat
        ]
        
        judul = f"Pilih ayat:"
        
        action_msg = cl.AskActionMessage(
            content = judul,
            actions = actions
        )
        menu_messages = cl.user_session.get("menu_messages") or []
        menu_messages.append(action_msg)
        cl.user_session.set("menu_messages", menu_messages)
        res = await action_msg.send()
        if res:
            action_msg.content = f"Dipilih: {res.get('label')}"
            await action_msg.update()
        return res
    
    async def showTeksAyat(self, ayat: int) -> str:
        kitab, pasal, ayat, teks = await self._controller.getTeksAyat(ayat)
        actions = [
            cl.Action(
                name = "kembali_menu", 
                payload = {"value": "kembali"}, 
                label = "🏠 Kembali ke Menu Utama"
            )
        ]
        
        action_msg = cl.AskActionMessage(
            content = f"📖 **{kitab} {pasal}:{ayat}**\n\n> {teks}",
            actions = actions
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