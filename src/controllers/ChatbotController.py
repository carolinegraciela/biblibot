import asyncio
import json
import os

#CLASS
from src.services.BibleRetriever import BibleRetriever 
from src.services.LLmService import LLmService

#COMMUNITY
from langchain_huggingface import HuggingFaceEmbeddings

#QUERY REWRITING
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

#MEMORY
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


class ChatbotController:
    def __init__(self):
        self.__embedding_model = HuggingFaceEmbeddings(
            model_name = "BAAI/bge-m3",
            model_kwargs = {'device': 'cpu'}
        )
        self._retriever = BibleRetriever(self.__embedding_model)
        self.llama_service = LLmService()
        self.llm = self.llama_service.generateResponse()
        self.reranker = self.llama_service.rerankerModel()
        self.store = {}

        
    # 2nd MENU: Question-Answering with Biblibot
    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory: 
        """Get or create a history"""
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def intent_classification(self, original_query, session_id):
        examples = [
            {"input": "ayat Alkitab yang menyatakan bahwa ...", "output": "[VALID_QUERY]"},
            {"input": "ayat Alkitab ttg Allah Tritunggal", "output": "[VALID_QUERY]"},
            {"input": "ayat yang menyatakan Yesus adalah Allah", "output": "[VALID_QUERY]"},
            
            # --- CONTRASTIVE PAIRING ---
            {"input": "ayat tentang keselamatan hanya ada dalam yesus", "output": "[VALID_QUERY]"},
            {"input": "apakah benar keselamatan hanya ada dalam yesus? tolong jelaskan", "output": "[EXEGESIS_REQUEST]"},            
            {"input": "ayat tentang perumpamaan anak yang hilang", "output": "[VALID_QUERY]"},
            {"input": "apa makna dari perumpamaan anak yang hilang?", "output": "[EXEGESIS_REQUEST]"},
            {"input": "arti dari ayat ...?", "output": "[EXEGESIS_REQUEST]"},
            {"input": "makna perumpamaan ...?", "output": "[EXEGESIS_REQUEST]"},

            # ---------------------------

            {"input": "gimana cara bikin nasi goreng jawa?", "output": "[OUT_OF_DOMAIN]"},
            {"input": "halo biblibot!", "output": "[GREETING]"},
            {"input": "asdfghjkl", "output": "[GIBBERISH]"}
        ]

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}"),
        ])

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt = example_prompt,
            examples = examples,
        )

        system_instruction = """### INSTRUKSI
        Anda adalah sistem klasifikasi Niat Pengguna (Intent Classification) untuk chatbot pencarian ayat.
        Tugas Anda adalah menganalisis input pengguna dan WAJIB memberikan output SATU TAG SAJA berdasarkan hierarki 5 kondisi berikut:

        1. [EXEGESIS_REQUEST] : Jika pengguna secara eksplisit menanyakan makna, arti, maksud, tafsiran, atau penjelasan teologis (misal: "apa arti", "maksud dari", "tafsirkan"). JANGAN gunakan tag ini jika pengguna murni mencari ayat atau menceritakan peristiwa.
        2. [GIBBERISH] : Jika input adalah ketikan acak, tidak bermakna, atau hanya simbol (misal: "asdfg").
        3. [GREETING] : Jika input murni sapaan (halo, pagi, shalom).
        4. [OUT_OF_DOMAIN] : HANYA gunakan jika input MURNI tentang hal duniawi (tutorial, tanya resep, info pariwisata murni seperti "berapa harga tiket ke bali"). 
           *PANTANGAN KERAS*: Jika pengguna menanyakan ETIKA, BATASAN MORAL, atau BOLEH/TIDAKNYA suatu aktivitas secara rohani (misal: "boleh gak pakai baju terbuka saat liburan?", "nonton bioskop dosa gak?"), JANGAN gunakan tag ini! Itu adalah [VALID_QUERY].
        5. [VALID_QUERY] : Jika pengguna membagikan masalah hidup, mencari ayat, menyebutkan nama tokoh, atau meminta solusi rohani.

        ### ATURAN MUTLAK
        - OUTPUT HANYA BOLEH BERUPA SATU TAG KODE DI ATAS. 
        - DILARANG MEMBERIKAN PENJELASAN, BASA-BASI, ATAU TEKS TAMBAHAN APAPUN.
        """

        final_prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            few_shot_prompt,  
            ("placeholder", "{chat_history}"),                
            ("human", "{original_query}")    
        ])

        model_with_stop = self.llm.bind(stop=["User:", "###", "Human:"])
        
        intent_classifier = final_prompt | model_with_stop | StrOutputParser()
        history_obj = self.get_session_history(session_id)
        pesan_riwayat_sebelumnya = history_obj.messages

        response = intent_classifier.invoke({
            "chat_history": pesan_riwayat_sebelumnya,
            "original_query": original_query})

        clean_response = response.strip()
        return clean_response

    def hyde(self, kueri):
        system_template = """### INSTRUKSI
        Anda adalah mesin menjawab pertanyaan (kueri) BERUPA DOKUMEN HIPOTESIS BERBASIS AYAT ALKITAB dengan tujuan mengoptimalkan pencarian semantik (Vector Search) pada Alkitab Terjemahan Baru (TB).
        Input yang Anda terima adalah kueri mentah dari pengguna (bisa berupa curhatan, bahasa gaul, pertanyaan historis, atau masalah hidup).

        Tugas Anda: Ekstrak inti teologis atau historis dari input tersebut, lalu tulis jawaban sesuai PRINSIP ALKITABIAH dengan referensi jawaban BERDASARKAN AYAT ALKITAB.

        ### ATURAN PERCABANGAN 5W1H (WAJIB DIPATUHI):
        1. SOLUSI / EKSPLANASI (Masalah Hidup/Curhat/Doktrin): Buat 2-3 kalimat padat bergaya Alkitab TB yang memuat prinsip teologis tegas untuk kueri tersebut.
        2. FAKTA SPESIFIK (Who/Where/When - Siapa/Di mana/Kapan): Jika kueri mencari fakta tunggal, daftar nama, atau sejarah, DILARANG KERAS membuat dialog karangan. Buat 1-2 kalimat deklaratif yang LANGSUNG MENYEBUTKAN NAMA TOKOH, LOKASI, ATAU WAKTU secara spesifik, meniru gaya penulisan silsilah atau daftar nama dalam Alkitab.
        3. FAKTA TIDAK DIKETAHUI (FAIL-SAFE MUTLAK): Jika Anda tidak ingat persis 100% fakta sejarah atau nama tokoh yang ditanyakan (seperti murid pengganti, nama raja, dsb), DILARANG KERAS MENGARANG CERITA atau menebak-nebak nama. Cukup ulangi inti pertanyaan menjadi pernyataan pencarian kata kunci yang padat tanpa gaya bahasa arkais.
        4. DAFTAR PANJANG & KISAH UMUM (Summary Rule): Jika kueri meminta topik yang sangat luas (contoh: semua mujizat) atau daftar panjang (contoh: 10 Hukum Allah), JANGAN tulis seluruh daftar secara lengkap. Tuliskan ayat pengantar utama, ayat kesimpulan, atau merangkumnya menggunakan kata kunci dari ayat tersebut agar jangkar pencarian tetap fokus.

        ### ATURAN MUTLAK & ANTI-HALUSINASI:
        1. Jawab HANYA dengan teks hipotetis. DILARANG KERAS memberikan pengantar atau akhiran (seperti "Ayat yang relevan:", "Mari kita baca"). Langsung berikan isi teks ayatnya.
        2. KESETARAAN TEOLOGIS (THEOLOGICAL SYNONYM): Jika pengguna menanyakan konsep doktrin menggunakan istilah modern/awam, Anda WAJIB mengubahnya ke dalam terminologi Alkitab TB yang tepat. 
        - "Hukum taurat" -> jangan hanya melihat "TAURAT" saja. Dua kata ini jangan terpisahkan.
        - "Karma / balas dendam alam" -> "Tabur tuai / pembalasan adalah hak Tuhan"
        - "Reinkarnasi / hantu" -> "Kebangkitan orang mati / roh-roh jahat"
        3. TRANSLASI BAHASA MODERN (KAMUS WAJIB): 
        - "komsel / ibadah / gereja" = "pertemuan ibadah / persekutuan"
        - "badminton / padel / hobi / nongkrong" = "kesenangan duniawi / perkara dunia" (PANTANGAN: Jangan hubungkan olahraga dengan ayat tentang suami-istri, percabulan, atau hawa nafsu!).
        - "pacaran beda agama / beda keyakinan" = "pasangan yang tidak seimbang / terang dan gelap" (PANTANGAN: Jangan gunakan ayat tentang kesetaraan manusia dalam Kristus untuk menoleransi hal ini!).
        - "bos / atasan" = "tuan".
        - "rokok / merokok / tato / vaping / miras / pinjol / judi slot" = "merajah tanda pada kulit / mencemari bait Roh Kudus / diperhamba oleh sesuatu / hamba uang / ketamakan"
        """

        examples = [
            {
                "input": "saya sedang diajak teman saya untuk padel, tapi jamnya berkaitan dengan jam komsel. bimbang.",
                "output": "Carilah dahulu Kerajaan Allah dan kebenarannya, maka semuanya itu akan ditambahkan kepadamu. Janganlah kita menjauhkan diri dari pertemuan-pertemuan ibadah kita, seperti dibiasakan oleh beberapa orang, tetapi marilah kita saling menasihati."
            },
            {
                "input": "Saya mau ke lombok sama temen2, ke pantainya. Apakah tidak apa-apa untuk pakai pakaian yang terbuka?",
                "output": "Muliakanlah Allah dengan tubuhmu, sebab tubuhmu adalah bait Roh Kudus yang diam di dalam kamu. Demikian juga hendaknya perempuan berdandan dengan pantas, dengan sopan dan sederhana."
            },
            {
                "input": "Ada orang yang deketin saya, dan dia tipe saya pol. Tapi dia dari agama berbeda, gimana menurutmu?",
                "output": "Janganlah kamu merupakan pasangan yang tidak seimbang dengan orang-orang yang tak percaya. Sebab persamaan apakah terdapat antara kebenaran dan kedurhakaan? Atau bagaimanakah terang dapat bersatu dengan gelap?"
            },
            {
                "input": "ayat untuk orang malas ke gereja",
                "output": "Ingatlah dan kuduskanlah hari sabat. Jangan menjauhkan diri dari pertemuan ibadah."
            },
            {
                "input": "ayat untuk orang yang malas baca Alkitab",
                "output": "Firman-Mu itu pelita bagi kakiku dan terang bagi jalanku. Janganlah engkau lupa memperkatakan kitab Taurat ini, tetapi renungkanlah itu siang dan malam, supaya engkau bertindak hati-hati sesuai dengan segala yang tertulis di dalamnya, sebab dengan demikian perjalananmu akan berhasil dan engkau akan beruntung."
            },
            {
                "input": "Siapakah anak daud yg menjadi raja israel?",
                "output": "Kemudian seluruh jemaat mengakui Salomo, anak Daud, sebagai raja. Mereka mengurapi dia bagi TUHAN menjadi raja menggantikan Daud, ayahnya."
            },
            {
                "input": "siapa saja murid yesus?",
                "output": "Inilah nama kedua belas rasul itu: Pertama Simon yang disebut Petrus dan Andreas saudaranya, dan Yakobus anak Zebedeus dan Yohanes saudaranya, Filipus dan Bartolomeus, Tomas dan Matius pemungut cukai, Yakobus anak Alfeus, dan Tadeus, Simon orang Zelot dan Yudas Iskariot yang mengkhianati Dia."
            },
            {
                "input": "Ayat ttg kisah Abraham mau membunuh Ishak",
                "output": "Allah mencobai Abraham dan berfirman kepadanya agar mempersembahkan Ishak, anak tunggalnya yang dikasihinya, sebagai korban bakaran. Ketika Abraham mengulurkan tangannya dan mengambil pisau untuk menyembelih anaknya, berserulah Malaikat TUHAN dari langit melarangnya."
            },
            {
                "input": "Ayat tentang keselamatan",
                "output": "Sebab karena kasih karunia kamu diselamatkan oleh iman; itu bukan hasil usahamu, tetapi pemberian Allah. Keselamatan tidak ada di dalam siapapun juga selain di dalam Dia. Sebab di bawah kolong langit ini tidak ada nama lain yang diberikan kepada manusia yang olehnya kita dapat diselamatkan."
            },
            {
                "input": "siapakah murid pengganti yudas iskariot",
                "output": "Mereka membuang undi bagi kedua orang itu dan undi itu jatuh kepada Matias dan dengan demikian ia ditambahkan kepada bilangan kesebelas rasul itu menggantikan Yudas Iskariot."
            },
            {
                "input": "apakah hukum yg menggantikan hukum taurat",
                "output": "Sebab hukum Roh yang memberi hidup telah memerdekakan kamu dalam Kristus dari hukum dosa dan hukum maut. Kasih karunia Allah yang menyelamatkan semua manusia sudah nyata, dan kasih adalah kegenapan hukum Taurat."
            },            
            {
                "input": "Insecure banget lihat temen-temen seangkatan udah pada sukses, fomo rasanya, depresi mau nyerah aja.",
                "output": "Janganlah kamu menjadi hamba uang dan cukupkanlah dirimu dengan apa yang ada padamu. Serahkanlah segala kekuatiranmu kepada-Nya, sebab Ia yang memelihara kamu. Hati yang tenang menyegarkan tubuh, tetapi iri hati membusukkan tulang."
            },
            {
                "input": "Kerja di kantorku toxic banget, bosnya semena-mena dan temennya pada suka julid dan nusuk dari belakang.",
                "output": "Hai hamba-hamba, taatilah tuanmu yang di dunia dengan takut dan gentar. Janganlah ada perkataan kotor keluar dari mulutmu, tetapi pakailah perkataan yang baik untuk membangun. Orang yang curang menimbulkan pertengkaran, dan seorang pemfitnah menceraikan sahabat yang karib."
            },
            {
                "input": "Saya lagi pusing banget, terjerat pinjol dan slot karena pengen cepet kaya. Sekarang dikejar utang.",
                "output": "Siapa yang ingin cepat menjadi kaya, tidak akan luput dari hukuman. Orang kaya menguasai orang miskin, yang berhutang menjadi budak dari yang menghutangi. Peliharalah dirimu dari segala ketamakan."
            },
            {
                "input": "Siapa nama penjaga pintu gerbang bait suci pada zaman raja Yosia?",
                "output": "Penjaga pintu gerbang bait suci pada zaman raja Yosia di Yerusalem." # FAIL-SAFE Rule teraktivasi
            },
            {
                "input": "kisah mujizat yesus",
                "output": "Yesus dari Nazaret adalah seorang yang telah ditentukan Allah dan yang dinyatakan kepadamu dengan kekuatan-kekuatan dan mujizat-mujizat dan tanda-tanda. Masih banyak hal-hal lain lagi yang diperbuat oleh Yesus, tetapi jikalau semuanya itu dituliskan satu per satu, maka agaknya dunia ini tidak dapat memuat semua kitab yang harus ditulis itu." # Summary Rule untuk kisah masif/jamak
            },
            {
                "input": "mujizat pertama yesus",
                "output": "Hal itu dibuat Yesus di Kana yang di Galilea, sebagai yang pertama dari tanda-tanda-Nya dan dengan itu Ia telah menyatakan kemuliaan-Nya, yaitu mengubah air menjadi anggur pada perjamuan kawin."
            },
            {
                "input": "ayat tentang 9 buah roh",
                "output": "Tetapi buah Roh ialah: kasih, sukacita, damai sejahtera, kesabaran, kemurahan, kebaikan, kesetiaan, kelemahlembutan, penguasaan diri. Tidak ada hukum yang menentang hal-hal itu."
            },
            {
                "input": "ayat tentang 10 hukum Allah",
                "output": "Jangan ada padamu allah lain di hadapanku. Jangan membuat bagimu patung yang menyerupai apa pun. Jangan menyebut nama Tuhan Allahmu dengan sembarangan. Ingat dan kuduskanlah hari sabat. Hormatilah ayahmu dan ibumu. Jangan membunuh. Jangan berzinah. Jangan mencuri. Jangan mengucapkan saksi duta. Jangan mengingini milik sesamamu."
            },                        
            {
                "input": "ayat tentang hukum kasih",
                "output": "Kasihilah Tuhan Allahmu dengan segenap hatimu dan segenap kekuatanmu dan segenap akal budimu. Kasihilah sesamamu manusia seperti diri sendiri."
            }

        ]

        example_prompt = ChatPromptTemplate.from_messages([
        ("human", "Input:\n{input}"),
        ("ai", "{output}")])

        few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
         )
        
        hyde_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        few_shot_prompt,
        ("human", "Input:\n{query}")    
        ])

        try:
            hyde_chain = hyde_prompt | self.llm | StrOutputParser()
            response = hyde_chain.invoke({"query": kueri}).strip()
            
            if response.lower().startswith("berikut adalah") or response.lower().startswith("tentu"):
                response = '\n'.join(response.split('\n')[1:]).strip()

            print(f"Dokumen hipotesis: {response}")
            return response
            
        except Exception as e:
            print(f"[API ERROR] HyDE Generation Failed: {str(e)}")
            return kueri 

    def reranking(self, query: str, retrieved_docs: str, top_n: int = 3):
        sentence_pairs = [[query, doc.page_content] for doc in retrieved_docs]
        scores = self.reranker.score(
            sentence_pairs
        )
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        
        ranked = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
        best_docs = []

        for score, doc in ranked[:top_n]:
            print(f"Rerank Score: {score:.4f} | {doc.page_content[:80]}...")        
            if hasattr(doc, 'metadata'):
                doc.metadata["rerank_score"] = float(score)            
            best_docs.append(doc)
            
        return best_docs

    async def generate_response(self, user_query: str, session_id: str):
        intent_tag = await asyncio.to_thread(self.intent_classification, user_query, session_id)
        print(f"Query Intention: {intent_tag}")

        if "[GIBBERISH]" in intent_tag:
            return "Maaf, saya tidak mengerti pesan yang kamu kirimkan. Bisa tolong ketik ulang dengan lebih jelas? 😊"
        elif "[GREETING]" in intent_tag:
            return "Halo, shalom! Saya Biblibot, asisten pencarian ayat Alkitab secara kontekstual. Ada topik pergumulan atau kisah Alkitab yang ingin kamu cari hari ini? 🙏"
        elif "[OUT_OF_DOMAIN]" in intent_tag:
            return (
                "Maaf, saya hanya didesain untuk membantu mencari ayat Alkitab berdasarkan topik atau pergumulan rohani. "
                "Saya tidak bisa menjawab pertanyaan umum di luar Alkitab seperti resep masakan, tips teknologi, atau pariwisata. "
                "Silakan ketikkan topik rohani atau masalah yang sedang kamu hadapi ya! ✨"
            )
        elif "[EXEGESIS_REQUEST]" in intent_tag:
            return (
                "Maaf, saya dirancang khusus sebagai asisten mesin pencari ayat secara kontekstual, "
                "bukan untuk menafsirkan, menjelaskan makna teologis, atau memberikan arti pada ayat tertentu.\n\n"
                "Untuk mendapatkan penjelasan yang mendalam dan akurat mengenai tafsiran tersebut, "
                "saya sangat menyarankan untuk berdiskusi dengan pendeta, pembimbing rohani, atau membaca buku tafsir Alkitab. 🙏\n\n"
                "Namun, jika kamu ingin mencari ayat-ayat lain dengan *tema serupa*, silakan ketikkan topik pergumulan atau kata kuncinya!"
            )
        elif "[KUERI TIDAK VALID" in intent_tag or not intent_tag.strip():
            return (
                "Maaf, kueri yang kamu masukkan tidak dapat saya proses. "
                "Pastikan kamu memasukkan topik kehidupan, cerita tokoh Alkitab, atau situasi yang sedang kamu alami "
                "agar saya bisa mencarikan ayat Alkitab yang tepat untukmu. Silakan coba lagi! 😊"
            )
        
        hyde = await asyncio.to_thread(self.hyde, user_query)
        retrieval_docs = await asyncio.to_thread(self._retriever.retrieveAnswers, query=hyde)        
        final_retrieval = await asyncio.to_thread(self.reranking, query=hyde, retrieved_docs=retrieval_docs)

        system_instruction = system_instruction = """
        Anda adalah seorang mentor rohani dan sahabat seiman yang hangat, bijak, namun TEGAS dalam memegang prinsip Firman Tuhan. Tugas Anda adalah merangkum jawaban untuk pengguna secara percakapan berdasarkan konteks ayat Alkitab yang disajikan sistem.

        ### ATURAN GAYA BAHASA & TONE (WAJIB DIPATUHI):
        1. BERBICARA SEPERTI MANUSIA: Gunakan bahasa sehari-hari yang mengalir, santai, dan penuh empati layaknya sahabat yang sedang ngobrol. 
        2. DILARANG META-TALK: Jangan menjelaskan proses sistem. HAPUS frasa kaku seperti "Berdasarkan ayat yang diberikan", "Menurut konteks di atas", atau "Saya menemukan ayat". Langsung masuk ke inti jawaban.
        3. ANTI SATU KATA (PENTING): Meskipun pertanyaan pengguna sangat singkat atau hanya mencari nama tokoh, DILARANG KERAS menjawab hanya dengan satu kata. Anda WAJIB merangkai jawaban dalam kalimat utuh.

        ### KERANGKA PENJAWABAN (PILIH SESUAI PERTANYAAN PENGGUNA):
        - JIKA MENCARI FAKTA (Siapa/Di mana/Kapan): Sebutkan nama/faktanya di kalimat pertama, LALU gunakan kalimat berikutnya untuk menceritakan sedikit latar belakang peristiwa tersebut berdasarkan konteks ayat yang disajikan.
        - JIKA MENCARI EKSPLANASI (Apa/Mengapa): Fokus jelaskan makna doktrin, latar belakang teologis, atau tujuan Allah berdasarkan ayat.
        - JIKA MEMINTA SOLUSI (Curhat/Masalah): Berikan teguran kasih atau langkah nyata yang harus diambil pengguna sesuai prinsip ayat. Jika konteks ayat TIDAK mendukung perbuatan pengguna (misal: kompromi dosa), tegur dengan tegas tanpa menghakimi.

        ### BATASAN MUTLAK FORMAT:
        1. PANJANG JAWABAN: WAJIB terdiri dari 3 SAMPAI 4 KALIMAT. Jangan kurang dan jangan lebih.
        2. REFERENSI NATURAL: Ekstrak intisari ayat referensi ke dalam penjelasan Anda dengan gaya bahasa santai (contoh: "Kisah Para Rasul mencatat bahwa...", "Amsal mengingatkan kita..."). DILARANG menyalin ulang teks ayat kata per kata.
        3. PENYARINGAN CERDAS: Abaikan konteks ayat yang diberikan sistem jika isinya tidak nyambung dengan masalah pengguna.
        """      
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("placeholder", "{chat_history}"),
            ("human", "Ayat Alkitab:\n{context}\n\nPERTANYAAN PENGGUNA: {question}\n\nKALIMAT PENGANTAR JAWABAN:")        
        ])

        chain = prompt | self.llm | StrOutputParser()
        memory_chain = RunnableWithMessageHistory(
            runnable = chain,
            get_session_history = self.get_session_history,
            input_messages_key = "question",
            history_messages_key = "chat_history"
        )

        ringkasan_llm = await memory_chain.ainvoke({"context": final_retrieval, 
                                      "question": user_query},
                                      config = {"configurable": {"session_id": session_id}}
                                      )
        ringkasan_llm = ringkasan_llm.strip()

        jawaban_final = ""
        for doc in final_retrieval:
            sumber = doc.metadata.get('sumber', '-')
            isi_ayat = doc.page_content
            jawaban_final += f"- **{sumber}**: \"{isi_ayat}\"\n"
            jawaban_final += "\n"

        if not ringkasan_llm:
            ringkasan_llm = "Prinsip Alkitab yang sesuai pertanyaan anda dapat dilihat pada referensi ayat berikut:"
        jawaban_final += f"{ringkasan_llm}\n\n"

        return jawaban_final

    # 1st MENU: Digital Bible reading   
    def getDaftarKitab(self, id:int) -> list[str]:
        return self._retriever.requestDaftarKitab(id)
    
    def getDaftarPasal(self, kitab:str) -> list[int]:
        return self._retriever.requestListPasal(kitab)
    
    def getDaftarAyat(self, kitab: str, pasal:int) -> list[int]:
        return self._retriever.requestListAyat(kitab, pasal)
    
    def getTeksAyat(self, kitab: str, pasal: int, ayat:int) -> str:
        return self._retriever.requestTeksAyat(kitab, pasal, ayat)