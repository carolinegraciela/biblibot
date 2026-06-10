#CLASS
from src.services.BibleRetriever import BibleRetriever 
from src.services.LLmService import LLmService

#COMMUNITY
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

#QUERY REWRITING
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

#CHAIN
from langchain_core.runnables import RunnablePassthrough

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

        self.system_instruction = """
            Anda adalah ahli teologi Kristen yang menjawab pertanyaan pengguna dengan menggunakan prinsip-prinsip teologis dari AYAT Alkitab referensi yang disediakan. 
            Tugas Anda adalah memberikan penjelasan teologis yang mendalam, kontekstual, dan sepenuhnya didasarkan pada prinsip-prinsip tersebut.

            Ketentuan Penting:
            1. HANYA buat SATU paragraf yang terdiri dari 4-5 kalimat. Jangan panjang-panjang.
            2. Jawablah menggunakan prinsip-prinsip dari AYAT ALKITAB yang diberikan.
            3. DILARANG KERAS MENGUTIP TEKS AYAT SECARA HARFIAH (literal). Tugas Anda adalah menjelaskan PRINSIP TEOLOGISnya dalam kata-kata Anda sendiri, bukan menyalin teks ayat tersebut.
            4. DILARANG KERAS mengarang fakta sejarah atau informasi fiktif.
            5. Jika pertanyaan membahas hal modern (misalnya rokok, internet) yang tidak disebutkan secara harfiah dalam Alkitab, jelaskan bagaimana PRINSIP MORAL/ETIKA dari ayat referensi dapat diterapkan dalam konteks modern tersebut.
            6. Dalam penjelasan Anda, jangan mengulangi kata-kata dalam ayat secara harfiah. Gunakan sinonim atau penjelasan konseptual.
            """
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_instruction),
            ("human", "AYAT ALKITAB REFERENSI (untuk prinsip):\n{context}\n\nPERTANYAAN PENGGUNA: {question}\n\nPENJELASAN PRINSIP TEOLOGIS (dalam 4-5 kalimat, dilarang mengutip teks):") 
        ])
        self.generation_chain = self.prompt | self.llm | StrOutputParser()


    # 2nd MENU: Question-Answering with Biblibot
    def query_rewriting(self, original_query):
        examples = [
        {
            "input": "saya sedang diajak teman saya untuk padel, tapi jamnya berkaitan dengan jam komsel. saya bimbang banget, enaknya gimana ya?",
            "output": "mengutamakan Kerajaan Allah. pentingnya kesetiaan pada pertemuan ibadah persekutuan. mendahulukan Tuhan di atas kesenangan duniawi dan hobi."
        },
        {
            "input": "saya mau ke bali sama temen2, apakah boleh pakai baju terbuka? adakah ayatnya?",
            "output": "Pentingnya menjaga kekudusan tubuh sebagai bait Roh Kudus. Panduan berdandan dengan pantas dan sopan bagi orang percaya. Tubuh harus digunakan untuk memuliakan Allah dalam kehidupan sehari-hari."
        },
        {
            "input": "boleh gak pacaran beda agama",
            "output": "Larangan menjadi pasangan yang tidak seimbang dengan orang yang tidak percaya. terang tidak dapat bersatu dengan gelap. jangan menjadi pasangan yang tidak seimbang dengan orang tidak percaya."
        },
        {
            "input": "ayat yang bahas tentang Daud mengalahkan Goliat",
            "output": "Daud mengalahkan Goliat orang Filistin menggunakan umban dan batu. maju demi nama TUHAN semesta alam."
        },
        {
            "input": "ayat tentang 9 buah roh",
            "output": "Sembilan buah Roh Kudus yang tertulis dalam surat Rasul Paulus kepada jemaat di Galatia. Karakteristik kehidupan orang percaya yang dipimpin oleh Roh Kudus seperti kasih, sukacita, dan damai sejahtera, kesabaran, kemurahan, kebaikan, kesetiaan, kelemahlembutan, penguasaan diri."
        },
        {
            "input": "ayat tentang zakheus",
            "output": "zakheus pemungut cukai yang pendek. zakheus memanjat pohon melihat yesus. yesus menginap di rumah zakheus."
        },
        {
            "input": "orang yang suka tatoan atau tindikan di seluruh badan itu sebenarnya dosa gak sih di mata Tuhan?",
            "output": "larangan membuat tanda rajah pada kulit. menjaga kekudusan tubuh fisik. memuliakan Allah dengan tubuh."
        },
        {
            "input": "gimana cara bikin nasi goreng jawa?",
            "output": "[OUT_OF_DOMAIN]"
        },
        {
            "input": "apa makna dari perumpamaan anak yang hilang?",
            "output": "[EXEGESIS_REQUEST]"
        },
        {
            "input": "arti perumpamaan ...?",
            "output": "[EXEGESIS_REQUEST]"
        },
        {
            "input": "ayat tentang perumpamaan anak yang hilang?",
            "output": "Ayat tentang seorang mempunyai dua anak laki-laki. Anak bungsu yang hilang. Seorang bapa punya anak sulung dan bungsu."
        },
        {
            "input": "halo biblibot!",
            "output": "[GREETING]"
        },
        {
            "input": "asdfghjkl",
            "output": "[GIBBERISH]"
        },
        {
            "input": "ayat untuk orang yang malas ke gereja",
            "output": "Nasihat untuk tidak menjauhkan diri dari pertemuan ibadah. Latihan badani ada batasnya, tapi roh tidak. Roh memang penurut tapi daging lemah."
        },
        {
            "input": "ayat untuk orang yang malas baca Alkitab",
            "output": "Peringatan bagi orang yang tegar tengkuk dan malas mendengarkan perkataan Tuhan. Pentingnya merenungkan Taurat Tuhan siang dan malam agar jalan hidup beruntung. Firman Tuhan sebagai pelita bagi kaki dan terang bagi jalan orang percaya."
        }        
        ]

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}"),
        ])

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt = example_prompt,
            examples = examples,
        )

        system_instruction =  system_instruction =  """### INSTRUKSI
        Anda adalah sistem klasifikasi dan penulis ulang kueri (Query Rewriter) untuk chatbot pencarian ayat Alkitab.
        Tugas Anda adalah menganalisis input pengguna dan WAJIB memberikan output berdasarkan hierarki 5 kondisi berikut (Cek dari No. 1 hingga 5):

        1. PERMINTAAN TAFSIR (PRIORITAS UTAMA): HANYA gunakan tag ini jika pengguna secara eksplisit menanyakan makna, arti, maksud, tafsiran, atau penjelasan teologis (misal menggunakan kata kunci: "apa arti", "maksud dari", "jelaskan makna", "artikan kisah ... di kitab ...", "arti dari", "tafsirkan"). Balas HANYA dengan: [EXEGESIS_REQUEST]
           *PANTANGAN KERAS*: Jika pengguna mencari ayat, lokasi pasal, atau menyebutkan suatu peristiwa, mukjizat, dan kisah Alkitab (misal: "cerita yesus memecah2kan 5 roti", "mukjizat air jadi anggur", "ayat tentang kisah Zakheus"), JANGAN PERNAH gunakan tag ini. Kueri yang menceritakan plot/peristiwa untuk dicari ayatnya adalah KUERI VALID, langsung lanjutkan ke No. 5.
        2. GIBBERISH: Jika input adalah ketikan acak, tidak bermakna, atau hanya simbol (misal: "asdfg", "husdbuwiefue"), balas HANYA dengan: [GIBBERISH]
        3. SAPAAN: Jika input murni sapaan, balas HANYA dengan: [GREETING]
        4. LUAR TOPIK: Jika input MURNI tentang hal duniawi (resep masakan, cuaca, tutorial IT, travel) TANPA meminta pandangan rohani, balas HANYA dengan: [OUT_OF_DOMAIN]. 
        KECUALI: Jika pengguna menanyakan pandangan Alkitab/ayat terkait aktivitas sehari-hari tersebut (misal: gaya berpakaian, hiburan, hobi), JANGAN gunakan tag ini, melainkan lanjutkan ke No. 5 (KUERI VALID).        
        5. KUERI VALID (PENCARIAN TOPIK): Jika pengguna membagikan masalah hidup (curhat) atau mencari ayat berdasarkan topik tertentu (BUKAN minta tafsir pasal), ubah kueri menjadi 3 KALIMAT DEKLARATIF TEOLOGIS untuk pencarian semantic search.
        
        ### CONTOH OUTPUT UNTUK KONDISI 5 (KUERI VALID):
        - Input: "pusing nih"
        Output: Ayat Alkitab tentang ketenangan pikiran dan jiwa dalam menghadapi kesesakan. Janji Tuhan bagi manusia yang sedang memikul beban berat dan stres. Penghiburan dan kekuatan dari Allah saat menghadapi badai hidup.
        - Input: "apa perbedaan cinta dan nafsu"
        Output: Definisi kasih sejati yang tulus, sabar, dan tidak mementingkan diri sendiri menurut Alkitab. Peringatan Alkitab mengenai bahaya hawa nafsu kedagingan dan keinginan duniawi. Perbandingan antara kasih agape yang kudus dengan perbuatan daging dalam Galatia 5.

        ### ATURAN MUTLAK
        Aturan Penting:
        1. JANGAN PERNAH MENJAWAB PERTANYAAN PENGGUNA ATAU MENAMPILKAN AYAT UTUH DI SINI. Tugas Anda di modul ini HANYA melakukan klasifikasi tag atau melakukan perluasan kueri (query expansion).
        2. Jika pengguna menyebutkan aktivitas modern, hobi, atau istilah gaul (seperti 'padel', 'komsel', 'nongkrong', 'pacaran'), terjemahkan situasi tersebut ke dalam prinsip teologis dasarnya (misalnya: 'komsel' = persekutuan ibadah, 'padel/nongkrong' = waktu luang/kesenangan hidup) ke dalam 3 kalimat deklaratif tersebut.
        3. Output HARUS langsung berupa TAG KODE (seperti [GIBBERISH]) atau 3 KALIMAT DEKLARATIF TEOLOGIS. 
        4. DILARANG MEMBERIKAN PENJELASAN, BASA-BASI, TANDA KUTIP EXTRA, ATAU MENYAPA BALIK PENGGUNA.
        5. SETIAP KALIMAT HARUS DIAKHIRI DENGAN TITIK (.).
        6. DILARANG MENGULANG-NGULANG KALIMAT YANG SAMA.
        """

        final_prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            few_shot_prompt,                  
            ("human", "{original_query}")    
        ])

        model_with_stop = self.llm.bind(stop=["User:", "###", "Human:"])
        
        query_rewriter = final_prompt | model_with_stop | StrOutputParser()
        response = query_rewriter.invoke({"original_query": original_query})

        clean_response = response.strip()
        return clean_response

    def hyde(self, kueri):
        system_template = """### INSTRUKSI
        Anda adalah mesin penasihat kristiani yang menjawab pertanyaan dan permintaan pengguna BERDASARKAN Alkitab Terjemahan Baru (TB).
        Input yang Anda terima adalah sekumpulan kata kunci atau ringkasan teologis.

        Tugas Anda: Ubah input tersebut menjadi 3-4 KALIMAT PADAT yang GAYA BAHASA, KOSAKATA, dan STRUKTUR KALIMATNYA MENIRU ayat Alkitab asli.
        Teks ini HARUS "terdengar dan terasa" seperti kutipan langsung dari Alkitab TB, dengan MENGGUNAKAN prinsip Alkitab dan kebenaran kristen.

        ### ATURAN MUTLAK:
        1. Jawab HANYA dengan teks hipotetis tersebut. Tanpa awalan, tanpa akhiran, tanpa basa-basi.
        2. DILARANG KERAS menggunakan kata-kata modern atau non-Alkitabiah (seperti: "hobi", "duniawi", "gereja modern", "komsel", "teologi", "relevan").
        3. DILARANG KERAS menggunakan gaya bahasa khotbah atau renungan harian (JANGAN gunakan: "Ingatlah saudara-saudara", "Mari kita", "Tuhan ingin kita").
        4. JANGAN tambahkan frasa pengantar seperti "Berfirmanlah Tuhan:" atau "Yesus berkata:", KECUALI inputnya secara eksplisit mencari kisah tokoh. Langsung saja tulis inti pengajarannya.
        5. GUNAKAN KOSAKATA ALKITAB TB (contoh: "Carilah dahulu", "Sebab sesungguhnya", "Kasih karunia", "Kerajaan Allah", "pertemuan ibadah", "Bait Suci").
        6. JANGAN PERNAH MENGARANG ATAU MENAMBAHKAN DAFTAR ISTILAH ALKITAB JIKA ANDA TIDAK TAHU VERBATIM ASLINYA. Jika input menyebutkan sebagian daftar (seperti buah roh, karunia, hukum), cukup sebutkan bagian yang Anda ketahui pasti dari Alkitab TB atau gunakan kalimat generalisasi Alkitabiah tanpa mengarang kata baru.

        ### CONTOH 1 (Topik Moral/Pergumulan):
        Input: mengasihi musuh, amarah, pembalasan adalah hak Tuhan
        Dokumen Hipotetis: Kasihilah musuhmu dan berdoalah bagi mereka yang menganiaya kamu. Janganlah kamu sendiri menuntut balas, melainkan berilah tempat kepada murka Allah, sebab ada tertulis: Pembalasan itu adalah hak-Ku. Hendaklah segala kepahitan, kegeraman, dan kemarahan dibuang dari antara kamu.

        ### CONTOH 2 (Topik Prioritas - Berhubungan dengan aktivitas modern):
        Input: mengutamakan Kerajaan Allah, pentingnya kesetiaan pada pertemuan ibadah persekutuan, mendahulukan Tuhan di atas kesenangan
        Dokumen Hipotetis: Carilah dahulu Kerajaan Allah dan kebenarannya, maka semuanya itu akan ditambahkan kepadamu. Janganlah kita menjauhkan diri dari pertemuan-pertemuan ibadah kita, seperti dibiasakan oleh beberapa orang, tetapi marilah kita saling menasihati. Sebab barangsiapa yang menjadi sahabat dunia, ia menjadikan dirinya musuh Allah.

        ### CONTOH 3 (Topik Kekudusan Tubuh):
        Input: menjaga kekudusan tubuh, berdandan sopan, bait Roh Kudus
        Dokumen Hipotetis: Muliakanlah Allah dengan tubuhmu, sebab tubuhmu adalah bait Roh Kudus yang diam di dalam kamu. Demikian juga hendaknya perempuan berdandan dengan pantas, dengan sopan dan sederhana.

        ### CONTOH 4 (Topik Kisah Tokoh / Narasi Sejarah):
        Input: kisah daniel di goa singa, daniel tidak takut pada raja babilon, daniel percaya pada Tuhan yang selalu menyertai
        Dokumen Hipotetis: Maka dilemparkanlah Daniel ke dalam gua singa atas perintah raja, namun ia tidak gentar terhadap ancaman itu. Sebab ia senantiasa menaruh percaya kepada Allahnya dan tidak berhenti bersujud serta berdoa dengan setia. Lalu Allah mengutus malaikat-Nya untuk mengatupkan mulut singa-singa itu, sehingga tidak ada bahaya menimpa dia karena imannya. Sesungguhnya, Tuhan menyertai orang yang benar dan melepaskan hamba-Nya yang berharap kepada-Nya.

        ### CONTOH 5 (Topik Kisah Yesus / Perumpamaan):
        Input: Ayat tentang seorang mempunyai dua anak laki-laki. Anak bungsu yang hilang. Seorang bapa punya anak sulung dan bungsu.
        Dokumen Hipotetis: Ada seorang mempunyai dua anak laki-laki, lalu kata yang bungsu kepada ayahnya: Bapa, berikanlah kepadaku bagian harta milik kita yang menjadi hakku. Setelah anak itu pergi ke negeri yang jauh dan memboroskan harta miliknya, ia menyesal lalu kembali kepada bapanya. Ketika ia masih jauh, ayahnya telah melihatnya, lalu tergeraklah hatinya oleh belas kasihan. Bapa itu berlari merangkul anak yang telah hilang dan didapat kembali itu, serta menyambutnya dengan sukacita.
        
        ### CONTOH 6 (Topik Daftar / Karakteristik Rohani):
        Input: Sembilan buah Roh Kudus yang tertulis dalam surat Rasul Paulus kepada jemaat di Galatia. Karakteristik kehidupan orang percaya yang dipimpin oleh Roh Kudus seperti kasih, sukacita, dan damai sejahtera.
        Dokumen Hipotetis: Jikalau kita hidup oleh Roh, baiklah hidup kita juga dipimpin oleh Roh. Sebab buah Roh ialah kasih, sukacita, damai sejahtera, .......
        """

        hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "Input:\n{query}\n\nDokumen Hipotetis:")    
        ])

        hyde_chain = hyde_prompt | self.llm | StrOutputParser()
        response = hyde_chain.invoke({"query": kueri}).strip()
        print(f"Dokumen hipotesis: {response.strip()}")

        return response.strip()

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

    def generate_response(self, user_query: str):
        rewritten_query = self.query_rewriting(user_query)
        print(f"Query hasil rewriting: {rewritten_query}")

        if "[GIBBERISH]" in rewritten_query:
            return "Maaf, saya tidak mengerti pesan yang kamu kirimkan. Bisa tolong ketik ulang dengan lebih jelas? 😊"
        elif "[GREETING]" in rewritten_query:
            return "Halo, shalom! Saya Biblibot, asisten pencarian ayat Alkitab secara kontekstual. Ada topik pergumulan atau kisah Alkitab yang ingin kamu cari hari ini? 🙏"
        elif "[OUT_OF_DOMAIN]" in rewritten_query:
            return (
                "Maaf, saya hanya didesain untuk membantu mencari ayat Alkitab berdasarkan topik atau pergumulan rohani. "
                "Saya tidak bisa menjawab pertanyaan umum di luar Alkitab seperti resep masakan, tips teknologi, atau pariwisata. "
                "Silakan ketikkan topik rohani atau masalah yang sedang kamu hadapi ya! ✨"
            )
        elif "[EXEGESIS_REQUEST]" in rewritten_query:
            return (
                "Maaf, saya dirancang khusus sebagai asisten mesin pencari ayat secara kontekstual, "
                "bukan untuk menafsirkan, menjelaskan makna teologis, atau memberikan arti pada ayat tertentu.\n\n"
                "Untuk mendapatkan penjelasan yang mendalam dan akurat mengenai tafsiran tersebut, "
                "saya sangat menyarankan untuk berdiskusi dengan pendeta, pembimbing rohani, atau membaca buku tafsir Alkitab. 🙏\n\n"
                "Namun, jika kamu ingin mencari ayat-ayat lain dengan *tema serupa*, silakan ketikkan topik pergumulan atau kata kuncinya!"
            )
        elif "[KUERI TIDAK VALID" in rewritten_query or not rewritten_query.strip():
            return (
                "Maaf, kueri yang kamu masukkan tidak dapat saya proses. "
                "Pastikan kamu memasukkan topik kehidupan, cerita tokoh Alkitab, atau situasi yang sedang kamu alami "
                "agar saya bisa mencarikan ayat Alkitab yang tepat untukmu. Silakan coba lagi! 😊"
            )
        
        hasil_hyde = self.hyde(rewritten_query)   
        retrieval_docs = self._retriever.retrieveAnswers(hasil_hyde)
        final_retrieval = self.reranking(hasil_hyde, retrieval_docs)

        system_instruction = """### INSTRUKSI
        Anda adalah ahli teologi Kristen yang tajam, analitis, dan menjawab secara TEGAS serta EKSKLUSIF berbasis KEBENARAN Alkitab.
        Tugas Anda adalah menjawab pergumulan pengguna BERDASARKAN ayat Alkitab yang disediakan di konteks.

        ### ATURAN MUTLAK PENYARINGAN KONTEKS (DYNAMIC STATE-AWARENESS):
        1. ANALISIS SITUASI PENGGUNA: Deduksi secara logis fase hidup, status, atau situasi spesifik yang sedang dialami pengguna berdasarkan pertanyaannya (Contoh: apakah ia sedang lajang, berduka, bekerja, sakit, atau sekadar bertanya prinsip umum).
        2. BUANG KONTEKS SAMPAH: Sistem pencari mungkin secara tidak sengaja menyertakan ayat yang mirip secara kata, namun salah sasaran secara situasi. Anda WAJIB MENGABAIKAN ayat dalam konteks yang tidak sesuai dengan situasi spesifik pengguna. 
           - Contoh 1: Jika pengguna bertanya tentang PDKT/Pacaran, abaikan ayat tentang perceraian suami-istri.
           - Contoh 2: Jika pengguna bertanya masalah bos di tempat kerja, abaikan ayat tentang konflik orang tua dan anak.
        
        ### ATURAN MUTLAK KETEGASAN & GAYA BAHASA:
        3. ALIGNMENT OVERRIDE: JANGAN mencampurkan jawaban dengan pandangan sekuler, saran psikologi populer, kompromi asmara, atau toleransi modern. 
        4. KETEGASAN: Fokus HANYA pada ayat, pasal, dan kitab yang paling relevan dengan situasi pengguna. Jawab dengan tegas, lugas, dan tegak lurus pada prinsip firman tanpa kompromi.
        5. STRUKTUR: Jawab dalam 1 paragraf padat (4-5 kalimat). Sampaikan dengan penuh kasih namun objektif. DILARANG mengutip ulang isi ayat secara harfiah (tugasmu HANYA memberi penjelasan prinsipnya).
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", "Ayat Alkitab:\n{context}\n\nPERTANYAAN PENGGUNA: {question}\n\nKALIMAT PENGANTAR JAWABAN:")        
        ])

        chain = prompt | self.llm | StrOutputParser()
        ringkasan_llm = chain.invoke({"context": final_retrieval[0], "question": user_query}).strip()

        if not ringkasan_llm:
            ringkasan_llm = "Prinsip Alkitab yang sesuai pertanyaan anda dapat dilihat pada referensi ayat berikut:"
        jawaban_final = f"{ringkasan_llm}\n\n"
        jawaban_final += "Berikut adalah referensi ayat yang relevan:\n"

        for doc in final_retrieval:
            sumber = doc.metadata.get('sumber', '-')
            isi_ayat = doc.page_content
            jawaban_final += f"- **{sumber}**: \"{isi_ayat}\"\n"
            jawaban_final += "\n"

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