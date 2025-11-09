import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from medical_rag.models.config import Config
from medical_rag.core.rag_system import MedicalRAGSystem
from medical_rag.models.response import ResponseMetadata, Response
from medical_rag.models.query import Query


CORPUS_PATH = Path("data/corpus.jsonl")


@pytest.mark.skipif(not CORPUS_PATH.exists(), reason="Корпус data/corpus.jsonl не найден")
class TestRAGQueryResponses:
    """Интеграционный тест запросов к реальному корпусу."""

    KEYWORDS: List[str] = [
        "тромбофлебит",
        "поверхностн",
        "варикоз",
        "веноз",
        "вен",
        "варикотромбофлебит",
        "код",
        "мкб",
        "стад",
        "риск",
        "тромбоэмбол",
        "тгв",
        "трофическ",
        "язв",
        "хроническ",
        "недостаточн",
        "аневризм",
        "аорт",
        "инсульт",
        "онмк",
        "антикоагулянт",
        "посттромботическ",
    ]

    TEST_CASES: List[tuple[str, str, str]] = [
        (
            "Что такое тромбофлебит поверхностных вен?",
            "тромбофлебит поверхностных вен – патологическое состояние",
            "Флебит",
        ),
        (
            "Что такое варикотромбофлебит?",
            "варикотромбофлебит – тромбофлебит (тромбоз) варикозно измененных поверхностных вен",
            "Флебит",
        ),
        (
            "Каков код тромбофлебита поверхностных сосудов нижних конечностей по МКБ-10?",
            "I80.0 – флебит и тромбофлебит поверхностных сосудов нижних конечностей",
            "Флебит",
        ),
        (
            "Какова стадийность тромбофлебита поверхностных вен?",
            "острый (0–7 дней), стихающий (1–3 недели), стихший (более 3 недель)",
            "Флебит",
        ),
        (
            "Что означает высокий риск перехода тромба на глубокие вены?",
            "проксимальная граница тромба ≤3 см от сафено-феморального или сафено-поплитеального соустья",
            "Флебит",
        ),
        (
            "Что такое венозные тромбоэмболические осложнения?",
            "венозные тромбоэмболические осложнения – собирательное понятие",
            "Диагностика, лечение и профилактика венозных тромбоэмболических осложнений",
        ),
        (
            "Что такое тромбоз глубоких вен?",
            "тромбоз глубоких вен – наличие тромба в глубокой вене, который может вызвать ее окклюзию",
            "Диагностика, лечение и профилактика венозных тромбоэмболических осложнений",
        ),
        (
            "Что такое тромбоз поверхностных вен?",
            "тромбоз поверхностных вен (тромбофлебит) – наличие тромба в поверхностной вене",
            "Диагностика, лечение и профилактика венозных тромбоэмболических осложнений",
        ),
        (
            "Как обозначается тромбоэмболия легочной артерии по МКБ-10?",
            "I26 – тромбоэмболия легочной артерии",
            "Диагностика, лечение и профилактика венозных тромбоэмболических осложнений",
        ),
        (
            "Что такое венозные трофические язвы?",
            "венозная трофическая язва – дефект кожи и мягких тканей",
            "Варикозное расширение вен",
        ),
        (
            "Что такое варикозная болезнь нижних конечностей?",
            "варикозная болезнь нижних конечностей – хроническое заболевание с первичным варикозным расширением подкожных вен",
            "Варикозное расширение вен",
        ),
        (
            "Что такое хроническая венозная недостаточность?",
            "хроническая венозная недостаточность – патологическое состояние, обусловленное нарушением венозного оттока",
            "Варикозное расширение вен",
        ),
        (
            "Что такое посттромботическая болезнь?",
            "посттромботическая болезнь – хроническое заболевание, обусловленное органическим поражением глубоких вен",
            "Диагностика, лечение и профилактика венозных тромбоэмболических осложнений",
        ),
        (
            "Что такое хронические заболевания вен?",
            "хронические заболевания вен – все морфологические и функциональные нарушения венозной системы",
            "Варикозное расширение вен",
        ),
        (
            "Что такое антикоагулянты?",
            "антикоагулянты – препараты, объединяющие средства для парентерального и перорального введения",
            "Диагностика, лечение и профилактика венозных тромбоэмболических осложнений",
        ),
        (
            "Что такое аортальная недостаточность?",
            "аортальная недостаточность – обратный ток крови из аорты в левый желудочек в фазу диастолы",
            "Аневризмы грудной и торакоабдоминальной аорты",
        ),
        (
            "Что такое острое нарушение мозгового кровообращения (ОНМК)?",
            "острое нарушение мозгового кровообращения – клинический синдром, характеризующийся внезапным развитием очаговой неврологической симптоматики",
            "Ишемический инсульт и транзиторная ишемическая атака",
        ),
    ]
    @pytest.fixture(scope="class", autouse=True)
    def setup_rag(self, request, tmp_path_factory):
        """Инициализирует RAG систему со стабами эмбеддингов и генерации."""

        from medical_rag.core import embedding_service as embedding_module
        from medical_rag.core import rag_system as rag_module
        from medical_rag.core import retrieval_service as retrieval_module
        from medical_rag.core import generation_service as generation_module
        mp = pytest.MonkeyPatch()
        request.addfinalizer(mp.undo)

        class StubEmbeddingService(embedding_module.EmbeddingService):
            """Эмуляция сервисa эмбеддингов через подсчёт ключевых слов."""

            def _initialize_model(self):
                self.model = "stub"
                self._vectorizer: Optional[TfidfVectorizer] = None

            def create_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:  # noqa: D401
                if self._vectorizer is None:
                    return np.zeros((len(texts), len(TestRAGQueryResponses.KEYWORDS)), dtype=np.float32)
                embeddings = self._vectorizer.transform(texts)
                return embeddings.astype(np.float32).toarray()

            def create_embedding(self, text: str) -> np.ndarray:
                return self.create_embeddings([text])[0]

            def get_embedding_dimension(self) -> int:
                if self._vectorizer is None:
                    return len(TestRAGQueryResponses.KEYWORDS)
                return len(self._vectorizer.get_feature_names_out())

            def create_document_embeddings(self, documents):  # noqa: D401
                if not documents:
                    return documents
                texts = [doc.content for doc in documents]
                if self._vectorizer is None:
                    self._vectorizer = TfidfVectorizer(
                        max_features=4096,
                        ngram_range=(1, 2),
                        analyzer="word",
                        lowercase=True,
                    )
                    embeddings = self._vectorizer.fit_transform(texts)
                else:
                    embeddings = self._vectorizer.transform(texts)
                array = embeddings.astype(np.float32).toarray()
                for doc, embedding in zip(documents, array):
                    doc.embedding = embedding.tolist()
                return documents

        class StubGenerationService(generation_module.GenerationService):
            """Генерация ответа за счёт прямого цитирования лучшего документа."""

            def _initialize_model(self):
                self.model = None
                self.tokenizer = None
                self.generator = None

            def generate_response(self, query, relevant_documents, processing_time, num_documents_searched=0):  # noqa: D401
                answer = "Информация не найдена."
                sources = []
                if relevant_documents:
                    best_doc = max(relevant_documents, key=lambda doc: doc.similarity_score or 0)
                    answer = best_doc.answer
                    sources = [best_doc.get_source_info()]

                metadata = ResponseMetadata(
                    processing_time=processing_time,
                    num_documents_searched=num_documents_searched,
                    num_documents_found=len(relevant_documents),
                    best_similarity_score=max(
                        [doc.similarity_score or 0 for doc in relevant_documents],
                        default=0.0,
                    ),
                    confidence_level="high" if relevant_documents else "low",
                    query_type=query.get_query_type(),
                    is_medical_query=query.is_medical_query(),
                )
                return Response(
                    query=query.text,
                    answer=answer,
                    relevant_documents=relevant_documents,
                    metadata=metadata,
                    sources=sources,
                    warnings=[],
                )

        mp.setattr(embedding_module, "EmbeddingService", StubEmbeddingService)
        mp.setattr(generation_module, "GenerationService", StubGenerationService)
        mp.setattr(rag_module, "EmbeddingService", StubEmbeddingService)
        mp.setattr(rag_module, "GenerationService", StubGenerationService)
        mp.setattr(retrieval_module, "EmbeddingService", StubEmbeddingService)

        tmp_dir = tmp_path_factory.mktemp("rag_cache")
        config = Config.load("config/development.json")
        config.data.use_corpus = True
        config.data.corpus_path = str(CORPUS_PATH)
        config.cache_embeddings = False  # исключаем sqlite во время тестов
        config.cache_dir = tmp_dir
        config.embeddings_cache_path = tmp_dir / "embeddings.sqlite3"
        config.vector_index_path = tmp_dir / "vector_index.faiss"
        config.model.embedding_model = "stub"
        config.model.generation_model = "stub"

        rag = MedicalRAGSystem(config)
        rag.initialize()

        request.cls.rag_system = rag
        corpus_entries = rag.data_processor.corpus_entries or rag.data_processor.load_corpus_entries()
        snippet_to_doc: Dict[str, str] = {}
        for _, snippet, _ in TestRAGQueryResponses.TEST_CASES:
            norm_snippet = snippet.lower().replace("–", "-").replace("—", "-")
            candidate_id: Optional[str] = None
            # Прямая проверка подстроки
            for entry in corpus_entries:
                text_lower = entry.get("text", "").lower().replace("–", "-").replace("—", "-")
                if norm_snippet in text_lower:
                    candidate_id = entry["id"]
                    break
            if candidate_id is None:
                tokens = [t for t in re.split(r"[\\s,.;:()«»\"\\-]+", norm_snippet) if len(t) >= 2]
                if tokens:
                    for entry in corpus_entries:
                        text_lower = entry.get("text", "").lower()
                        if all(token in text_lower for token in tokens[:3]):
                            candidate_id = entry["id"]
                            break
            if candidate_id:
                snippet_to_doc.setdefault(snippet, candidate_id)

        request.cls.snippet_to_doc_id = snippet_to_doc

    @pytest.mark.parametrize(
        "question, expected_snippet, expected_source",
        TEST_CASES,
    )
    def test_query_answer_contains_expected_content(
        self,
        question: str,
        expected_snippet: str,
        expected_source: str,
    ):
        target_doc_id = self.snippet_to_doc_id.get(expected_snippet)
        assert target_doc_id is not None, f"Не найден документ для фрагмента: {expected_snippet}"

        query = Query(text=question, max_results=8, similarity_threshold=0.0)
        retrieved_docs = self.rag_system.retrieval_service.search(query)
        assert retrieved_docs, "Индекс не вернул документов"

        retrieved_ids = [doc.id for doc in retrieved_docs]
        assert target_doc_id in retrieved_ids, f"Ожидали документ {target_doc_id}, но получили {retrieved_ids}"

        target_doc = self.rag_system.retrieval_service.get_document_by_id(target_doc_id)
        assert target_doc is not None, f"Документ {target_doc_id} не найден в индексе"
        assert expected_snippet.lower() in target_doc.content.lower()
        assert expected_source.lower() in target_doc.metadata.source_file.lower()


