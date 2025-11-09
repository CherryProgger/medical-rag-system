#!/usr/bin/env python3
"""
Простой локальный веб-интерфейс для вопросов и ответов.
Использует Gradio и текущую RAG систему.
"""

from __future__ import annotations

import re
import textwrap

import gradio as gr

from medical_rag.core.rag_system import MedicalRAGSystem
from medical_rag.models.config import Config


def _load_system() -> MedicalRAGSystem:
    """Инициализирует и возвращает RAG систему."""
    config = Config.load("config/default.json")
    rag_system = MedicalRAGSystem(config)
    rag_system.initialize()
    return rag_system


RAG_SYSTEM = _load_system()


def _clean_text(text: str) -> str:
    """Удаляет Markdown-разметку и приводит текст к аккуратному виду."""
    if not text:
        return ""
    cleaned = text.replace("**", "")
    cleaned = cleaned.replace('\\"', '"').replace("\\'", "'")
    cleaned = cleaned.replace("\\n", "\n")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    return cleaned.strip()


def _format_sources(documents, top_k: int) -> str:
    """Формирует аккуратный список источников."""
    if not documents:
        return "Источники не найдены."

    lines = []
    for idx, doc in enumerate(documents[: int(top_k)], 1):
        meta_parts = []
        if doc.metadata and doc.metadata.source_file:
            meta_parts.append(f"файл: {doc.metadata.source_file}")
        if doc.metadata and doc.metadata.section:
            meta_parts.append(f"раздел: {doc.metadata.section}")
        if doc.metadata and doc.metadata.topic:
            meta_parts.append(f"тема: {doc.metadata.topic}")

        snippet_source = doc.answer or doc.content
        snippet = _clean_text(snippet_source)
        snippet = textwrap.shorten(snippet, width=600, placeholder=" …")

        meta = ", ".join(meta_parts) if meta_parts else "метаданные отсутствуют"
        lines.append(f"- **Источник {idx}** ({meta})\n  > {snippet}")

    return "\n\n".join(lines)


def generate_answer(question: str, top_k: int) -> tuple[str, str]:
    """Возвращает ответ системы и информацию об источниках."""
    clean_question = question.strip()
    if not clean_question:
        return (
            "Введите вопрос, чтобы получить ответ.",
            "Источники появятся после ввода вопроса.",
        )

    response = RAG_SYSTEM.answer_question(clean_question, max_results=int(top_k))
    answer_text = _clean_text(response.answer) or "Ответ не получен."
    sources_text = _format_sources(response.relevant_documents, top_k)

    return answer_text, sources_text


with gr.Blocks(title="Медицинская RAG система") as demo:
    gr.Markdown(
        """
        # Медицинская RAG система (локальный режим)

        Введите медицинский вопрос, система подберёт релевантные фрагменты документации и сформирует ответ.
        """
    )

    with gr.Row():
        question_input = gr.Textbox(
            label="Вопрос",
            placeholder="Например: Что такое тромбофлебит поверхностных вен?",
            lines=3,
        )
        top_k_slider = gr.Slider(
            label="Количество источников",
            minimum=1,
            maximum=5,
            value=3,
            step=1,
        )

    answer_button = gr.Button("Получить ответ")
    answer_box = gr.Markdown(label="Ответ")
    sources_box = gr.Markdown(label="Источники")

    answer_button.click(
        generate_answer,
        inputs=[question_input, top_k_slider],
        outputs=[answer_box, sources_box],
    )

    gr.Markdown(
        """
        После запуска скрипта откройте браузер и перейдите по адресу: http://127.0.0.1:7860
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False)



