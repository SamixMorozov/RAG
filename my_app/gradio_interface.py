import time
import logging
import gradio as gr

logger = logging.getLogger(__name__)

def create_gradio_interface(search_service):
    def gr_interface(question):
        start_time = time.time()
        try:
            answer = search_service.answer_question(question)
            return answer
        except Exception as e:
            logger.error(f"Ошибка в gr_interface: {e}", exc_info=True)
            return "Произошла ошибка во время обработки."
        finally:
            end_time = time.time()
            logger.info(f"Время обработки запроса: {end_time - start_time} секунд")

    with gr.Blocks() as interface:
        gr.Markdown("## Система Retrieval-Augmented Generation")

        with gr.Row():
            question_input = gr.Textbox(
                label="Введите ваш вопрос по содержимому файла",
                placeholder="Введите вопрос здесь..."
            )

        output_text = gr.Textbox(
            label="Ответ",
            placeholder="Здесь будет отображён ответ на ваш вопрос."
        )

        submit_button = gr.Button("Получить ответ")

        submit_button.click(
            fn=gr_interface,
            inputs=question_input,
            outputs=output_text,
        )

    interface.queue()
    return interface
