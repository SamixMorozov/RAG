
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Функция для чтения CSV с автоопределением кодировки
def read_csv_autoencoding(path):
    for enc in ['utf-8', 'cp1251', 'latin1']:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Не удалось прочитать файл {path}")

# Загрузка данных
put_data = read_csv_autoencoding('input_put_data.csv')
rosberta_1 = read_csv_autoencoding('gemma3_27b_ai-forever_ru-en-RoSBERTa_results.csv')
rosberta_2 = read_csv_autoencoding('hf_co_t_tech_T_pro_it_1_0_Q4_K_M_GGUF_Q4_K_M_ai_forever_ru_en_RoSBERTa.csv')

# Обработка пропусков
rosberta_2['Answer'] = rosberta_2['Answer'].fillna('')
rosberta_2['Perfect_Answer'] = rosberta_2['Perfect_Answer'].fillna('')
rosberta_1['Answer'] = rosberta_1['Answer'].fillna('')
put_data['B.1'] = put_data['B.1'].fillna('')

# Загрузка модели эмбеддингов
model = SentenceTransformer('sberbank-ai/ruBert-base')

# Сходство rosberta_2: Answer vs Perfect_Answer
emb_2_ans = model.encode(rosberta_2['Answer'].tolist(), convert_to_tensor=True)
emb_2_perf = model.encode(rosberta_2['Perfect_Answer'].tolist(), convert_to_tensor=True)
rosberta_2['Cosine_Similarity'] = cosine_similarity(emb_2_ans, emb_2_perf).diagonal()

# Сходство rosberta_1 vs PUT: Answer vs B.1
emb_1_ans = model.encode(rosberta_1['Answer'].tolist(), convert_to_tensor=True)
emb_1_perf = model.encode(put_data['B.1'].tolist(), convert_to_tensor=True)
rosberta_1['Cosine_Similarity'] = cosine_similarity(emb_1_ans, emb_1_perf).diagonal()

# Сохраняем результаты
rosberta_1.to_csv('rosberta_1_with_similarity.csv', index=False)
rosberta_2.to_csv('rosberta_2_with_similarity.csv', index=False)

print("✅ Готово. Файлы сохранены.")
