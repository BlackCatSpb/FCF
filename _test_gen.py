import sys, os, json
sys.path.insert(0, '.')
from fcf.primordial_layer import PrimordialLayer
from fcf.tokenizer_utils import load_tokenizer
from fcf.utils import load_primordial_layer

layer = load_primordial_layer('checkpoints/language/step_023000', PrimordialLayer)
t = load_tokenizer('tokenizer.json')

prompts = [
    "История это наука которая изучает",
    "Математика помогает человечеству",
    "Природа Земли удивительна потому что",
    "Компьютеры обрабатывают данные с помощью",
    "Философия задает вечные вопросы о",
    "Сложный вопрос: объясни взаимосвязь между квантовой физикой и сознанием",
    "Что такое жизнь с точки зрения биологии",
    "Искусство это способ выражения",
    "Экономика изучает как общество",
    "Медицина спасает жизни используя",
]

print("=" * 60)
print("FCF — ТЕСТ ГЕНЕРАЦИИ (чекпоинт step_023000)")
print(f"Токенизатор: {t.get_vocab_size()} слов")
print(f"Слепков: {len(layer.state_storage)}")
print(f"SRG avg: {layer.meta.average_confidence():.3f}")
print("=" * 60)

for prompt in prompts:
    result = layer.process_query(prompt, t, max_new_tokens=60, temperature=0.7)
    response = result['response']
    conf = result['confidence']
    sim = result['similarity']
    eth = result['ethics_score']
    
    print()
    print(f"Q: {prompt}")
    print(f"A: {response}")
    print(f"   conf={conf:.3f} sim={sim:.3f} ethics={eth:.2f}")

print()
print("=" * 60)
print("Тест завершён.")
input("Нажмите Enter для выхода...")
