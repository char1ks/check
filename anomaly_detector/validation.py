from anomaly_detector.detector import AnomalyDetector

# Инициализация
detector = AnomalyDetector(data_root='datasets/carpet', device='cpu')

# Обучение
detector.train()

# Оценка
metrics = detector.evaluate()
print(f"AUC-ROC: {metrics['roc_auc']:.3f}")

# Визуализация t-SNE
detector.visualize_tsne().show()

# Предсказание
result = detector.predict('datasets/carpet/test/color/001.png')
print(f"Аномалия: {result['anomaly']}, Score: {result['score']:.3f}")
