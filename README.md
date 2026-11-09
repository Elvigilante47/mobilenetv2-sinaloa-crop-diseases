# Diagnóstico de Enfermedades Foliares en Hortalizas de Sinaloa mediante MobileNetV2

Este repositorio contiene el código fuente y la configuración experimental para el entrenamiento y evaluación de modelos de aprendizaje profundo enfocados en el diagnóstico de enfermedades agrícolas. El proyecto utiliza la arquitectura MobileNetV2 para clasificar 5 patologías comunes en cultivos de la región de Sinaloa, México.

## Descripción del Proyecto
El sistema está diseñado para operar en entornos con recursos limitados (dispositivos móviles), optimizando el entrenamiento en estaciones de trabajo mediante la carga del conjunto de datos directamente en la memoria RAM (`/dev/shm`) para eliminar cuellos de botella de I/O en discos mecánicos. Además, implementa un pipeline de evaluación de técnicas de aumento de datos (recorte aleatorio, volteo horizontal, fluctuación de color y desenfoque gaussiano).

Las 5 clases evaluadas son:
1. `Cucumber_gummy_stem_blight`
2. `Pumpkin_leaf_blight`
3. `Basil_downy_mildew`
4. `Melon_gummy_stem_blight`
5. `Chiltepin_chili_early_Bacterial_blight`

## Requisitos del Sistema
El código fue desarrollado y probado en un entorno Linux (Ubuntu) utilizando Anaconda. Las dependencias principales incluyen:

* Python 3.10
* PyTorch 2.6.0 (con soporte para CUDA 12.4)
* Torchvision
* Scikit-learn
* Matplotlib & Seaborn
* Pillow
* Tqdm

## Estructura del Repositorio
* `configuracion.py`: Archivo central que controla hiperparámetros (épocas, batch size, learning rate), rutas del dataset y activación de métodos de aumento de datos.
* `manifiest.py`: Script para generar el manifiesto JSON que divide de forma reproducible el dataset (80% entrenamiento / 20% validación).
* `ram_dataset_loader_v2.py`: Herramienta para transferir las imágenes redimensionadas (224x224) a la memoria RAM antes del entrenamiento.
* `dataset.py`: Clase personalizada de PyTorch (`HortiJsonDataset`) que lee el manifiesto y aplica las transformaciones dinámicas.
* `entrenamiento.py`: Script principal que ejecuta el entrenamiento utilizando precisión mixta (AMP), optimizador AdamW y guarda el mejor checkpoint basado en la pérdida de validación.
* `evaluacion.py`: Genera automáticamente las matrices de confusión y reportes de métricas (F1-score, Precision, Recall) al finalizar el entrenamiento.
* `ejecutar_experimentos_paralelos.py`: Orquestador para ejecutar múltiples configuraciones de aumento de datos utilizando múltiples GPUs de forma paralela.

## Instrucciones de Uso

### 1. Preparación de los datos
Asegúrate de tener el conjunto de datos redimensionado a 224x224 y ejecuta el generador del manifiesto:
```bash
python manifiest.py
```

Posteriormente, carga los datos a la memoria RAM para acelerar la lectura:
Bash
python ram_dataset_loader_v2.py
### 2. Entrenamiento
Para iniciar una corrida individual (la configuración activa dependerá de las variables en configuracion.py), ejecuta:
Bash
python entrenamiento.py
Si deseas ejecutar la batería completa de experimentos comparando métodos de aumento de datos en paralelo:
Bash
python ejecutar_experimentos_paralelos.py
### 3. Resultados
Al finalizar, los pesos del mejor modelo se guardarán en la carpeta Modelos/. Las métricas consolidadas (.csv), la configuración utilizada (.json) y las matrices de confusión visuales (.png) se exportarán automáticamente a la carpeta de reportes especificada.

### Citación
Si este código o el conjunto de datos resultan útiles para tu investigación, por favor cita el artículo original:

#### Fragmento de código
@article{baldenegro2026efecto,
  title={Efecto de los métodos de aumento de datos en MobileNetV2 para el diagnóstico de enfermedades foliares en hortalizas de Sinaloa},
  author={Baldenegro Gómez, Jared Eliezer and Rodríguez Rangel, Héctor and González Huitrón, Víctor Alejandro},
  journal={International Journal of Combinatorial Optimization Problems and Informatics (IJISTA)},
  year={2026}
}

