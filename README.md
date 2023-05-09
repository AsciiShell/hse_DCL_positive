# hse_DCL_positive

## Анализ смещения распределений в контрастивном обучении

- [datasets.py](datasets.py) - модифицированные версии датасетов STL10 и CIFAR10
- [loss.py](loss.py) - реализации функций потерь Contrastive, DebiasedNeg и DebiasedPos
- [model.py](model.py) - модифицированная версия модели ResNet
- [utils.py](utils.py) - утилиты, аугментации
- [main.py](main.py) - основная точка входа, содержит скрипты для обучения модели
- [main.ipynb](main.ipynb) - пример обучения модели в среде Jupyter Lab
- [plot_experiments.ipynb](plot_experiments.ipynb) - ноутбук с визуализацией результатов экспериментов


## Acknowledgements

Part of this code is inspired by [chingyaoc/DCL](https://github.com/chingyaoc/DCL).

---

**Проект выполнили:** [Алексей Подчезерцев](https://github.com/AsciiShell),
  [Мария Самоделкина](https://github.com/goo-goo-goo-joob)