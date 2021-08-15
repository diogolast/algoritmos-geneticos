# Trabalho de Otimização - Mestrado em Inteligência Artificial
## O projeto apresenta solução de três problemas utilizando algorítmos genéticos com elitismo. (Informações em problems.pdf)
* Problema do caxeiro viajante (tsp)
* Problema de regressão
* Problema de classificação usando KNN


## Instale o pipenv
```
pip install --user pipenv
```

## Intale as depedências
```
pipenv shell
pip install
```

## Execução
### Cada problema é executado cinco vezes seguidas, os resultados são apresentados na pasta report

```
python main.py --problem tsp  --population_size 200 --n_generations 1000 --mutation_rate 0.1

python main.py --problem regression  --population_size 200 --n_generations 1000 --mutation_rate 0.1

python main.py --problem classification  --population_size 200 --n_generations 1000 --mutation_rate 0.1
```
