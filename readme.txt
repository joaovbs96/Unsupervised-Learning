João Vítor Buscatto Silva - RA155951
Tamara Martinelli de Campos - RA157324

MC886 2018 s2 - Assignment 3 - Unsupervised Learning

 - Cada modelo proposto nesse projeto foi implementado em seu próprio arquivo
fonte, executados pelos seguintes comandos:

# k-means gerando curva do cotovelo:
python kmeans.py bags.csv

# DBSCAN
Python dbscan.py bags.csv

# Melhor modelo encontrado e análise dos clusters gerados
python bestModel_Analysis.py bags.csv health.txt

# Análise dos medóides e dos vizinhos mais próximos do grupo
Python kmeans_medoids.py bags.csv health.txt

# Melhor modelo com redução de dimensionalidade
python pca.py bags.csv


- Efetuar o download do arquivo da base de dados, indicada no link do Dropbox fornecido no enunciado e replicado a seguir, e extrair na mesma pasta dos códigos fonte.

https://www.dropbox.com/s/ahkim9u103v0q9i/health-dataset.zip
