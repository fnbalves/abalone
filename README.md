Dependencias:

pandas
numpy
matplotlib
scikit-learn
scipy
imbalanced-learn

Instale o Python 2.7, configure o pip e execute o comando

sudo pip install -r requirements.txt

para instalar todas as dependencias

Descricao dos arquivos:

data - pasta com os arquivos originais do UCI
docs - pasta com o enunciado do projeto
result_images - pasta com as imagens geradas ao longo do estudo
result_pickles - pasta com arquivos pickle resultantes
CombinedClassifier.py - classificador combinado com aprendizado no espaco de saidas
compare_all_classifiers.py - compara a acuracia dos classificadores em um teste pareado
compare_all_classifiers_plus_combined.py - compara o mlp com os classificadores combinados
compare_all_classifiers_time.py - compara os classificadores em relacao ao tempo de processamento
covariance_all_classifiers.py - calcula a covariancia entre as saidas dos classificadores
fatures.pickle - mapeamento de features aleatorias salvo
investigate_data.py - cria os scatter plots dos dados iniciais (ao executar, nao mostra os graficos, salva diretamente na pasta result_images)
(os demais arquivos com graficos exibem o resultado ao fim da execucao)
MajorityVoteClassifier.py - classificador combinado voto majoritario
make_friedman_and_nemenyi_test.py - faz os testes estatisticos
MLP.py - rede neural implementada
random_tansformations.py - arquivo que cria os mapeamentos aleatorios de features
requirements.txt - arquivo de dependencias para o pip
test_knn_dist_weight.py - arquivo que testa as ponderacoes do knn
test_knn_num_bins.py - arquivo que testa no numero de bins com um knn de 7 vizinhos
test_knn_y_conv_accuracy.py - testa o numero de vizinhos usando acuracia como medida
test_knn_y_conv_f1_score.py - testa o numero de vizinhos usando o f1 score como medida
test_knn_y_conv_inbalance.py - testa o wilson editing
test_mlp_activation.py - testa as funcoes de ativacao da mlp
test_mlp_learning_rates.py - testa as taxas de aprendizado da mlp
test_mlp_max_iter.py - testa o numero maximo de iteracoes na mlp
test_mlp_num_layers.py - testa o numero de camadas ocultas na mlp
test_mlp_regularization.py - testa o termo de regularizacao na mlp
test_mlp_y_conv_f1_score.py - testa o tamanho da camada oculta na mlp utilizando o f1 score como medida
test_self_mlp.py - testa a mlp implementada
test_svm_y_conv_f1_score.py - testa o valor de C na svm
test_svm_kernel.py - testa o kernel e o valor de C da svm
test_tree_max_depth.py - testa a profundidade maxima da arvore
test_tree_max_leaf.py - testa o maximo numero de nos folha da arvore
test_tree_metric.py - testa o tipo de metrica usada na arvore
util.py - funcoes comuns usadas por muitos arquivos
