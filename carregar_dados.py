## Função para carregar o dataset
import pandas as pd
import seaborn as sns

# Mapeamento utilizado para transformar as variáveis de string para int
iris_map = {"Iris-setosa":0,
            "Iris-versicolor":1,
            "Iris-virginica":2}

def carregar_dados(var_map=iris_map):
    """
    Função que carrega os dados e os separam em treinamento e teste.
    var_map: como default, o mapeamento é para o iris dataset
    
    A variável raw é utilizada apenas para gráficar o dataset.
    """

    dados = pd.read_csv("iris.data", sep=",", header=None)
    raw  = pd.read_csv("iris.data", sep=",", header=None)
    dados.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]
    raw.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]
    dados["class"] = dados["class"].map(var_map)
    
    return raw, dados

def plotar_iris(dados):
    """
    Função para plotar as variáveis do dataset iris.
    dados: tipo pandas.DataFrame com o dataset carregado
    """
    sns.set(style="ticks")

    sns.pairplot(dados, hue="class")

def preparar_iris(dados, train_fraction):
    """
    Prepara os dados para treinamento.
    Separa-os entre treinamento e teste.
    
    dados: Sempre no formato DataFrame.
    train_fraction: A fração a ser usada no treinamento, de 0 a 100.
    """
    
    # 50 de cada flor
    setosa = dados[0:50]
    versicolor = dados[50:100]
    virginica = dados[100:150]

    fracao = (len(setosa) * train_fraction)//100

    train = pd.concat([setosa[0:fracao], versicolor[0:fracao], virginica[0:fracao]])
    test = pd.concat([setosa[fracao:50], versicolor[fracao:50], virginica[fracao:50]])

    train_y = pd.get_dummies(train["class"], prefix="class").to_numpy()
    train_x = train.drop(columns=["class"]).to_numpy()

    test_y = pd.get_dummies(test["class"], prefix="class").to_numpy()
    test_x = test.drop(columns=["class"]).to_numpy()

    return train_x, train_y, test_x, test_y

if __name__ == "__main__":
    raw, dados = carregar_dados()
    print(dados.sample(n = 5))
    print("N de dados:", len(dados))
    plotar_iris(raw)
    train_x, train_y, test_x, test_y = preparar_iris(dados, 50)