pip install scikit-learn numpy

# Importar bibliotecas necessárias
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados Iris
iris = datasets.load_iris()
X = iris.data  # Dados de entrada (características)
y = iris.target  # Rótulos (classes)

# Dividir os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o classificador SVM
clf = SVC(kernel='linear')  # Usando um kernel linear

# Treinar o classificador com os dados de treinamento
clf.fit(X_train, y_train)

# Fazer previsões com o conjunto de teste
y_pred = clf.predict(X_test)

# Avaliar o desempenho do classificador
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Fazer uma previsão com novos dados (exemplo)
new_data = [[5.1, 3.5, 1.4, 0.2]]  # Dados de exemplo
prediction = clf.predict(new_data)
print(f'Prediction for new data: {iris.target_names[prediction][0]}')
