# Visão Geral do Projeto
Em 2000, Enron era uma das maiores empresas dos Estados Unidos.
Já em 2002, ela colapsou e quebrou devido a uma fraude que envolveu grande parte da corporação.
Resultando em uma investigação federal, muitos dados que são normalmente confidenciais, se tornaram públicos,
incluindo dezenas de milhares de e-mails e detalhes financeiros para os executivos dos mais altos níveis da empresa.
Neste projeto, você irá bancar o detetive, e colocar suas habilidades na construção de um modelo preditivo que visará
determinar se um funcionário é ou não um funcionário de interesse (POI).
Um funcionário de interesse é um funcionário que participou do escândalo da empresa Enron.
Para te auxiliar neste trabalho de detetive, nós combinamos os dados financeiros e sobre e-mails dos funcionários
investigados neste caso de fraude, o que significa que eles foram indiciados, fecharam acordos com o governo, ou
testemunharam em troca de imunidade no processo.

Você pode obter este projeto inicial usando o git: git clone https://github.com/udacity/ud120-projects.git

poi_id.py : Código inicial do identificar de pessoas de interesse (POI, do inglês Person of Interest).
É neste arquivo que você escreverá sua análise.
Você também enviará uma versão deste arquivo para que o avaliador verifique seu algoritmo e resultados. 

final_project_dataset.pkl : O conjunto de dados para o projeto. Veja mais detalhes abaixo. 

tester.py : Ao enviar sua análise para avaliação para o Udacity, você enviará o algoritmo, conjunto de dados,
e a lista de atributos que você utilizou (criados automaticamente pelo arquivo poi_id.py).

emails_by_address : Este diretório contém diversos arquivos de texto, cada um contendo todas as mensagens de ou
para um endereço de email específico. Estes dados estão aqui para referência, ou caso você deseje criar atributos
mais complexos baseando-se nos detalhes dos emails. Você não precisa processar estes dados para completar este projeto.

dos2Unix: Script para resolver problema de formatação com os dados originais do projeto.
